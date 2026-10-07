#!/usr/bin/env python3
"""Produce a leaderboard submission parquet from a trained LightGBM model.

Reusable across all submission versions (v1, v2, ...). Loads the model,
scores ranking DEP rows, joins predictions to submitting.parquet MVT_IDs,
and validates output integrity before saving.

Usage:
    python src/make_submission.py                        # default: baseline model -> v1
    python src/make_submission.py --version 2            # output v2
    python src/make_submission.py --model models/lgbm_v2.txt --version 2

Categorical vocabulary (Task 05 finding).
    pdc26.features.build_features() derives categorical codes from whatever
    DataFrame it is handed (pandas .astype("category").cat.codes is dataframe-
    local). Scoring the ranking file without a shared vocabulary feeds the
    booster relabelled RUNWAY/STAND/AIRCRAFT_TYPE/OPERATOR codes. This script
    therefore fits the vocabulary on the TRAINING data once, persists it next
    to the booster, and applies it at score time:

        python src/make_submission.py --fit-vocab         # build + verify + save

    Scoring requires the vocabulary file; pass --no-vocab to opt out.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import polars as pl

# Ensure src/ is on the path when running as a script
sys.path.insert(0, os.path.dirname(__file__))

from pdc26.data import load_ranking, load_training
from pdc26.features import (
    CAT_FEATURES,
    UNSEEN_CODE,
    apply_categorical_vocab,
    build_features,
    fit_categorical_vocab,
    load_vocab,
    persist_vocab,
)
from pdc26.train import load_model, score_ranking

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("make_submission")

# --- Project paths ---
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_RAW = os.path.join(_PROJECT_ROOT, "data", "raw")
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
SUBMISSIONS_DIR = os.path.join(_PROJECT_ROOT, "submissions")

# --- Validation bounds ---
VALID_LO = 1.0
VALID_HI = None  # No upper bound — the [0,7200] clip harmed raw RMSE by ~42s

def vocab_path_for(model_path: str) -> str:
    """Sidecar path where a model's categorical vocabulary is persisted."""
    base = os.path.splitext(os.path.basename(model_path))[0]
    return os.path.join(MODELS_DIR, f"{base}_vocab.json")


def fit_vocab(model_path: str) -> str:
    """Derive the categorical vocabulary from training data and persist it.

    Thin wrapper around :func:`pdc26.features.fit_categorical_vocab` and
    :func:`pdc26.features.persist_vocab`. Self-verification is handled
    inside the library function.

    Args:
        model_path: Booster whose scoring path will consume this vocabulary.

    Returns:
        Path to the written vocabulary JSON.
    """
    train_df = load_training()
    vocab = fit_categorical_vocab(train_df)
    out = vocab_path_for(model_path)
    persist_vocab(vocab, out)
    return out


def apply_vocab(X_rank: pd.DataFrame, rank_df: pl.DataFrame, vocab: dict) -> dict:
    """Relabel ranking categorical codes onto the training vocabulary.

    Thin wrapper around :func:`pdc26.features.apply_categorical_vocab`.

    Args:
        X_rank: Feature matrix from build_features(rank_df) — mutated in place.
        rank_df: Source frame, same row order as X_rank.
        vocab: Mapping of column -> ordered value list from fit_vocab().

    Returns:
        Dict of column -> count of values not present in the training vocabulary.
    """
    return apply_categorical_vocab(X_rank, rank_df, vocab)


def main():
    parser = argparse.ArgumentParser(description="Produce leaderboard submission parquet.")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to LightGBM model file (default: models/baseline_lgbm.txt)."
    )
    parser.add_argument(
        "--version", type=int, default=1,
        help="Submission version number (default: 1)."
    )
    parser.add_argument(
        "--fit-vocab", action="store_true",
        help="Build + verify + persist the categorical vocabulary, then exit."
    )
    parser.add_argument(
        "--vocab", type=str, default=None,
        help="Path to vocabulary JSON (default: <model>_vocab.json beside the model)."
    )
    parser.add_argument(
        "--no-vocab", action="store_true",
        help="Score with dataframe-local codes (reproduces the v1 skew; not advised)."
    )
    args = parser.parse_args()

    model_path = args.model or os.path.join(MODELS_DIR, "baseline_lgbm.txt")

    if args.fit_vocab:
        fit_vocab(model_path)
        return

    t0 = time.time()
    log.info("=" * 60)
    log.info(f"PRC-OpenSky 2026 — Making Submission v{args.version}")
    log.info("=" * 60)

    # --- 1. Load model ---
    log.info("Step 1: Loading model...")
    model = load_model(model_path)

    # --- 1b. Resolve categorical vocabulary (train/serve skew guard) ---
    vocab = None
    if not args.no_vocab:
        vpath = args.vocab or vocab_path_for(model_path)
        if not os.path.exists(vpath):
            raise SystemExit(
                f"ERROR: no categorical vocabulary at {vpath}.\n"
                f"       Scoring without it relabels categoricals vs the trained "
                f"mapping (the v1 bug). Run:\n"
                f"         python {os.path.basename(__file__)} --fit-vocab --model {model_path}\n"
                f"       or pass --no-vocab to proceed anyway."
            )
        log.info("Step 1b: Loading categorical vocabulary %s", vpath)
        vocab = load_vocab(vpath)
    else:
        log.warning("--no-vocab: scoring with dataframe-local codes (skew not guarded)")

    # --- 2. Load and featurize ranking data ---
    log.info("Step 2: Loading ranking DEP data...")
    rank_df = load_ranking()

    log.info("Step 3: Building features...")
    X_rank, _ = build_features(rank_df)
    X_rank.index = range(len(X_rank))

    if vocab is not None:
        log.info("Step 3b: Rebinding ranking categoricals to the training vocabulary...")
        unseen = apply_vocab(X_rank, rank_df, vocab)
        n_unseen_total = sum(unseen.values())
        log.info("  Total unseen categorical cells: %s", f"{n_unseen_total:,}")

    # --- 3. Score ---
    log.info("Step 4: Scoring ranking DEP rows...")
    rank_mvt_ids = rank_df["MVT_ID_mvt"].to_pandas()
    rank_mvt_ids.index = range(len(rank_mvt_ids))
    predictions = score_ranking(model, X_rank, rank_mvt_ids)
    log.info("Predictions shape: %s", predictions.shape)

    # --- 4. Join to submitting.parquet ---
    log.info("Step 5: Joining predictions to submitting.parquet...")
    submitting = pl.read_parquet(os.path.join(DATA_RAW, "submitting.parquet"))
    sub_ids = submitting["MVT_ID_mvt"].to_pandas()

    # Build prediction lookup keyed by MVT_ID
    pred_map = predictions.set_index("MVT_ID_mvt")["predicted_taxi_sec"]

    # Align: for each MVT_ID in submitting, get its prediction
    submission_values = sub_ids.map(pred_map)

    # --- 5. Assertions ---
    log.info("Step 6: Running validation assertions...")

    # (a) Row count matches
    assert len(submission_values) == submitting.height, (
        f"Row count mismatch: {len(submission_values)} vs {submitting.height}"
    )
    log.info("  ✓ Row count matches submitting.parquet: %s", f"{len(submission_values):,}")

    # (b) No nulls (all MVT_IDs matched)
    n_null = submission_values.isna().sum()
    assert n_null == 0, (
        f"{n_null} MVT_IDs in submitting.parquet have no matching prediction!"
    )
    log.info("  ✓ No null predictions")

    # (c) All values finite
    vals = submission_values.values.astype("float64")
    n_nonfinite = (~np.isfinite(vals)).sum()
    assert n_nonfinite == 0, (
        f"{n_nonfinite} non-finite values in predictions!"
    )
    log.info("  ✓ All values finite")

    # (d) Range check + clipping (floor at 1s only, no upper bound)
    n_below = (vals < VALID_LO).sum()
    if n_below > 0:
        log.warning(
            "  ⚠ Flooring %d values below %.0f",
            n_below, VALID_LO,
        )
        vals = np.maximum(vals, VALID_LO)
    log.info("  ✓ All values >= %.0f s (no upper clip)", VALID_LO)

    # Summary stats
    if VALID_HI is not None:
        n_above = (vals > VALID_HI).sum()
        if n_above > 0:
            log.warning("  ⚠ %d values above %.0f (unguarded)", n_above, VALID_HI)
    log.info("  Prediction stats: mean=%.1f  median=%.1f  min=%.1f  max=%.1f  std=%.1f",
             vals.mean(), np.median(vals), vals.min(), vals.max(), vals.std())

    # --- 6. Save output ---
    log.info("Step 7: Saving submission...")
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    out_filename = f"thoughtful-glove_v{args.version}.parquet"
    out_path = os.path.join(SUBMISSIONS_DIR, out_filename)

    # Build output DataFrame matching submitting.parquet schema
    output_df = pd.DataFrame({
        "MVT_ID_mvt": sub_ids.values,
        "TAXITIME_SEC_mvt": vals,
    })
    output_df.to_parquet(out_path, index=False)
    log.info("  Saved: %s (%s rows)", out_path, f"{len(output_df):,}")

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("SUBMISSION READY in %.1fs", elapsed)
    log.info("  File: submissions/%s", out_filename)
    log.info("  Upload: mc cp %s opensky/prc-2026-thoughtful-glove/%s",
             out_path, out_filename)
    log.info("=" * 60)


if __name__ == "__main__":
    main()