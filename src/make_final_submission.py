#!/usr/bin/env python3
"""Build the final-phase submission (Task 18) from `final_ranking.parquet`.

Pipeline (frozen v5 model stack, no retraining):

  1. Load full 2025 training DEP data and fit the serving artifacts exactly as
     `run_blend.py::build_ranking_features` does: categorical vocabularies
     (`fit_d4_vocabs`), the E-winsorized target (`winsorize_errors`), and the
     full-train operator target encoding (`fit_operator_target_encoding`).
  2. Load `final_ranking.parquet` (DEP rows for scoring; DEP+ARR movements for
     congestion features) and build the 63-feature D4 matrix with the
     training-fitted vocabularies and operator mapping.
  3. Load the frozen saved models from `models/blend/` — `tuned_lgbm_full.txt`
     plus ten `per_airport_lgbm_<AP>_full.txt` — and score them.
  4. Blend with the v5 weights (`src/configs/blend.json`), apply the accepted
     per-airport affine recalibration (`mi2_affine_s50` from
     `src/configs/blend_postproc.json`), floor at 1 s (no upper clip).
  5. Write `submissions/final/thoughtful-glove_final_v1.parquet` in
     `final_submitting.parquet` MVT_ID order and a per-component side table for
     the validation battery.

No model retraining, no feature changes, no recalibration refits: the model
stack is the frozen v5 stack, loaded from disk. `run_blend.py` and `pdc26/`
are untouched, keeping the v5 reproduction path byte-stable.

Usage:
    uv run python src/make_final_submission.py              # build final_v1
    uv run python src/make_final_submission.py --emit-v6    # v6 from side table
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import polars as pl

from pdc26.airports import load_coords
from pdc26.data import load_training
from pdc26.features import (
    apply_operator_target_encoding,
    fit_operator_target_encoding,
    get_month_labels,
)
from pdc26.tuning import build_d4_features, fit_d4_vocabs, load_metar, winsorize_errors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("make_final_submission")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_RAW = os.path.join(_PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models", "blend")
SUBMISSIONS_DIR = os.path.join(_PROJECT_ROOT, "submissions")
FINAL_DIR = os.path.join(SUBMISSIONS_DIR, "final")
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")

FINAL_RANKING = os.path.join(DATA_RAW, "final_ranking.parquet")
FINAL_SUBMITTING = os.path.join(DATA_RAW, "final_submitting.parquet")
RANKING_SUBMITTING = os.path.join(DATA_RAW, "submitting.parquet")
FINAL_V1 = os.path.join(FINAL_DIR, "thoughtful-glove_final_v1.parquet")
FINAL_V2 = os.path.join(FINAL_DIR, "thoughtful-glove_final_v2.parquet")
SIDE_TABLE = os.path.join(DATA_PROCESSED, "final_predictions_v1.parquet")
SIDE_TABLE_V2 = os.path.join(DATA_PROCESSED, "final_predictions_v2.parquet")
V6_PATH = os.path.join(SUBMISSIONS_DIR, "thoughtful-glove_v6.parquet")

BLEND_CONFIG = os.path.join(CONFIG_DIR, "blend.json")
POSTPROC_CONFIG = os.path.join(CONFIG_DIR, "blend_postproc.json")
POSTPROC_CANDIDATE = "mi2_affine_s50"
HARDROW_CONFIG = os.path.join(CONFIG_DIR, "hardrow.json")
HARDROW_SERVING = os.path.join(MODELS_DIR, "hardrow_lgbm_full.txt")

# Expected final-file row counts (Task 17 verification report).
EXPECTED_DEP_ROWS = 670_790
EXPECTED_V6_ROWS = 344_841
EXPECTED_HARD_ROWS = 9_077


def load_final_ranking() -> pl.DataFrame:
    """Load `final_ranking.parquet` filtered to DEP, sorted by time.

    Mirrors `pdc26.data.load_ranking` but for the final-phase file.
    """
    if not os.path.exists(FINAL_RANKING):
        raise FileNotFoundError(f"final ranking parquet not found: {FINAL_RANKING}")
    df = (
        pl.scan_parquet(FINAL_RANKING)
        .filter(pl.col("PHASE_mvt") == "DEP")
        .sort("MVT_TIME_UTC_mvt")
        .collect()
    )
    log.info("Final ranking DEP: %s rows", f"{df.height:,}")
    return df


def load_final_ranking_movements() -> pl.DataFrame:
    """Load `final_ranking.parquet` with BOTH phases (congestion features)."""
    if not os.path.exists(FINAL_RANKING):
        raise FileNotFoundError(f"final ranking parquet not found: {FINAL_RANKING}")
    df = pl.read_parquet(FINAL_RANKING).sort("MVT_TIME_UTC_mvt")
    log.info("Final ranking movements: %s rows (DEP+ARR)", f"{df.height:,}")
    return df


def cast_features(X: pd.DataFrame) -> pd.DataFrame:
    """Cast numeric feature columns to float32 (matches the v5 serve path)."""
    for col in X.select_dtypes(include=["float64", "int64"]).columns:
        X[col] = X[col].astype("float32")
    return X


class ServingArtifacts:
    """Training-fitted pieces required to build final features (no models)."""

    def __init__(self) -> None:
        log.info("Loading full 2025 training DEP data...")
        dep = load_training()
        pdf_full = dep.to_pandas()
        raw_target = dep["TAXITIME_SEC_mvt"].to_pandas().astype("float64")
        airports = dep["ADEP_mvt"].to_pandas().astype("string").fillna("__NA__")
        for s in (raw_target, airports):
            s.index = range(len(s))

        log.info("Fitting D4 vocabularies on full training...")
        self.vocab, self.domain_vocab = fit_d4_vocabs(dep)
        self.y_winsor = winsorize_errors(raw_target, airports)
        self.operator_enc = fit_operator_target_encoding(
            pdf_full["AIRCRAFT_OPERATOR_flt"], self.y_winsor)
        self.op_fallback = float(self.y_winsor.mean())
        self.metar = load_metar()
        self.coords = load_coords()


def build_final_features(art: ServingArtifacts) -> tuple[pl.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the 63-feature D4 matrix for the final ranking DEP rows."""
    rank_df = load_final_ranking()
    if rank_df["MVT_ID_mvt"].n_unique() != rank_df.height:
        raise ValueError("final ranking DEP rows contain duplicate MVT_ID_mvt")
    rank_movements = load_final_ranking_movements().to_pandas()
    rank_pdf = rank_df.to_pandas()

    log.info("Applying full-train operator target encoding to final rows...")
    op_rank = apply_operator_target_encoding(
        rank_pdf["AIRCRAFT_OPERATOR_flt"], art.operator_enc, fallback=art.op_fallback)

    log.info("Building D4 final features...")
    X = build_d4_features(
        rank_df, rank_movements, art.vocab, art.domain_vocab,
        art.metar, art.coords, operator_enc=op_rank,
    )
    X = cast_features(X)
    X.index = range(len(X))
    log.info("Final feature matrix: %s rows x %d cols", f"{len(X):,}", X.shape[1])
    return rank_df, rank_pdf, X


def load_frozen_models():
    """Load the v5 frozen models: one global + ten per-airport LightGBM boosters."""
    import lightgbm as lgb

    tuned_path = os.path.join(MODELS_DIR, "tuned_lgbm_full.txt")
    if not os.path.exists(tuned_path):
        raise FileNotFoundError(f"frozen global model not found: {tuned_path}")
    tuned = lgb.Booster(model_file=tuned_path)

    per_airport: dict[str, object] = {}
    for path in sorted(glob.glob(os.path.join(MODELS_DIR, "per_airport_lgbm_*_full.txt"))):
        stem = os.path.basename(path)
        apt = stem[len("per_airport_lgbm_"):-len("_full.txt")]
        per_airport[apt] = lgb.Booster(model_file=path)
    if len(per_airport) != 10:
        raise ValueError(f"expected 10 per-airport models, found {len(per_airport)}")
    log.info("Loaded frozen models: 1 global + %d per-airport", len(per_airport))
    return tuned, per_airport


def check_feature_gate(tuned, per_airport: dict, X: pd.DataFrame) -> list[str]:
    """Assert the feature matrix matches the saved models' feature names exactly."""
    names = tuned.feature_name()
    missing = [c for c in names if c not in X.columns]
    extra = [c for c in X.columns if c not in names]
    if missing or extra:
        raise ValueError(f"feature gate failed: missing={missing} extra={extra}")
    for apt, model in per_airport.items():
        if model.feature_name() != names:
            raise ValueError(f"per-airport model {apt} feature names differ from global")
    log.info("Feature gate PASS: %d features match all saved models", len(names))
    return names


def score_components(tuned, per_airport: dict, X: pd.DataFrame,
                     rank_pdf: pd.DataFrame) -> dict[str, np.ndarray]:
    """Score the global and per-airport models on the final feature matrix."""
    rank_airport = rank_pdf["ADEP_mvt"].astype("string").fillna("__NA__").values
    preds: dict[str, np.ndarray] = {"tuned_lgbm": tuned.predict(X)}

    per = np.full(len(X), np.nan, dtype="float64")
    for apt, model in per_airport.items():
        mask = rank_airport == apt
        if mask.any():
            per[mask] = model.predict(X.loc[mask].reset_index(drop=True))
    n_missing = int(np.isnan(per).sum())
    if n_missing:
        raise ValueError(f"{n_missing} final rows have no per-airport model")
    preds["per_airport_lgbm"] = per
    for name, p in preds.items():
        log.info("  %-17s mean=%.1f median=%.1f min=%.1f max=%.1f",
                 name, p.mean(), np.median(p), p.min(), p.max())
    return preds


def blend_and_recalibrate(preds: dict[str, np.ndarray], rank_pdf: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Apply the v5 blend weights then the accepted per-airport affine."""
    with open(BLEND_CONFIG) as f:
        blend = json.load(f)
    weights = blend["selected"]["active_weights"]
    log.info("Blend weights: %s", {k: round(v, 4) for k, v in weights.items()})
    blended = np.zeros(len(rank_pdf), dtype="float64")
    for name, w in weights.items():
        blended += float(w) * preds[name]

    with open(POSTPROC_CONFIG) as f:
        pp = json.load(f)
    spec = pp["full_fit"][POSTPROC_CANDIDATE]
    if spec.get("type") != "affine":
        raise ValueError(f"unexpected postproc type for {POSTPROC_CANDIDATE}: {spec.get('type')}")
    affine = {k: tuple(v) for k, v in spec["airport_affine"].items()}
    slope_map = {k: v[0] for k, v in affine.items()}
    inter_map = {k: v[1] for k, v in affine.items()}
    rank_airport = rank_pdf["ADEP_mvt"].astype("string").fillna("__NA__")
    slopes = rank_airport.map(slope_map).fillna(1.0).to_numpy("float64")
    inters = rank_airport.map(inter_map).fillna(0.0).to_numpy("float64")
    final = slopes * blended + inters
    final = np.maximum(final, 1.0)
    log.info("Post-affine predictions: mean=%.1f median=%.1f min=%.1f max=%.1f",
             final.mean(), np.median(final), final.min(), final.max())
    return blended, final


def write_final_v1(final: np.ndarray, rank_df: pl.DataFrame,
                   rank_pdf: pd.DataFrame, preds: dict[str, np.ndarray],
                   blended: np.ndarray) -> str:
    """Write the final-phase submission and the per-component side table."""
    submitting = pl.read_parquet(FINAL_SUBMITTING)
    sub_ids = submitting["MVT_ID_mvt"].to_pandas()
    rank_ids = rank_df["MVT_ID_mvt"].to_pandas()
    if not rank_ids.is_unique:
        raise ValueError("final ranking MVT_IDs are not unique")

    pred_map = pd.Series(final, index=rank_ids.values)
    vals = sub_ids.map(pred_map).to_numpy("float64")

    # --- Pre-upload assertions (mandatory before the artifact is trusted) ---
    assert len(vals) == submitting.height == EXPECTED_DEP_ROWS, \
        f"row count mismatch: {len(vals)} vs {submitting.height} (expected {EXPECTED_DEP_ROWS})"
    assert not np.isnan(vals).any(), "null predictions after join"
    assert np.isfinite(vals).all(), "non-finite predictions"
    assert vals.min() >= 1.0, f"prediction below 1 s floor: {vals.min()}"
    if not np.array_equal(sub_ids.to_numpy(), submitting["MVT_ID_mvt"].to_numpy()):
        raise ValueError("MVT_ID order is not exact vs final_submitting")
    log.info("PRE-UPLOAD ASSERTIONS PASS: rows=%s nulls=0 finite=True min=%.3f "
             "order_exact=True", f"{len(vals):,}", vals.min())

    os.makedirs(FINAL_DIR, exist_ok=True)
    out = pd.DataFrame({"MVT_ID_mvt": sub_ids.to_numpy(), "TAXITIME_SEC_mvt": vals})
    out.to_parquet(FINAL_V1, index=False)

    months = get_month_labels(rank_df).values
    side = pd.DataFrame({
        "MVT_ID_mvt": rank_ids.values,
        "month": months,
        "ADEP_mvt": rank_pdf["ADEP_mvt"].astype("string").values,
        "operator_null": rank_pdf["AIRCRAFT_OPERATOR_flt"].isna().values,
        "pred_tuned_lgbm": preds["tuned_lgbm"],
        "pred_per_airport_lgbm": preds["per_airport_lgbm"],
        "blended": blended,
        "final_value": final,
    })
    side.to_parquet(SIDE_TABLE, index=False)

    sha = hashlib.sha256(open(FINAL_V1, "rb").read()).hexdigest()
    log.info("Saved %s (%s rows)", FINAL_V1, f"{len(out):,}")
    log.info("Saved side table %s", SIDE_TABLE)
    log.info("SHA256 %s", sha)
    log.info("Prediction stats: mean=%.1f median=%.1f min=%.1f max=%.1f std=%.1f",
             vals.mean(), np.median(vals), vals.min(), vals.max(), vals.std())
    log.info("Upload (Task 20): uv run --with minio python src/minio_transfer.py up %s "
             "opensky/prc-2026-thoughtful-glove/", FINAL_V1)
    return FINAL_V1


def emit_v6() -> str:
    """Restrict the validated final predictions to Jan/Jul rows for v6 validation."""
    if not os.path.exists(SIDE_TABLE):
        raise SystemExit(f"side table not found: {SIDE_TABLE}; run the build first")
    side = pd.read_parquet(SIDE_TABLE)
    submitting = pl.read_parquet(RANKING_SUBMITTING)
    sub_ids = submitting["MVT_ID_mvt"].to_pandas()

    final_ids = set(side["MVT_ID_mvt"].to_numpy())
    sub_set = set(sub_ids.to_numpy())
    janjul = side[side["month"].isin(["2026-01", "2026-07"])]["MVT_ID_mvt"]
    if len(janjul) != EXPECTED_V6_ROWS or set(janjul.to_numpy()) != sub_set:
        raise ValueError(
            f"Jan/Jul ID set mismatch vs submitting.parquet: final Jan/Jul={len(janjul)}, "
            f"submitting={len(sub_set)}, equal={set(janjul.to_numpy()) == sub_set}")
    if not sub_set.issubset(final_ids):
        raise ValueError("submitting.parquet IDs are not a subset of the final predictions")

    pred_map = pd.Series(side["final_value"].to_numpy(), index=side["MVT_ID_mvt"].to_numpy())
    vals = sub_ids.map(pred_map).to_numpy("float64")

    assert len(vals) == submitting.height == EXPECTED_V6_ROWS, \
        f"v6 row count mismatch: {len(vals)} vs {submitting.height}"
    assert not np.isnan(vals).any(), "v6 null predictions after join"
    assert np.isfinite(vals).all(), "v6 non-finite predictions"
    assert vals.min() >= 1.0, f"v6 prediction below 1 s floor: {vals.min()}"
    log.info("V6 ASSERTIONS PASS: rows=%s ID set equality=True nulls=0 finite=True min=%.3f",
             f"{len(vals):,}", vals.min())

    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    out = pd.DataFrame({"MVT_ID_mvt": sub_ids.to_numpy(), "TAXITIME_SEC_mvt": vals})
    out.to_parquet(V6_PATH, index=False)
    sha = hashlib.sha256(open(V6_PATH, "rb").read()).hexdigest()
    log.info("Saved %s (%s rows) SHA256 %s", V6_PATH, f"{len(out):,}", sha)
    log.info("Prediction stats: mean=%.1f median=%.1f min=%.1f max=%.1f",
             vals.mean(), np.median(vals), vals.min(), vals.max())
    log.info("Upload: uv run --with minio python src/minio_transfer.py up %s "
             "opensky/prc-2026-thoughtful-glove/", V6_PATH)
    return V6_PATH


def build() -> str:
    t0 = time.time()
    art = ServingArtifacts()
    rank_df, rank_pdf, X = build_final_features(art)
    tuned, per_airport = load_frozen_models()
    names = check_feature_gate(tuned, per_airport, X)
    X = X[names]
    preds = score_components(tuned, per_airport, X, rank_pdf)
    blended, final = blend_and_recalibrate(preds, rank_pdf)
    path = write_final_v1(final, rank_df, rank_pdf, preds, blended)
    log.info("Build done in %.0fs (%.1f min)", time.time() - t0, (time.time() - t0) / 60)
    return path


def build_hardrow_v2() -> str:
    """Build final_v2 = final_v1 with the operator-null (hard) rows replaced by the
    Task 19 dedicated hard-row model (pre-registered protocol, gate ACCEPTED).

    Base rows are taken verbatim from final_v1, so their predictions are byte-identical
    by construction. `run_blend.py`/`pdc26/` and the default v1 path are untouched.
    """
    import lightgbm as lgb

    t0 = time.time()
    if not os.path.exists(HARDROW_SERVING) or not os.path.exists(HARDROW_CONFIG):
        raise SystemExit(f"hard-row serving artifacts missing: run `src/run_hardrow_experiment.py "
                         f"--phase serve-train` first ({HARDROW_SERVING})")
    with open(HARDROW_CONFIG) as f:
        cfg = json.load(f)
    model = lgb.Booster(model_file=HARDROW_SERVING)
    feat = list(cfg["feature_order"])
    log.info("Hard-row serving model: %d trees, %d features, rounds=%s, hard-row training n=%s",
             model.num_trees(), len(feat), cfg.get("rounds"), cfg.get("n_train_hard_rows"))
    if list(model.feature_name()) != feat:
        raise ValueError("hardrow config feature_order disagrees with the saved model")

    art = ServingArtifacts()
    rank_df, rank_pdf, X = build_final_features(art)
    missing = [c for c in feat if c not in X.columns]
    extra = [c for c in X.columns if c not in feat]
    if missing or extra or list(X[feat].columns) != feat:
        raise ValueError(f"hardrow feature-order gate failed: missing={missing} extra={extra}")

    # v1 values in ranking order (base rows are taken verbatim from this).
    v1 = pd.read_parquet(FINAL_V1)
    rank_ids = rank_df["MVT_ID_mvt"].to_pandas()
    v1_map = pd.Series(v1["TAXITIME_SEC_mvt"].to_numpy(), index=v1["MVT_ID_mvt"].to_numpy())
    v1_rank = rank_ids.map(v1_map).to_numpy("float64")
    if np.isnan(v1_rank).any():
        raise ValueError("some final ranking rows are absent from final_v1")

    hard = rank_pdf["AIRCRAFT_OPERATOR_flt"].isna().to_numpy()
    if int(hard.sum()) != EXPECTED_HARD_ROWS:
        raise ValueError(f"expected {EXPECTED_HARD_ROWS} hard final rows, found {int(hard.sum())}")
    pred_hard = np.maximum(model.predict(X.loc[hard, feat]), 1.0)
    log.info("Hard-row model predictions: mean=%.1f median=%.1f min=%.1f max=%.1f",
             pred_hard.mean(), np.median(pred_hard), pred_hard.min(), pred_hard.max())
    log.info("v1 on the same hard rows:   mean=%.1f median=%.1f min=%.1f max=%.1f",
             v1_rank[hard].mean(), np.median(v1_rank[hard]), v1_rank[hard].min(),
             v1_rank[hard].max())

    final_v2 = v1_rank.copy()
    final_v2[hard] = pred_hard
    # Base rows byte-identical by construction; assert it explicitly.
    if not np.array_equal(final_v2[~hard], v1_rank[~hard]):
        raise ValueError("base rows changed — treatment must be row-restricted")
    log.info("BASE-ROW INVARIANCE PASS: %s base rows byte-identical to final_v1",
             f"{int((~hard).sum()):,}")

    submitting = pl.read_parquet(FINAL_SUBMITTING)
    sub_ids = submitting["MVT_ID_mvt"].to_pandas()
    pred_map = pd.Series(final_v2, index=rank_ids.values)
    vals = sub_ids.map(pred_map).to_numpy("float64")
    assert len(vals) == submitting.height == EXPECTED_DEP_ROWS, \
        f"row count mismatch: {len(vals)} vs {submitting.height}"
    assert not np.isnan(vals).any(), "null predictions after join"
    assert np.isfinite(vals).all(), "non-finite predictions"
    assert vals.min() >= 1.0, f"prediction below 1 s floor: {vals.min()}"
    if not np.array_equal(sub_ids.to_numpy(), submitting["MVT_ID_mvt"].to_numpy()):
        raise ValueError("MVT_ID order is not exact vs final_submitting")

    os.makedirs(FINAL_DIR, exist_ok=True)
    out = pd.DataFrame({"MVT_ID_mvt": sub_ids.to_numpy(), "TAXITIME_SEC_mvt": vals})
    out.to_parquet(FINAL_V2, index=False)

    months = get_month_labels(rank_df).values
    side = pd.DataFrame({
        "MVT_ID_mvt": rank_ids.values, "month": months,
        "ADEP_mvt": rank_pdf["ADEP_mvt"].astype("string").values,
        "operator_null": hard, "v1_value": v1_rank, "v2_value": final_v2,
        "delta": final_v2 - v1_rank,
    })
    side.to_parquet(SIDE_TABLE_V2, index=False)

    sha = hashlib.sha256(open(FINAL_V2, "rb").read()).hexdigest()
    log.info("Saved %s (%s rows) SHA256 %s", FINAL_V2, f"{len(out):,}", sha)
    log.info("Changed rows: %s (all operator-null); mean Δ %+.1f s on those rows",
             f"{int(hard.sum()):,}", final_v2[hard].mean() - v1_rank[hard].mean())
    log.info("Prediction stats: mean=%.1f median=%.1f min=%.1f max=%.1f std=%.1f",
             vals.mean(), np.median(vals), vals.min(), vals.max(), vals.std())
    log.info("Build v2 done in %.0fs (%.1f min)", time.time() - t0, (time.time() - t0) / 60)
    return FINAL_V2




def main() -> None:
    parser = argparse.ArgumentParser(description="Task 18 final-phase submission build")
    parser.add_argument("--emit-v6", action="store_true",
                        help="Emit v6 (Jan/Jul restriction) from the existing side table")
    parser.add_argument("--hardrow-v2", action="store_true",
                        help="Build final_v2 (Task 19 hard-row swap; gate ACCEPTED)")
    args = parser.parse_args()

    os.makedirs(DATA_PROCESSED, exist_ok=True)
    if args.emit_v6:
        emit_v6()
    elif args.hardrow_v2:
        build_hardrow_v2()
    else:
        build()


if __name__ == "__main__":
    main()
