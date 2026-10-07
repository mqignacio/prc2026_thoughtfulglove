#!/usr/bin/env python3
"""End-to-end baseline pipeline: load → features → 12-fold CV → report → model → score.

Usage:
    python src/run_baseline.py              # full run (all 12 months)
    python src/run_baseline.py --months 2   # quick iteration (sample 2 months for CV)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# Ensure src/ is on the path when running as a script
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import polars as pl

from pdc26 import SEED
from pdc26.cv import month_blocked_cv, persist_oof
from pdc26.data import load_ranking, load_training
from pdc26.features import (
    CAT_FEATURES,
    TARGET_CLIP_HI,
    TARGET_CLIP_LO,
    apply_categorical_vocab,
    build_features,
    fit_categorical_vocab,
    get_month_labels,
    persist_vocab,
)
from pdc26.train import get_default_params, save_model, score_ranking, train_full_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_baseline")

REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")


def compute_per_airport_rmse(
    oof_df: pd.DataFrame,
    airport_series: pd.Series,
    fold_month: str,
) -> pd.DataFrame:
    """Compute per-airport RMSE for a specific CV fold.

    Args:
        oof_df: Full OOF predictions DataFrame.
        airport_series: ADEP_mvt per row (aligned with training data index).
        fold_month: The fold to filter.

    Returns:
        DataFrame with columns: airport, n, rmse, mae, mean_actual.
    """
    fold = oof_df[oof_df["fold_month"] == fold_month].copy()
    # Merge airport info
    fold["ADEP_mvt"] = airport_series.loc[fold.index.intersection(airport_series.index)].values if len(fold) > 0 else None

    rows = []
    for apt in sorted(fold["ADEP_mvt"].dropna().unique()):
        subset = fold[fold["ADEP_mvt"] == apt]
        residuals = subset["prediction"] - subset["actual"]
        rows.append({
            "airport": apt,
            "n": len(subset),
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "mae": float(np.mean(np.abs(residuals))),
            "mean_actual": float(subset["actual"].mean()),
        })
    return pd.DataFrame(rows).sort_values("rmse", ascending=False)


def write_report(
    cv_result,
    per_airport_df: pd.DataFrame | None,
    worst_fold: str,
    elapsed_sec: float,
    n_train_rows: int,
    n_features: int,
    params: dict,
) -> str:
    """Write baseline_results.md report."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, "baseline_results.md")

    # ---- Season-matched (Jan+Jul) raw RMSE ----
    has_raw = cv_result.overall_rmse_raw > 0 and "rmse_raw" in cv_result.per_month.columns
    season_raw: float | None = None
    if has_raw:
        season_mask = cv_result.per_month["fold_month"].isin(["2025-01", "2025-07"])
        season_rows = cv_result.per_month[season_mask]
        if len(season_rows) > 0:
            # Compute from the OOF predictions for those months directly
            oof = cv_result.oof_predictions
            if oof is not None and "actual_raw" in oof.columns:
                jan_jul = oof[oof["fold_month"].isin(["2025-01", "2025-07"])]
                season_raw = float(np.sqrt(np.mean(
                    (jan_jul["prediction"] - jan_jul["actual_raw"]) ** 2
                )))

    lines = [
        "# Baseline Results — LightGBM Month-Blocked CV",
        "",
        f"Task: `04-baseline_pipeline` · Team: `thoughtful-glove`",
        "Status: **UPDATED (Task 2.2b — winsorize-errors target treatment + unclipped predictions)**",
        f"Generated: {time.strftime('%Y-%m-%d')}",
        f"Runtime: {elapsed_sec:.0f}s ({elapsed_sec / 60:.1f} min)",
        "",
        "---",
        "",
        "## Configuration",
        "",
        f"- **Model:** LightGBM regression",
        f"- **Features:** {n_features} (temporal + anchors + categoricals, no congestion)",
        f"- **Training rows:** {n_train_rows:,} DEP movements (Jan\u2013Dec 2025)",
        f"- **Target:** TAXITIME_SEC_mvt (winsorize-errors treatment: negatives < 1s → 1s, > 86400s → airport-specific P99; "
        f"**raw unclipped** used for the headline CV metric)",
        f"- **Categorical vocabulary:** persisted to `models/baseline_lgbm_vocab.json` (shared train/score mapping)",
        f"- **Seed:** {SEED}",
        f"- **Params:** lr={params['learning_rate']}, leaves={params['num_leaves']}, "
        f"subsample={params['subsample']}, colsample={params['colsample_bytree']}",
        f"- **Max rounds:** 500 (early stopping @ 50 rounds)",
        f"- **Prediction clip:** floor at 1s, no upper bound (relaxed from [0, 7200]s)",
        f"- **Task 2.2b finding:** E-winsorize-errors target treatment **344.94s** vs baseline 365.07s (−5.51%)",
        "",
        "---",
        "",
        "## Headline CV Metrics (selection signal)",
        "",
        f"| Metric | RMSE (s) | MAE (s) |",
        f"|--------|---------:|--------:|",
        f"| **Raw** unclipped target (competition metric) | **{cv_result.overall_rmse_raw:.2f}** | **{cv_result.overall_mae_raw:.2f}** |",
        f"| Clipped target [{TARGET_CLIP_LO}, {TARGET_CLIP_HI}s] (informational) | {cv_result.overall_rmse:.2f} | {cv_result.overall_mae:.2f} |",
        f"| Constant-baseline (raw) \u2014 predict overall training mean for every row | {cv_result.constant_baseline_rmse_raw:.2f} | \u2014 |",
        "",
        f"> **Raw-target RMSE is the headline model-selection metric.** The clipped metric is retained for continuity.\n> The constant baseline makes the raw RMSE interpretable: the model's RMSE of {cv_result.overall_rmse_raw:.2f}s "
        f"is {'{:.1f}%'.format((1 - cv_result.overall_rmse_raw / cv_result.constant_baseline_rmse_raw) * 100)} "
        f"vs the trivial predictor ({cv_result.constant_baseline_rmse_raw:.2f}s).",
        "",
        "---",
        "",
        "## CV Results \u2014 Per-Month",
        "",
    ]

    if has_raw:
        lines += [
            "| Fold Month | N Val | Best Iter | RMSE (clipped) | MAE (clipped) | RMSE (raw) | MAE (raw) |",
            "|------------|------:|----------:|----------------:|--------------:|-----------:|----------:|",
        ]
        for _, row in cv_result.per_month.iterrows():
            lines.append(
                f"| {row['fold_month']} | {int(row['n_val']):,} | {int(row['best_iteration'])} "
                f"| {row['rmse']:.2f} | {row['mae']:.2f} "
                f"| {row['rmse_raw']:.2f} | {row['mae_raw']:.2f} |"
            )
    else:
        lines += [
            "| Fold Month | N Val | Best Iter | RMSE (s) | MAE (s) |",
            "|------------|------:|----------:|---------:|--------:|",
        ]
        for _, row in cv_result.per_month.iterrows():
            lines.append(
                f"| {row['fold_month']} | {int(row['n_val']):,} | {int(row['best_iteration'])} "
                f"| {row['rmse']:.2f} | {row['mae']:.2f} |"
            )

    lines += [
        "",
        f"**Season-matched (Jan+Jul) raw RMSE: {season_raw:.2f}s** \u2014 the closest CV analog to the "
        f"leaderboard scoring set. Compare with the overall raw RMSE ({cv_result.overall_rmse_raw:.2f}s) "
        f"to see the season-mix effect (gap = {season_raw - cv_result.overall_rmse_raw:.2f}s).",
        "",
        "---",
        "",
        f"## Per-Airport RMSE \u2014 Worst Fold ({worst_fold})",
        "",
    ]

    if per_airport_df is not None and len(per_airport_df) > 0:
        lines += [
            f"Worst fold by RMSE: **{worst_fold}**",
            "",
            "| Airport | N | Mean Actual (s) | RMSE (s) | MAE (s) |",
            "|---------|--:|----------------:|---------:|--------:|",
        ]
        for _, row in per_airport_df.iterrows():
            lines.append(
                f"| {row['airport']} | {int(row['n']):,} | {row['mean_actual']:.1f} "
                f"| {row['rmse']:.2f} | {row['mae']:.2f} |"
            )
    else:
        lines.append("_Per-airport breakdown unavailable (insufficient data in fold)._")

    lines += [
        "",
        "---",
        "",
        "## Artifacts",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `models/baseline_lgbm.txt` | Trained LightGBM booster (full 2025 data) |",
        "| `models/baseline_lgbm_vocab.json` | Categorical vocabulary (shared train/score mapping) |",
        "| `data/processed/oof_predictions.parquet` | Out-of-fold predictions (per MVT_ID, incl. `actual_raw`) |",
        "| `data/processed/ranking_predictions.parquet` | Ranking DEP predictions from the full model |",
        "",
        "## Key Observations",
        "",
    ]

    # Auto-generated observations
    has_raw_col = "rmse_raw" in cv_result.per_month.columns
    if has_raw_col:
        best_fold_row = cv_result.per_month.loc[cv_result.per_month["rmse_raw"].idxmin()]
        worst_fold_row = cv_result.per_month.loc[cv_result.per_month["rmse_raw"].idxmax()]
        spread = worst_fold_row["rmse_raw"] - best_fold_row["rmse_raw"]
        metric_label = "raw"
    else:
        best_fold_row = cv_result.per_month.loc[cv_result.per_month["rmse"].idxmin()]
        worst_fold_row = cv_result.per_month.loc[cv_result.per_month["rmse"].idxmax()]
        spread = worst_fold_row["rmse"] - best_fold_row["rmse"]
        metric_label = "clipped"

    lines += [
        f"- Best fold ({metric_label}): {best_fold_row['fold_month']} (RMSE {best_fold_row['rmse_raw'] if has_raw_col else best_fold_row['rmse']:.2f}s)",
        f"- Worst fold ({metric_label}): {worst_fold_row['fold_month']} (RMSE {worst_fold_row['rmse_raw'] if has_raw_col else worst_fold_row['rmse']:.2f}s)",
        f"- Inter-fold ({metric_label}) RMSE spread: {spread:.2f}s \u2014 indicates {'high' if spread > 50 else 'moderate' if spread > 20 else 'low'} temporal sensitivity",
        f"- Near-oracle anchor RMSE floor: ~384s (MVT_TIME \u2212 AOBT_3 alone). "
        f"Overall raw RMSE of {cv_result.overall_rmse_raw:.2f}s "
        f"{'beats' if cv_result.overall_rmse_raw < 384 else 'is above'} this floor.",
        f"- Raw RMSE vs season-matched (Jan+Jul) {season_raw:.2f}s: "
        f"{'above' if cv_result.overall_rmse_raw < season_raw else 'below' if cv_result.overall_rmse_raw > season_raw else 'matches'} the season comparator "
        f"by {abs(season_raw - cv_result.overall_rmse_raw):.2f}s.",
    ]

    if per_airport_df is not None and len(per_airport_df) > 0:
        worst_apt = per_airport_df.iloc[0]
        best_apt = per_airport_df.iloc[-1]
        lines.append(
            f"- Worst airport in worst fold: {worst_apt['airport']} "
            f"(RMSE {worst_apt['rmse']:.2f}s) vs best: {best_apt['airport']} "
            f"(RMSE {best_apt['rmse']:.2f}s)"
        )

    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))

    log.info("Report written to %s", path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Run baseline LightGBM pipeline.")
    parser.add_argument(
        "--months", type=int, default=None,
        help="Limit CV to N months for quick iteration (default: all 12)."
    )
    args = parser.parse_args()

    t0 = time.time()
    log.info("=" * 60)
    log.info("PRC-OpenSky 2026 — Baseline Pipeline")
    log.info("=" * 60)

    # --- 1. Load data ---
    log.info("Step 1: Loading training data...")
    train_df = load_training()

    log.info("Step 1b: Loading ranking data...")
    rank_df = load_ranking()

    # Extract raw (unclipped) target BEFORE feature engineering clips it
    raw_target: pd.Series = train_df["TAXITIME_SEC_mvt"].to_pandas().astype("float64")
    raw_target.index = range(len(raw_target))
    log.info("Raw target: mean=%.1f  min=%.0f  max=%.0f  n_rows=%s",
             raw_target.mean(), raw_target.min(), raw_target.max(), f"{len(raw_target):,}")

    log.info("Step 1c: Applying winsorize-errors target treatment...")
    airports_series = train_df["ADEP_mvt"].to_pandas()
    airports_series.index = range(len(airports_series))
    y_train_winsor = raw_target.copy()
    # Fix negatives (< 1s) -> 1s
    y_train_winsor[y_train_winsor < 1] = 1
    # Fix extreme > 86400s (24h, clear data errors) -> cap at airport-specific P99
    extreme_mask = y_train_winsor > 86400
    if extreme_mask.any():
        for apt in airports_series[extreme_mask].unique():
            idx = extreme_mask & (airports_series.values == apt)
            upper = float(raw_target[airports_series.values == apt].quantile(0.99))
            y_train_winsor[idx] = upper
        log.info("  Winsorized %d extreme rows (> 86400s)", int(extreme_mask.sum()))
    n_neg = int((raw_target < 1).sum())
    log.info("  Fixed %d negative/sub-1s targets", n_neg)

    # --- 2. Build features with shared vocabulary ---
    log.info("Step 2: Fitting categorical vocabulary on training data...")
    vocab = fit_categorical_vocab(train_df)

    log.info("Step 2b: Building features for training and ranking...")
    X_train, y_train = build_features(train_df, vocab=vocab)
    X_rank, _ = build_features(rank_df, vocab=vocab)

    # Month labels and MVT IDs for CV
    months = get_month_labels(train_df)
    mvt_ids = train_df["MVT_ID_mvt"].to_pandas()
    airports = train_df["ADEP_mvt"].to_pandas()

    # Optionally limit months for quick iteration
    if args.months:
        unique_months = sorted(months.unique())
        keep_months = unique_months[:args.months]
        mask = months.isin(keep_months)
        X_train = X_train[mask]
        y_train = y_train[mask]
        months = months[mask]
        mvt_ids = mvt_ids[mask]
        airports = airports[mask]
        raw_target = raw_target[mask]
        log.info("Limited to %d months: %s", len(keep_months), keep_months)

    # Align indices
    X_train.index = range(len(X_train))
    y_train.index = range(len(y_train))
    months.index = range(len(months))
    mvt_ids.index = range(len(mvt_ids))
    airports.index = range(len(airports))
    raw_target.index = range(len(raw_target))

    # --- 3. Cross-validation ---
    log.info("Step 3: Running month-blocked CV...")
    params = get_default_params(seed=SEED)

    # Use winsorized target for training (E-winsorize-errors treatment)
    y_cv = y_train_winsor.copy()
    y_cv.index = range(len(y_cv))

    cv_result = month_blocked_cv(
        X=X_train,
        y=y_cv,
        months=months,
        mvt_ids=mvt_ids,
        cat_features=CAT_FEATURES,
        params=params,
        n_estimators=500,
        early_stopping_rounds=50,
        y_raw=raw_target,
    )

    # Persist OOF predictions
    persist_oof(cv_result.oof_predictions)

    # --- 4. Per-airport analysis on worst fold ---
    log.info("Step 4: Computing per-airport RMSE on worst fold...")
    worst_fold = cv_result.per_month.loc[cv_result.per_month["rmse"].idxmax(), "fold_month"]

    # Re-align airports with OOF predictions
    # OOF predictions have original indices from the training data
    # We need to map MVT_ID back to airport
    train_airport_map = dict(zip(mvt_ids.values, airports.values))
    cv_result.oof_predictions["ADEP_mvt"] = cv_result.oof_predictions["MVT_ID_mvt"].map(train_airport_map)

    per_airport_df = compute_per_airport_rmse(
        cv_result.oof_predictions,
        airports,
        worst_fold,
    )

    # --- 5. Train full model ---
    log.info("Step 5: Training full model on all data (winsorized target)...")
    # Full model uses same winsorized target as CV
    y_full = y_train_winsor.copy()
    y_full.index = range(len(y_full))
    full_model = train_full_model(
        X=X_train,
        y=y_full,
        cat_features=CAT_FEATURES,
        params=params,
        n_estimators=500,
    )
    model_path = save_model(full_model)

    # Persist categorical vocabulary alongside the booster
    vocab_path = model_path.replace(".txt", "_vocab.json")
    persist_vocab(vocab, vocab_path)

    # --- 5b. Score ranking with unclipped predictions ---
    log.info("Step 5b: Scoring with unclipped predictions...")

    # --- 6. Score ranking ---
    log.info("Step 6: Scoring ranking DEP rows...")
    rank_mvt_ids = rank_df["MVT_ID_mvt"].to_pandas()
    rank_mvt_ids.index = range(len(rank_mvt_ids))
    X_rank.index = range(len(X_rank))

    predictions = score_ranking(
        full_model, X_rank, rank_mvt_ids,
        clip_lo=1.0, clip_hi=None,  # floor at 1s, no upper clip
    )

    # Save ranking predictions
    pred_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", "ranking_predictions.parquet"
    )
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    predictions.to_parquet(pred_path, index=False)
    log.info("Ranking predictions saved to %s", pred_path)

    # --- 7. Write report ---
    log.info("Step 7: Writing results report...")
    elapsed = time.time() - t0
    write_report(
        cv_result=cv_result,
        per_airport_df=per_airport_df,
        worst_fold=worst_fold,
        elapsed_sec=elapsed,
        n_train_rows=len(X_train),
        n_features=X_train.shape[1],
        params=params,
    )

    # --- Summary ---
    log.info("=" * 60)
    log.info("PIPELINE COMPLETE in %.0fs (%.1f min)", elapsed, elapsed / 60)
    log.info("  Target treatment: winsorize-errors (negatives -> 1s, >86400s -> airport P99)")
    log.info("  Prediction clip:  floor at 1s, no upper bound")
    log.info("  Overall CV RMSE: %.2fs (clipped) | %.2fs (raw)", cv_result.overall_rmse, cv_result.overall_rmse_raw)
    log.info("  Constant baseline (raw): %.2fs", cv_result.constant_baseline_rmse_raw)
    log.info("  Model: models/baseline_lgbm.txt")
    log.info("  Vocab: models/baseline_lgbm_vocab.json")
    log.info("  OOF:   data/processed/oof_predictions.parquet")
    log.info("  Report: src/reports/baseline_results.md")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
