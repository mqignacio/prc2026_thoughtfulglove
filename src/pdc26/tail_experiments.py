"""Experiment runner for tail/outlier target treatments.

Each treatment transforms the raw TAXITIME_SEC_mvt differently during training,
and the CV harness evaluates against the raw target (competition metric).
Results are logged to a structured dict for report generation.

Treatments compared:
  A — Current baseline: clip(y_raw, 1, 14400)
  B — Raw/unclipped: only clip negatives to 0
  C — Per-airport P99.5 clip
  D — Global P99.95 clip
  E — Winsorise only clear data errors (negatives -> 1, >86400 -> P99 airport)
  F — log1p-transform (train on log1p, predict with expm1)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pdc26 import SEED
from pdc26.cv import CVResult, month_blocked_cv
from pdc26.data import load_training
from pdc26.features import (
    CAT_FEATURES,
    FEATURE_COLUMNS,
    TARGET_CLIP_HI,
    TARGET_CLIP_LO,
    build_features,
    fit_categorical_vocab,
    get_month_labels,
)
from pdc26.train import get_default_params

log = logging.getLogger(__name__)

# ── Treatment configuration ──────────────────────────────────

TreatmentConfig = dict[str, object]


def _config_hash(config: dict) -> str:
    """Deterministic short hash for a config dict."""
    raw = json.dumps(config, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


# ── Target transformation functions ──────────────────────────

def treatment_baseline(raw_target: pd.Series, **_) -> pd.Series:
    """Current baseline: clip to [1, 14400]."""
    return raw_target.clip(TARGET_CLIP_LO, TARGET_CLIP_HI)


def treatment_raw(raw_target: pd.Series, **_) -> pd.Series:
    """No upper clipping — let model see full distribution. Negatives floor at 0."""
    return raw_target.clip(0, None)


def treatment_per_airport_p995(raw_target: pd.Series,
                                airports: pd.Series | None = None, **_) -> pd.Series:
    """Per-airport 99.5th percentile clip, lower bound at 1s."""
    if airports is None:
        raise ValueError("airports required for per-airport clip")
    y = raw_target.copy()
    for apt in airports.unique():
        mask = airports.values == apt
        upper = float(raw_target[mask].quantile(0.995))
        y[mask] = y[mask].clip(upper=upper)
    return y.clip(TARGET_CLIP_LO, None)


def treatment_global_p9995(raw_target: pd.Series, **_) -> pd.Series:
    """Global 99.95th percentile clip."""
    upper = float(raw_target.quantile(0.9995))
    return raw_target.clip(TARGET_CLIP_LO, upper)


def treatment_winsorize_errors(raw_target: pd.Series,
                                airports: pd.Series | None = None, **_) -> pd.Series:
    """Winsorise only clear data errors:
    - Negatives (< 1s) -> 1
    - > 86400s (24h, clear error) -> cap at airport-specific P99
    """
    y = raw_target.copy()
    # Fix negatives
    y[y < TARGET_CLIP_LO] = TARGET_CLIP_LO
    # Fix extreme > 24h
    extreme_mask = y > 86400
    if extreme_mask.any() and airports is not None:
        for apt in airports[extreme_mask].unique():
            idx = extreme_mask & (airports.values == apt)
            upper = float(raw_target[airports.values == apt].quantile(0.99))
            y[idx] = upper
    elif extreme_mask.any():
        # Fallback: global P99
        y[extreme_mask] = float(raw_target.quantile(0.99))
    return y


def treatment_log1p(raw_target: pd.Series, **_) -> pd.Series:
    """log1p-transform: train on log(1+y), predictions inverted via expm1."""
    return np.log1p(raw_target.clip(0, None))


# ── Post-processing functions ───────────────────────────────

def invert_log1p(preds: np.ndarray) -> np.ndarray:
    """Invert log1p predictions back to raw scale."""
    return np.expm1(preds)


def remove_prediction_clip(preds: np.ndarray) -> np.ndarray:
    """Remove the upper clip — only floor at 1s."""
    return np.maximum(preds, 1.0)


def apply_prediction_clip(preds: np.ndarray, lo: float = 1.0,
                           hi: float = 7200.0) -> np.ndarray:
    """Clip predictions to [lo, hi]."""
    return np.clip(preds, lo, hi)


# ── Available treatments registry ────────────────────────────

TREATMENTS: dict[str, tuple] = {
    "A-baseline": (treatment_baseline, {}, None),
    "B-raw": (treatment_raw, {}, None),
    "C-per-airport-P99.5": (treatment_per_airport_p995, {}, None),
    "D-global-P99.95": (treatment_global_p9995, {}, None),
    "E-winsorize-errors": (treatment_winsorize_errors, {}, None),
    "F-log1p": (treatment_log1p, {}, invert_log1p),
}

# Post-processing variants to test on the winning treatment
POST_PROCESSING: dict[str, tuple] = {
    "no-clip": (remove_prediction_clip, None),
    "relaxed-clip-14400": (lambda p: apply_prediction_clip(p, 1.0, 14400.0), None),
    "tight-clip-7200": (lambda p: apply_prediction_clip(p, 1.0, 7200.0), None),
}


# ── Experiment runner ────────────────────────────────────────

@dataclass
class ExperimentResult:
    """Result of one tail-treatment experiment."""
    label: str
    config_hash: str
    overall_rmse_raw: float
    overall_mae_raw: float
    constant_baseline_rmse_raw: float
    per_month: pd.DataFrame
    cv_result: CVResult
    elapsed_sec: float
    post_process_label: str | None = None


def run_experiment(
    *,
    X_train: pd.DataFrame,
    raw_target: pd.Series,
    months: pd.Series,
    mvt_ids: pd.Series,
    airports: pd.Series,
    cat_features: list[str],
    params: dict,
    treatment_name: str,
    treatment_fn,
    treatment_kwargs: dict,
    invert_pred_fn=None,
    n_estimators: int = 500,
    early_stopping_rounds: int = 50,
) -> ExperimentResult:
    """Run a single treatment experiment and return results.

    Args:
        treatment_fn: Function (raw_target, **kwargs) -> y_train.
        treatment_kwargs: Extra kwargs for the treatment function.
        invert_pred_fn: Optional function to invert predictions (e.g. expm1 for log1p).
    """
    t0 = time.time()

    # Build training target from the raw target
    kw = dict(airports=airports)
    kw.update(treatment_kwargs)
    y_train = treatment_fn(raw_target, **kw)

    config = {
        "treatment": treatment_name,
        "n_features": X_train.shape[1],
        "n_estimators": n_estimators,
        "params": {k: v for k, v in params.items() if k != "seed"},
    }
    c_hash = _config_hash(config)

    # Run CV
    cv_result = month_blocked_cv(
        X=X_train,
        y=y_train,
        months=months,
        mvt_ids=mvt_ids,
        cat_features=cat_features,
        params=params,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        y_raw=raw_target,
    )

    # Apply inverse transform if needed (e.g. log1p -> expm1)
    if invert_pred_fn is not None:
        oof = cv_result.oof_predictions
        oof["prediction"] = invert_pred_fn(oof["prediction"].values)

        # Recompute overall raw RMSE/MAE
        residuals_raw = oof["prediction"] - oof["actual_raw"]
        cv_result.overall_rmse_raw = float(np.sqrt(np.mean(residuals_raw ** 2)))
        cv_result.overall_mae_raw = float(np.mean(np.abs(residuals_raw)))

        # Recompute per-month raw metrics
        for month in cv_result.per_month["fold_month"]:
            mask = oof["fold_month"] == month
            r = oof[mask]
            if len(r) > 0:
                res_month = r["prediction"] - r["actual_raw"]
                cv_result.per_month.loc[
                    cv_result.per_month["fold_month"] == month, "rmse_raw"
                ] = float(np.sqrt(np.mean(res_month ** 2)))
                cv_result.per_month.loc[
                    cv_result.per_month["fold_month"] == month, "mae_raw"
                ] = float(np.mean(np.abs(res_month)))

    elapsed = time.time() - t0
    log.info(
        "Treatment %s: raw RMSE=%.2f (MAE=%.2f) | const baseline=%.2f | %.1fs",
        treatment_name,
        cv_result.overall_rmse_raw,
        cv_result.overall_mae_raw,
        cv_result.constant_baseline_rmse_raw,
        elapsed,
    )

    return ExperimentResult(
        label=treatment_name,
        config_hash=c_hash,
        overall_rmse_raw=cv_result.overall_rmse_raw,
        overall_mae_raw=cv_result.overall_mae_raw,
        constant_baseline_rmse_raw=cv_result.constant_baseline_rmse_raw,
        per_month=cv_result.per_month.copy(),
        cv_result=cv_result,
        elapsed_sec=elapsed,
    )


def run_post_processing_experiment(
    cv_result: CVResult,
    raw_target: pd.Series,
    label: str,
    post_process_fn,
    invert_pred_fn=None,
) -> ExperimentResult:
    """Evaluate a prediction post-processing function on existing OOF predictions.

    Args:
        cv_result: CVResult from the base treatment experiment.
        post_process_fn: Function (predictions) -> transformed predictions.
    """
    t0 = time.time()

    oof = cv_result.oof_predictions.copy()
    preds = oof["prediction"].values

    # Apply inverse if needed (log1p case), then post-process
    if invert_pred_fn is not None:
        preds = invert_pred_fn(preds)
    preds = post_process_fn(preds)

    # Evaluate
    residuals_raw = preds - oof["actual_raw"].values
    overall_rmse_raw = float(np.sqrt(np.mean(residuals_raw ** 2)))
    overall_mae_raw = float(np.mean(np.abs(residuals_raw)))

    elapsed = time.time() - t0
    log.info(
        "Post-process %s on %s: raw RMSE=%.2f (MAE=%.2f) | %.1fs",
        label, cv_result.label if hasattr(cv_result, 'label') else '?',
        overall_rmse_raw, overall_mae_raw, elapsed,
    )

    return ExperimentResult(
        label=f"{label}",
        config_hash="postproc",
        overall_rmse_raw=overall_rmse_raw,
        overall_mae_raw=overall_mae_raw,
        constant_baseline_rmse_raw=cv_result.constant_baseline_rmse_raw,
        per_month=None,
        cv_result=cv_result,
        elapsed_sec=elapsed,
        post_process_label=label,
    )


# ── Main experiment harness ─────────────────────────────────

def run_all_experiments(
    filter_months: list[str] | None = None,
    treatments: list[str] | None = None,
) -> dict[str, ExperimentResult]:
    """Run all tail treatments and return results dict.

    Args:
        filter_months: Optional list of months to limit (for quick iteration).
        treatments: Optional list of treatment names to run (default: all).

    Returns:
        Dict of treatment name -> ExperimentResult.
    """
    log.info("=" * 60)
    log.info("Tail / Outlier Treatment Experiments")
    log.info("=" * 60)

    # ── 1. Load data ──
    log.info("Loading training data...")
    train_df = load_training(filter_months=filter_months)

    raw_target = train_df["TAXITIME_SEC_mvt"].to_pandas().astype("float64")
    raw_target.index = range(len(raw_target))
    log.info("Raw target: mean=%.1f  min=%.0f  max=%.0f  n_rows=%s",
             raw_target.mean(), raw_target.min(), raw_target.max(), f"{len(raw_target):,}")

    # ── 2. Build features (shared across all treatments) ──
    log.info("Fitting categorical vocabulary...")
    vocab = fit_categorical_vocab(train_df)

    log.info("Building features...")
    X_train, _ = build_features(train_df, vocab=vocab)
    X_train.index = range(len(X_train))

    months = get_month_labels(train_df)
    mvt_ids = train_df["MVT_ID_mvt"].to_pandas()
    airports = train_df["ADEP_mvt"].to_pandas()

    # Align indices
    for s in [months, mvt_ids, airports, raw_target]:
        s.index = range(len(s))

    # If filter_months, subset
    if filter_months:
        unique_months = sorted(months.unique())
        keep = [m for m in unique_months if m[:7] in filter_months or
                any(f in m for f in filter_months)]
        if keep:
            mask = months.isin(keep)
            X_train = X_train[mask]
            months = months[mask]
            mvt_ids = mvt_ids[mask]
            airports = airports[mask]
            raw_target = raw_target[mask]

    log.info("Data ready: %s rows × %d features", f"{len(X_train):,}", X_train.shape[1])

    # ── 3. Run treatments ──
    params = get_default_params(seed=SEED)

    selected_treatments = treatments or list(TREATMENTS.keys())
    results: dict[str, ExperimentResult] = {}

    for name in selected_treatments:
        if name not in TREATMENTS:
            log.warning("Unknown treatment: %s, skipping", name)
            continue
        fn, kwargs, invert_fn = TREATMENTS[name]
        log.info("\n--- Treatment: %s ---", name)
        result = run_experiment(
            X_train=X_train,
            raw_target=raw_target,
            months=months,
            mvt_ids=mvt_ids,
            airports=airports,
            cat_features=CAT_FEATURES,
            params=params,
            treatment_name=name,
            treatment_fn=fn,
            treatment_kwargs=kwargs,
            invert_pred_fn=invert_fn,
        )
        results[name] = result

    return results


def format_results_table(results: dict[str, ExperimentResult]) -> str:
    """Format treatment results as a markdown table."""
    lines = [
        "| Treatment | Raw RMSE (s) | Raw MAE (s) | Const Baseline (s) | vs Baseline (s) | vs Baseline (%) | Time (s) |",
        "|-----------|-------------:|------------:|-------------------:|----------------:|----------------:|---------:|",
    ]

    baseline_rmse = results.get("A-baseline", {}).overall_rmse_raw
    if isinstance(baseline_rmse, ExperimentResult):
        baseline_rmse = baseline_rmse.overall_rmse_raw

    for name in sorted(results.keys()):
        r = results[name]
        delta = r.overall_rmse_raw - baseline_rmse
        pct = (delta / baseline_rmse) * 100
        lines.append(
            f"| {name} | {r.overall_rmse_raw:.2f} | {r.overall_mae_raw:.2f} "
            f"| {r.constant_baseline_rmse_raw:.2f} "
            f"| {delta:+.2f} | {pct:+.2f}% | {r.elapsed_sec:.1f} |"
        )

    return "\n".join(lines)


# ── CLI entry point ─────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    import pandas as pd  # noqa: F811

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    import argparse
    parser = argparse.ArgumentParser(description="Run tail treatment experiments")
    parser.add_argument("--months", type=int, default=None,
                        help="Limit to N months for quick iteration")
    parser.add_argument("--treatment", type=str, default=None,
                        help="Single treatment to run (default: all)")
    args = parser.parse_args()

    filter_m = None
    if args.months:
        # Load all months, take first N
        train_df = load_training()
        unique_months = sorted(
            pd.to_datetime(train_df["MVT_TIME_UTC_mvt"].to_pandas(), utc=True)
            .dt.strftime("%Y-%m").unique()
        )
        filter_m = unique_months[:args.months]
        log.info("Quick test: limiting to first %d months: %s", args.months, filter_m)

    treatments = [args.treatment] if args.treatment else None

    results = run_all_experiments(
        filter_months=filter_m,
        treatments=treatments,
    )

    print("\n\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(format_results_table(results))