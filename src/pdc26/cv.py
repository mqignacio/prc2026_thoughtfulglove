"""Month-blocked temporal cross-validation harness.

Validation design (locked decision from task spec):
  Train on all months except one, validate on the held-out month.
  Rotate over 12 months of 2025. Report per-month and aggregate RMSE.
  This is the PRIMARY model-selection signal.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")


@dataclass
class CVResult:
    """Container for cross-validation results."""

    per_month: pd.DataFrame = field(default_factory=pd.DataFrame)
    """Table with columns: fold_month, n_val, rmse, mae, rmse_raw, mae_raw."""

    overall_rmse: float = 0.0
    """RMSE across all OOF predictions, clipped target."""

    overall_mae: float = 0.0
    """MAE across all OOF predictions, clipped target."""

    overall_rmse_raw: float = 0.0
    """RMSE across all OOF predictions, raw (unclipped) target."""

    overall_mae_raw: float = 0.0
    """MAE across all OOF predictions, raw (unclipped) target."""

    constant_baseline_rmse_raw: float = 0.0
    """RMSE of a trivial constant (mean) predictor on the raw target."""

    oof_predictions: pd.DataFrame | None = None
    """DataFrame with columns: MVT_ID_mvt, fold_month, prediction, actual, actual_raw."""


def month_blocked_cv(
    X: pd.DataFrame,
    y: pd.Series,
    months: pd.Series,
    mvt_ids: pd.Series,
    cat_features: list[str],
    params: dict,
    n_estimators: int = 500,
    early_stopping_rounds: int = 50,
    y_raw: pd.Series | None = None,
) -> CVResult:
    """Run leave-one-month-out temporal cross-validation.

    Args:
        X: Feature matrix (pandas DataFrame).
        y: Target series (clipped TAXITIME_SEC_mvt).
        months: Month label per row (e.g. "2025-01").
        mvt_ids: MVT_ID_mvt per row (for OOF persistence).
        cat_features: Column names to treat as native LightGBM categoricals.
        params: LightGBM training parameters (excluding n_estimators/seed).
        n_estimators: Max boosting rounds.
        early_stopping_rounds: Stop if no improvement for N rounds on val fold.
        y_raw: Optional unclipped TAXITIME_SEC_mvt. When provided, RMSE/MAE
            are also computed against the raw target (the competition metric).

    Returns:
        CVResult with per-month table, overall metrics, and OOF predictions.
    """
    unique_months = sorted(months.unique())
    log.info("Running %d-fold month-blocked CV on %s rows", len(unique_months), f"{len(X):,}")

    per_month_rows: list[dict] = []
    oof_preds: list[pd.DataFrame] = []

    for fold_month in unique_months:
        val_mask = months == fold_month
        train_mask = ~val_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        log.info("  Fold %s: train=%s  val=%s",
                 fold_month, f"{len(X_train):,}", f"{len(X_val):,}")

        # Build DataMatrices with native categorical support
        dtrain = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=cat_features,
            free_raw_data=False,
        )
        dval = lgb.Dataset(
            X_val,
            label=y_val,
            categorical_feature=cat_features,
            reference=dtrain,
            free_raw_data=False,
        )

        callbacks = [
            lgb.log_evaluation(period=0),  # suppress per-round output
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        ]

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            valid_sets=[dval],
            callbacks=callbacks,
        )

        # Predict on validation fold
        preds = model.predict(X_val, num_iteration=model.best_iteration)
        residuals = preds - y_val.values
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        mae = float(np.mean(np.abs(residuals)))

        # ---- Raw-target metrics (competition metric) ----
        rmse_raw: float | None = None
        mae_raw: float | None = None
        if y_raw is not None:
            y_val_raw = y_raw[val_mask].values
            residuals_raw = preds - y_val_raw
            rmse_raw = float(np.sqrt(np.mean(residuals_raw ** 2)))
            mae_raw = float(np.mean(np.abs(residuals_raw)))

        row: dict = {
            "fold_month": fold_month,
            "n_val": int(val_mask.sum()),
            "best_iteration": model.best_iteration,
            "rmse": rmse,
            "mae": mae,
        }
        if rmse_raw is not None:
            row["rmse_raw"] = rmse_raw
            row["mae_raw"] = mae_raw
        per_month_rows.append(row)

        # Collect OOF predictions
        oof_data = {
            "MVT_ID_mvt": mvt_ids[val_mask].values,
            "fold_month": fold_month,
            "prediction": preds.astype("float64"),
            "actual": y_val.values.astype("float64"),
        }
        if y_raw is not None:
            oof_data["actual_raw"] = y_raw[val_mask].values.astype("float64")
        oof_fold = pd.DataFrame(oof_data)
        oof_preds.append(oof_fold)

        log_msg = f"  Fold {fold_month}: RMSE={rmse:.2f}  MAE={mae:.2f}  best_iter={model.best_iteration}"
        if rmse_raw is not None:
            log_msg += f"  raw_RMSE={rmse_raw:.2f}"
        log.info(log_msg)

    # Aggregate
    per_month_df = pd.DataFrame(per_month_rows)
    all_oof = pd.concat(oof_preds, ignore_index=True)

    overall_rmse = float(np.sqrt(np.mean((all_oof["prediction"] - all_oof["actual"]) ** 2)))
    overall_mae = float(np.mean(np.abs(all_oof["prediction"] - all_oof["actual"])))

    overall_rmse_raw = 0.0
    overall_mae_raw = 0.0
    constant_baseline_rmse_raw = 0.0
    if y_raw is not None and "actual_raw" in all_oof.columns:
        overall_rmse_raw = float(np.sqrt(np.mean(
            (all_oof["prediction"] - all_oof["actual_raw"]) ** 2
        )))
        overall_mae_raw = float(np.mean(np.abs(
            all_oof["prediction"] - all_oof["actual_raw"]
        )))
        # Constant baseline: predict the training-set mean for every row
        train_mean = y_raw[train_mask].mean() if len(y_raw) > 0 else y_raw.mean()
        # Use a split-independent constant: mean of all y_raw
        train_constant = y_raw.mean()
        constant_baseline_rmse_raw = float(np.sqrt(np.mean(
            (train_constant - all_oof["actual_raw"]) ** 2
        )))

    log.info("CV complete: RMSE=%.2f (MAE=%.2f) | raw RMSE=%.2f (raw MAE=%.2f) | const baseline raw=%.2f",
             overall_rmse, overall_mae, overall_rmse_raw, overall_mae_raw, constant_baseline_rmse_raw)

    return CVResult(
        per_month=per_month_df,
        overall_rmse=overall_rmse,
        overall_mae=overall_mae,
        overall_rmse_raw=overall_rmse_raw,
        overall_mae_raw=overall_mae_raw,
        constant_baseline_rmse_raw=constant_baseline_rmse_raw,
        oof_predictions=all_oof,
    )


def persist_oof(oof_df: pd.DataFrame) -> str:
    """Save out-of-fold predictions to parquet for downstream tasks.

    Args:
        oof_df: DataFrame with MVT_ID_mvt, fold_month, prediction, actual.

    Returns:
        Path to the written parquet file.
    """
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    path = os.path.join(DATA_PROCESSED, "oof_predictions.parquet")
    oof_df.to_parquet(path, index=False)
    log.info("Persisted OOF predictions (%s rows) to %s", f"{len(oof_df):,}", path)
    return path
