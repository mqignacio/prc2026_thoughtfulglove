"""Model comparison and hyperparameter tuning utilities (Task 3.3).

This module provides two reusable pieces on top of the existing pipeline:

1. `build_d4_features()` — assembles the full D4 feature matrix
   (baseline + METAR + congestion + domain), the current best arm from Task 09
   (raw RMSE 338.43s). It is extracted from `run_domain_ablation.py` so that
   every model family is compared on identical features.

2. `month_blocked_cv_model()` — a model-family-agnostic leave-one-month-out CV
   driver that mirrors `cv.month_blocked_cv` but supports LightGBM / XGBoost /
   CatBoost, returning identical metrics (per-month + overall raw RMSE/MAE +
   OOF predictions).

Target treatment is FIXED to E-winsorize-errors (settled in Task 05b) and is
NOT re-derived here.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")


# ---------------------------------------------------------------------------
# Target treatment (E-winsorize-errors, locked from Task 05b)
# ---------------------------------------------------------------------------

def winsorize_errors(raw_target: pd.Series, airports: pd.Series) -> pd.Series:
    """Apply the E-winsorize-errors target treatment (Task 05b winner).

    - Negatives (< 1s) -> 1
    - > 86400s (24h, clear data error) -> cap at airport-specific P99

    Args:
        raw_target: Unclipped TAXITIME_SEC_mvt per row.
        airports: ADEP_mvt per row (same index).

    Returns:
        Winsorized training target.
    """
    y = raw_target.copy()
    y[y < 1] = 1
    extreme_mask = y > 86400
    if extreme_mask.any():
        for apt in airports[extreme_mask].unique():
            idx = extreme_mask & (airports.values == apt)
            upper = float(raw_target[airports.values == apt].quantile(0.99))
            y[idx] = upper
    return y


# ---------------------------------------------------------------------------
# D4 feature assembly
# ---------------------------------------------------------------------------

def load_metar() -> pd.DataFrame:
    """Load pre-processed METAR feature table."""
    from pdc26.features import EXTERNAL_METAR_COLUMNS  # noqa: F401
    metar_path = os.path.join(DATA_PROCESSED, "metar_features.parquet")
    return pd.read_parquet(metar_path)


def fit_domain_cat_vocab(pdf: pd.DataFrame) -> dict:
    """Fit stable code vocabulary for domain categorical columns."""
    from pdc26.features import _extract_stand_zone

    stand = pdf["STAND_mvt"].astype("string").fillna("__NA__")
    runway = pdf["RUNWAY_mvt"].astype("string").fillna("__NA__")
    stand_zone = stand.map(_extract_stand_zone)
    interaction = stand_zone + "_" + runway
    market = pdf["MARKET_SEGMENT_flt"].astype("string").fillna("__NA__")
    flight_type = pdf["FLIGHT_TYPE_flt"].astype("string").fillna("__NA__")

    return {
        "stand_zone": sorted(stand_zone.unique().tolist()),
        "stand_runway_interaction": sorted(interaction.unique().tolist()),
        "market_segment_code": sorted(market.unique().tolist()),
        "flight_type_code": sorted(flight_type.unique().tolist()),
    }


def fit_d4_vocabs(dep_df) -> tuple[dict, dict]:
    """Fit the base categorical vocab and the domain categorical vocab."""
    from pdc26.features import fit_categorical_vocab

    vocab = fit_categorical_vocab(dep_df)
    pdf_full = dep_df.to_pandas()
    domain_vocab = fit_domain_cat_vocab(pdf_full)
    return vocab, domain_vocab


def build_d4_features(
    dep_df,
    movements_pdf: pd.DataFrame,
    vocab: dict,
    domain_vocab: dict,
    metar_df: pd.DataFrame,
    coords: dict,
    operator_enc: pd.Series | None = None,
) -> pd.DataFrame:
    """Build the full D4 feature matrix: baseline + METAR + congestion + domain.

    Operator target encoding is computed within-fold by the caller; pass a
    precomputed `operator_enc` aligned to dep_df rows when `flight_chars` is on.

    Args:
        dep_df: Polars DEP DataFrame (from load_training).
        movements_pdf: pandas DataFrame of ALL movements (DEP+ARR).
        vocab: base categorical vocab from fit_categorical_vocab.
        domain_vocab: domain categorical vocab from fit_domain_cat_vocab.
        metar_df: METAR feature table.
        coords: load_coords() dict.
        operator_enc: precomputed within-fold operator target encoding (optional).

    Returns:
        Feature matrix X (pandas DataFrame) with the full D4 column set.
    """
    from pdc26.features import (
        CAT_FEATURES,
        CONGESTION_ALL_COLUMNS,
        DOMAIN_ALL_COLUMNS,
        build_congestion_features,
        build_domain_features,
        build_features,
        join_external_features,
    )

    X, _ = build_features(dep_df, vocab=vocab)

    pdf = dep_df.to_pandas()

    # METAR
    join_external_features(X, pdf, metar_df=metar_df)

    # Congestion
    cong_X = build_congestion_features(pdf, movements_pdf)
    for col in CONGESTION_ALL_COLUMNS:
        if col in cong_X.columns:
            X[col] = cong_X[col].values

    # Domain (all families)
    domain_X = build_domain_features(
        pdf,
        groups=["local_time", "geometry", "flight_chars"],
        coords=coords,
        cat_vocab=domain_vocab,
        operator_enc=operator_enc,
    )
    for col in DOMAIN_ALL_COLUMNS:
        if col in domain_X.columns:
            X[col] = domain_X[col].values
        else:
            X[col] = np.nan

    X.index = range(len(X))
    return X


def encode_operator_within_fold(
    operator_series: pd.Series,
    fold_masks: dict,
    y_winsor: pd.Series,
) -> pd.Series:
    """Compute operator target encoding within each CV fold's training split.

    Fit on train rows, map to validation rows; leakage asserted per fold.
    """
    from pdc26.features import (
        apply_operator_target_encoding,
        assert_no_target_encoding_leakage,
    )

    n = len(operator_series)
    encoded = np.full(n, np.nan, dtype=np.float64)
    ops = operator_series.astype("string").fillna("__NA__")

    for fold_month, val_mask in fold_masks.items():
        train_mask = ~val_mask
        train_ops = ops[train_mask].reset_index(drop=True)
        train_y = y_winsor[train_mask].reset_index(drop=True)
        val_ops = ops[val_mask].reset_index(drop=True)

        result = assert_no_target_encoding_leakage(train_ops, val_ops, train_y)
        encoding = result["encoding"]
        val_encoded = apply_operator_target_encoding(val_ops, encoding)
        encoded[val_mask] = val_encoded.to_numpy()

    return pd.Series(encoded, index=range(n))


# ---------------------------------------------------------------------------
# Model-family-agnostic CV
# ---------------------------------------------------------------------------

@dataclass
class TuningCVResult:
    """Container for a family-agnostic cross-validation result."""
    per_month: pd.DataFrame = field(default_factory=pd.DataFrame)
    overall_rmse_raw: float = 0.0
    overall_mae_raw: float = 0.0
    oof_predictions: pd.DataFrame | None = None


def _train_predict_family(
    family: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_features: list[str],
    params: dict,
    n_estimators: int,
    early_stopping_rounds: int,
):
    """Train a model of `family` and return (validation predictions, best_iteration).

    best_iteration is the early-stopped round count for the fold, used by the
    blending stage to choose full-retrain round counts.
    """
    if family == "lgbm":
        import lightgbm as lgb

        params = {k: v for k, v in params.items() if v is not None}
        dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features,
                             free_raw_data=False)
        dval = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_features,
                           reference=dtrain, free_raw_data=False)
        callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
        model = lgb.train(
            params, dtrain, num_boost_round=n_estimators,
            valid_sets=[dval], callbacks=callbacks,
        )
        return model.predict(X_val, num_iteration=model.best_iteration), int(model.best_iteration)

    elif family == "xgb":
        import xgboost as xgb

        p = dict(params)
        p.pop("metric", None)
        p.pop("objective", None)
        p.pop("force_row_wise", None)
        p.pop("num_threads", None)
        p.pop("nthread", None)
        p["verbosity"] = 0
        # XGBoost needs numeric matrices; categoricals already integer-coded.
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        model = xgb.train(
            p, dtrain, num_boost_round=n_estimators,
            evals=[(dval, "val")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        return model.predict(dval, iteration_range=(0, model.best_iteration + 1)), int(model.best_iteration)

    elif family == "cat":
        from catboost import CatBoostRegressor, Pool

        p = dict(params)
        # Extract catboost-specific params if present; otherwise defaults.
        ctb_params = {}
        for key in ("learning_rate", "depth", "l2_leaf_reg", "iterations",
                    "random_seed", "loss_function", "eval_metric", "verbose"):
            if key in p and p[key] is not None:
                ctb_params[key] = p[key]
        ctb_params.setdefault("iterations", n_estimators)
        ctb_params.setdefault("random_seed", params.get("seed", 42))
        ctb_params.setdefault("loss_function", "RMSE")
        ctb_params.setdefault("verbose", False)

        # Treat categorical codes as numeric: the base categoricals include
        # very high-cardinality columns (STAND 1900, operator 677, stand-runway
        # interaction 1577 levels) which make CatBoost native ordered target
        # encoding impractically slow. Integer codes are already a valid numeric
        # representation (same as XGBoost path, keeping the comparison fair).
        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_val, y_val)
        model = CatBoostRegressor(**ctb_params)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=early_stopping_rounds,
                  use_best_model=True)
        return model.predict(val_pool), int(model.get_best_iteration())

    else:
        raise ValueError(f"Unknown family: {family}")


def family_default_params(family: str, seed: int = 42) -> dict:
    """Return family-equivalent 'default' parameters.

    All families use the same lr / tree-count / subsampling spirit as the
    LightGBM baseline so the comparison is fair rather than tuned.
    """
    if family == "lgbm":
        from pdc26.train import get_default_params
        return get_default_params(seed=seed)
    if family == "xgb":
        return {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "seed": seed,
            "nthread": 0,
            "verbosity": 0,
        }
    if family == "cat":
        return {
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_seed": seed,
            "verbose": False,
            "iterations": 300,
        }
    raise ValueError(f"Unknown family: {family}")


def month_blocked_cv_model(
    X: pd.DataFrame,
    y: pd.Series,
    months: pd.Series,
    mvt_ids: pd.Series,
    cat_features: list[str],
    family: str,
    params: dict,
    n_estimators: int = 500,
    early_stopping_rounds: int = 50,
    y_raw: pd.Series | None = None,
) -> TuningCVResult:
    """Leave-one-month-out CV across model families.

    Mirrors `cv.month_blocked_cv` but dispatches to LightGBM / XGBoost / CatBoost.

    Args:
        X: Feature matrix.
        y: Training target (winsorized).
        months: Month label per row.
        mvt_ids: MVT_ID_mvt per row.
        cat_features: Native categorical columns (LightGBM/CatBoost only).
        family: "lgbm", "xgb", or "cat".
        params: Family-specific parameter dict.
        n_estimators: Max boosting rounds.
        early_stopping_rounds: Early stopping patience.
        y_raw: Unclipped target for the raw (competition) metric.

    Returns:
        TuningCVResult.
    """
    unique_months = sorted(months.unique())
    log.info("Family=%s: %d-fold month-blocked CV on %s rows",
             family, len(unique_months), f"{len(X):,}")

    per_month_rows: list[dict] = []
    oof_preds: list[pd.DataFrame] = []

    for fold_month in unique_months:
        val_mask = months == fold_month
        train_mask = ~val_mask

        X_train, y_train = X[train_mask], y[train_mask].values
        X_val = X[val_mask]
        y_val = y[val_mask].values

        preds, best_iter = _train_predict_family(
            family, X_train, y_train, X_val, y_val, cat_features, params,
            n_estimators, early_stopping_rounds,
        )

        row = {"fold_month": fold_month, "n_val": int(val_mask.sum()),
               "best_iteration": best_iter}
        if y_raw is not None:
            y_val_raw = y_raw[val_mask].values
            residuals_raw = preds - y_val_raw
            row["rmse_raw"] = float(np.sqrt(np.mean(residuals_raw ** 2)))
            row["mae_raw"] = float(np.mean(np.abs(residuals_raw)))

        per_month_rows.append(row)

        oof_data = {
            "MVT_ID_mvt": mvt_ids[val_mask].values,
            "fold_month": fold_month,
            "prediction": preds.astype("float64"),
        }
        if y_raw is not None:
            oof_data["actual_raw"] = y_raw[val_mask].values.astype("float64")
        oof_preds.append(pd.DataFrame(oof_data))

        log.info("  Fold %s: raw_RMSE=%.2f best_iter=%d", fold_month,
                 row.get("rmse_raw", float("nan")), best_iter)

    per_month_df = pd.DataFrame(per_month_rows)
    all_oof = pd.concat(oof_preds, ignore_index=True)

    overall_rmse_raw = 0.0
    overall_mae_raw = 0.0
    if y_raw is not None:
        residuals_raw = all_oof["prediction"] - all_oof["actual_raw"]
        overall_rmse_raw = float(np.sqrt(np.mean(residuals_raw ** 2)))
        overall_mae_raw = float(np.mean(np.abs(residuals_raw)))

    log.info("Family=%s: overall raw RMSE=%.2f (MAE=%.2f)",
             family, overall_rmse_raw, overall_mae_raw)

    return TuningCVResult(
        per_month=per_month_df,
        overall_rmse_raw=overall_rmse_raw,
        overall_mae_raw=overall_mae_raw,
        oof_predictions=all_oof,
    )
