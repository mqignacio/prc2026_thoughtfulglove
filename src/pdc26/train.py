"""LightGBM trainer and ranking scorer for the PRC-OpenSky 2026 baseline.

Trains on full 2025 data, saves model to models/baseline_lgbm.txt,
and provides scoring function for ranking.parquet DEP rows.
"""

from __future__ import annotations

import logging
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")


def train_full_model(
    X: pd.DataFrame,
    y: pd.Series,
    cat_features: list[str],
    params: dict,
    n_estimators: int = 500,
) -> lgb.Booster:
    """Train LightGBM on the full training dataset (all months).

    Args:
        X: Feature matrix.
        y: Target series (clipped).
        cat_features: Columns treated as native LightGBM categoricals.
        params: LightGBM parameters.
        n_estimators: Number of boosting rounds.

    Returns:
        Trained Booster object.
    """
    log.info("Training full model on %s rows × %d features", f"{len(X):,}", X.shape[1])

    dtrain = lgb.Dataset(
        X,
        label=y,
        categorical_feature=cat_features,
        free_raw_data=False,
    )

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
    )

    log.info("Full model trained: %d trees", model.num_trees())
    return model


def save_model(model: lgb.Booster, filename: str = "baseline_lgbm.txt") -> str:
    """Save booster to text format.

    Args:
        model: Trained LightGBM Booster.
        filename: Output filename (within models/ directory).

    Returns:
        Absolute path to saved model file.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    model.save_model(path)
    log.info("Model saved to %s", path)
    return path


def load_model(path: str | None = None) -> lgb.Booster:
    """Load a saved LightGBM booster.

    Args:
        path: Model file path. Defaults to models/baseline_lgbm.txt.

    Returns:
        Loaded Booster.
    """
    if path is None:
        path = os.path.join(MODELS_DIR, "baseline_lgbm.txt")
    model = lgb.Booster(model_file=path)
    log.info("Model loaded from %s (%d trees)", path, model.num_trees())
    return model


def score_ranking(
    model: lgb.Booster,
    X_ranking: pd.DataFrame,
    mvt_ids: pd.Series,
    clip_lo: float = 1.0,
    clip_hi: float | None = None,
) -> pd.DataFrame:
    """Score ranking DEP rows using the trained model.

    Args:
        model: Trained LightGBM Booster.
        X_ranking: Feature matrix for ranking rows (same columns as training).
        mvt_ids: MVT_ID_mvt series for the ranking rows.
        clip_lo: Lower bound for predictions (default 1.0).
        clip_hi: Upper bound for predictions (default None = no upper clip).

    Returns:
        DataFrame with columns: MVT_ID_mvt, predicted_taxi_sec.

    Raises:
        ValueError: If any predictions are null.
    """
    log.info("Scoring %s ranking DEP rows", f"{len(X_ranking):,}")

    preds = model.predict(X_ranking)

    # Clip predictions to valid range
    preds = np.maximum(preds, clip_lo)
    if clip_hi is not None:
        preds = np.minimum(preds, clip_hi)

    result = pd.DataFrame({
        "MVT_ID_mvt": mvt_ids.values,
        "predicted_taxi_sec": preds.astype("float64"),
    })

    n_null = result["predicted_taxi_sec"].isna().sum()
    if n_null > 0:
        raise ValueError(f"Scoring produced {n_null} null predictions — expected zero!")

    log.info("Scoring complete: mean=%.1fs  median=%.1fs  min=%.1f  max=%.1f",
             result["predicted_taxi_sec"].mean(),
             result["predicted_taxi_sec"].median(),
             result["predicted_taxi_sec"].min(),
             result["predicted_taxi_sec"].max())

    return result


def get_default_params(seed: int = 42) -> dict:
    """Return default-ish LightGBM parameters per task spec.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        Dict of LightGBM parameters.
    """
    return {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
        "num_threads": 0,  # use all available cores
    }
