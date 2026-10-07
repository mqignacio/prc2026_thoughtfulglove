#!/usr/bin/env python3
"""Run only the CatBoost family for phase 1 (LGBM/XGB already done)."""

import sys, os, logging, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("catboost_only")

from pdc26 import SEED
from pdc26.features import CAT_FEATURES
from pdc26.tuning import (
    build_d4_features, family_default_params, month_blocked_cv_model,
)
from run_tuning import DataBundle, compute_per_airport_rmse

t0 = time.time()
bundle = DataBundle()
X = bundle.build_X()
log.info("D4: %s x %d", f"{X.shape[0]:,}", X.shape[1])

airport_map = dict(zip(bundle.mvt_ids.values, bundle.airports.values))
params = family_default_params("cat", seed=SEED)

cv = month_blocked_cv_model(
    X=X, y=bundle.y_winsor, months=bundle.months, mvt_ids=bundle.mvt_ids,
    cat_features=CAT_FEATURES, family="cat", params=params,
    n_estimators=500, early_stopping_rounds=50, y_raw=bundle.raw_target,
)
log.info("CATBOOST overall raw RMSE=%.2f (MAE=%.2f)", cv.overall_rmse_raw, cv.overall_mae_raw)

per_airport = compute_per_airport_rmse(cv.oof_predictions, airport_map)
log.info("Per-airport (top 5):")
for _, r in per_airport.head(5).iterrows():
    log.info("  %s: %.2f", r["airport"], r["rmse_raw"])

log.info("Done in %.0fs", time.time() - t0)
