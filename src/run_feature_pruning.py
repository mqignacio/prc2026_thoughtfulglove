#!/usr/bin/env python3
"""Task 4.1 - feature-pruning check (task goal 3).

Drops the lowest-gain features from the D4 set, re-runs the two active blend
components' month-blocked CV, and confirms whether the simpler model degrades.
The drop list is derived from the saved full-retrain boosters' gain importance
(never hand-typed), and the verdict follows the approved acceptance rule:
a pruning is kept as a justified simplification if OOF degrades <= 0.05%.

Also refits the accepted post-processing (per-airport affine, slope shrunk 50%)
on the pruned OOF with leave-one-month-out, to report the combined verdict.

Writes src/reports/feature_pruning.md, data/processed/oof_predictions_pruned.parquet,
and src/configs/pruned_features.json.

Usage:
    uv run python src/run_feature_pruning.py
"""

from __future__ import annotations

import glob
import json
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from run_blend import BlendData, build_components
from run_error_analysis import add_derived, load_and_join
from run_micro_iterations import _apply_affine, _fit_affine, lot_out, rmse
from pdc26.features import CAT_FEATURES
from pdc26.tuning import month_blocked_cv_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_feature_pruning")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models", "blend")
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")
PRUNED_CONFIG = os.path.join(CONFIG_DIR, "pruned_features.json")
PRUNED_OOF = os.path.join(DATA_PROCESSED, "oof_predictions_pruned.parquet")
REPORT_PATH = os.path.join(REPORT_DIR, "feature_pruning.md")

SEASON_MONTHS = ["2025-01", "2025-07"]
GAIN_DROP_THRESHOLD_PCT = 0.05   # drop features with individual mean gain share below this
ACCEPT_DEGRADE_PCT = 0.05        # keep pruning if OOF degrades no more than this (%)
ACTIVE = ["tuned_lgbm", "per_airport_lgbm"]


def compute_feature_gain() -> pd.DataFrame:
    """Aggregate gain importance across the saved full-retrain boosters."""
    import lightgbm as lgb

    paths = ([os.path.join(MODELS_DIR, "tuned_lgbm_full.txt")]
             + sorted(glob.glob(os.path.join(MODELS_DIR, "per_airport_lgbm_*_full.txt"))))
    gain: dict = {}
    for p in paths:
        b = lgb.Booster(model_file=p)
        for n, v in zip(b.feature_name(), b.feature_importance(importance_type="gain")):
            gain[n] = gain.get(n, 0.0) + float(v)
    df = pd.DataFrame({"feature": list(gain), "gain": [gain[k] for k in gain]})
    df["gain_share_pct"] = 100.0 * df["gain"] / df["gain"].sum()
    return df.sort_values("gain", ascending=False).reset_index(drop=True), len(paths)


def oof_component(bundle, X: pd.DataFrame, comp) -> pd.Series:
    """Month-blocked CV for one component on feature matrix X; returns preds by MVT_ID."""
    cat = [c for c in CAT_FEATURES if c in X.columns]

    def run(Xi, yi, months, ids, yraw):
        res = month_blocked_cv_model(
            X=Xi, y=yi, months=months, mvt_ids=ids, cat_features=cat,
            family=comp.family, params=comp.params, n_estimators=comp.n_estimators,
            early_stopping_rounds=comp.early_stopping, y_raw=yraw,
        )
        return res.oof_predictions

    if comp.mode == "global":
        oof = run(X, bundle.y_winsor, bundle.months, bundle.mvt_ids, bundle.raw_target)
    else:
        frames = []
        for apt in sorted(bundle.airports.unique()):
            idx = np.where(bundle.airports.values == apt)[0]
            f = run(X.iloc[idx].reset_index(drop=True),
                    bundle.y_winsor.iloc[idx].reset_index(drop=True),
                    bundle.months.iloc[idx].reset_index(drop=True),
                    bundle.mvt_ids.iloc[idx].reset_index(drop=True),
                    bundle.raw_target.iloc[idx].reset_index(drop=True))
            frames.append(f)
        oof = pd.concat(frames, ignore_index=True)
    return oof.set_index("MVT_ID_mvt")["prediction"]


def main() -> None:
    t0 = datetime.now()
    base, blend = load_and_join()
    base = add_derived(base)
    W = blend["selected"]["active_weights"]
    baseline_overall = float(blend["selected"]["rmse"])
    actual = base["actual_raw"].to_numpy()
    season = base["month"].isin(SEASON_MONTHS).to_numpy()
    baseline_season = rmse(base["pred_blend"].to_numpy()[season], actual[season])

    gain_df, n_models = compute_feature_gain()
    drop = gain_df[gain_df["gain_share_pct"] < GAIN_DROP_THRESHOLD_PCT]["feature"].tolist()
    kept = gain_df[~gain_df["feature"].isin(drop)]["feature"].tolist()
    drop_cum = float(gain_df[gain_df["feature"].isin(drop)]["gain_share_pct"].sum())
    log.info("Gain from %d models: %d features, dropping %d below %.2f%% (cumulative %.3f%% gain)",
             n_models, len(gain_df), len(drop), GAIN_DROP_THRESHOLD_PCT, drop_cum)
    log.info("Dropping: %s", drop)

    bundle = BlendData()
    X = bundle.build_X()
    present = set(X.columns)
    drop_present = [c for c in drop if c in present]
    Xp = X.drop(columns=drop_present)
    log.info("Feature matrix: %d -> %d cols", X.shape[1], Xp.shape[1])

    comps = build_components()
    preds = {name: oof_component(bundle, Xp, comps[name]) for name in ACTIVE}

    pruned = base[["mvt_id", "airport", "month", "actual_raw", "season_matched"]].copy()
    for name in ACTIVE:
        pruned[f"pred_{name}"] = pruned["mvt_id"].map(preds[name]).astype("float64")
    pruned["pred_blend"] = sum(W[n] * pruned[f"pred_{n}"] for n in ACTIVE)
    assert pruned["pred_blend"].notna().all(), "pruned blend has null predictions"
    pruned.to_parquet(PRUNED_OOF, index=False)

    po = float(np.sqrt(np.mean((pruned["pred_blend"].to_numpy() - actual) ** 2)))
    ps = float(np.sqrt(np.mean((pruned["pred_blend"].to_numpy()[season] - actual[season]) ** 2)))
    d_overall = po - baseline_overall
    d_season = ps - baseline_season
    degrade_pct = 100.0 * d_overall / baseline_overall
    kept_pruning = degrade_pct <= ACCEPT_DEGRADE_PCT and (d_season / baseline_season * 100.0) <= ACCEPT_DEGRADE_PCT

    # Combined: accepted affine (slope shrunk 50%) refit on the pruned OOF, LOT-out.
    months = sorted(pruned["month"].unique())
    key = "airport"
    aff_lot = lot_out(pruned, lambda tr, te: _apply_affine(
        te, key, te["pred_blend"].to_numpy(), _fit_affine(tr, key, tr["pred_blend"].to_numpy(), 0.5)), months)
    aff_overall = rmse(aff_lot, actual)
    aff_season = rmse(aff_lot[season], actual[season])

    full_affine = {k: [float(a), float(b)] for k, (a, b) in
                   _fit_affine(pruned, key, pruned["pred_blend"].to_numpy(), 0.5).items()}

    out = {
        "created": datetime.now().strftime("%Y-%m-%d"),
        "n_models_for_gain": n_models,
        "gain_drop_threshold_pct": GAIN_DROP_THRESHOLD_PCT,
        "dropped_features": drop,
        "dropped_cumulative_gain_pct": drop_cum,
        "kept_features": kept,
        "verdict": "KEEP_PRUNED" if kept_pruning else "REJECT",
        "baseline": {"overall": baseline_overall, "season": baseline_season},
        "pruned": {"overall": po, "season": ps, "delta_overall": -d_overall, "delta_season": -d_season},
        "pruned_plus_affine_s50_lot_out": {"overall": aff_overall, "season": aff_season},
        "full_affine_airport": full_affine,
    }
    with open(PRUNED_CONFIG, "w") as f:
        json.dump(out, f, indent=2)

    write_report(gain_df, drop, drop_cum, kept, baseline_overall, baseline_season,
                 po, ps, d_overall, d_season, aff_overall, aff_season, kept_pruning)
    log.info("Pruned overall %.3f (d %+.3f, %.3f%%)  season %.3f (d %+.3f) -> %s",
             po, -d_overall, degrade_pct, ps, -d_season, out["verdict"])
    log.info("Pruned+affine_s50 LOT-out: overall %.3f  season %.3f", aff_overall, aff_season)
    log.info("Done in %.1f min", (datetime.now() - t0).total_seconds() / 60)


def write_report(gain_df, drop, drop_cum, kept, bo, bs, po, ps, d_overall, d_season,
                 aff_o, aff_s, kept_pruning) -> None:
    now = datetime.now().astimezone()
    L = ["# Feature-Pruning Check — Task 4.1", "",
         f"Generated: {now:%Y-%m-%d %H:%M %Z} · Drop list derived from saved full-retrain "
         f"gainer gain importance ({len(gain_df)} features).", "",
         "## Dropped features (individual gain share < 0.05%)", "",
         "| Feature | Gain share (%) |", "|---|---:|"]
    for _, r in gain_df[gain_df["feature"].isin(drop)].iterrows():
        L.append(f"| `{r['feature']}` | {r['gain_share_pct']:.4f} |")
    L.append(f"")
    L.append(f"**{len(drop)} features dropped, cumulative gain {drop_cum:.3f}%** "
             f"(kept {len(kept)}).")
    L.append("")
    L.append("## CV verdict (two active components, month-blocked)")
    L.append("")
    L.append("| Model | Overall RMSE (s) | Δ vs baseline | Season RMSE (s) | Δ vs baseline |")
    L.append("|---|---:|---:|---:|---:|")
    L.append(f"| baseline blend (62 features) | {bo:.3f} | — | {bs:.3f} | — |")
    L.append(f"| pruned blend ({len(kept)} features) | {po:.3f} | {-d_overall:+.3f} | {ps:.3f} | {-d_season:+.3f} |")
    L.append(f"| pruned + affine s50 (LOT-out) | {aff_o:.3f} | {bo - aff_o:+.3f} | {aff_s:.3f} | {bs - aff_s:+.3f} |")
    L.append("")
    L.append(f"**Verdict: {('KEEP_PRUNED — justified simplification' if kept_pruning else 'REJECT — degrades too much')}**")
    L.append("")
    L.append("_Acceptance rule: pruning kept if OOF degrades <= 0.05% overall and season-matched._")
    L.append("")
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(L))
    log.info("Wrote report: %s", REPORT_PATH)


if __name__ == "__main__":
    main()
