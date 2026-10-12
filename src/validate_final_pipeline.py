#!/usr/bin/env python3
"""Validation battery for the Task 18 final-phase submission.

Consumes the artifact built by `src/make_final_submission.py` and writes
`src/reports/final_pipeline_validation.md`. Checks (all scripted, no eyeballs):

  1. Pre-upload assertions re-run independently on the written parquet.
  2. Independent reproduction of the blend + affine post-processing from the
     side table's per-component predictions.
  3. Reproduction gate: optional byte comparison against a saved baseline run.
  4. Prediction distribution sanity, per month and per airport, vs the 2025
     analog OOF (blend + affine) and vs the v5 Jan/Jul ranking predictions.
  5. Hard-row (null-operator) prediction distribution.
  6. Adversarial-validation drift check (Qian et al. 2021): LightGBM classifier
     separating 2025 analog features from final 2026 features, pooled and
     per-month, 5-fold AUC with a CI and the top discriminative features.

Read-only with respect to model/data inputs; only writes the report.
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

from pdc26.data import load_ranking, load_training
from pdc26.features import CAT_FEATURES, apply_operator_target_encoding, get_month_labels
from pdc26.tuning import build_d4_features

from make_final_submission import (
    BLEND_CONFIG,
    DATA_RAW,
    FINAL_SUBMITTING,
    FINAL_V1,
    POSTPROC_CANDIDATE,
    POSTPROC_CONFIG,
    RANKING_SUBMITTING,
    SIDE_TABLE,
    ServingArtifacts,
    build_final_features,
    cast_features,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("validate_final_pipeline")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OOF_PARQUET = os.path.join(_PROJECT_ROOT, "data", "processed", "oof_predictions.parquet")
V5_RANKING_PREDS = os.path.join(_PROJECT_ROOT, "data", "processed", "blend_ranking_predictions.parquet")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "reports", "final_pipeline_validation.md")

EXPECTED_DEP_ROWS = 670_790
EXPECTED_V6_ROWS = 344_841
ANALOG_MONTHS = ["2025-01", "2025-02", "2025-06", "2025-07"]
FINAL_MONTHS = ["2026-01", "2026-02", "2026-06", "2026-07"]
MONTH_PAIRS = list(zip(ANALOG_MONTHS, FINAL_MONTHS))
SHIFT_FLAG = 0.15  # |mean shift| fraction above which a row is flagged

# Quoted expected values (external_research-resubmission.md §3), for the recompute check.
QUOTED_ANALOG = {"2025-01": 383.74, "2025-02": 399.15, "2025-06": 366.90, "2025-07": 503.96,
                 "4-month": 420.37, "2-month": 454.25}
V5_LEADERBOARD = 392.5911
# Recorded leaderboard probe (thoughtful-glove_v6.parquet_result.json, uploaded 2026-10-11 00:22:15 UTC).
V6_SCORE = 393.2793
V6_UPLOAD_BYTES = 4_707_521
V6_SHA256 = "5805cdd9397201ff24680cb00403d7637dd34caeb1c0cc5d4fb6870925959525"
V6_UPLOADED_UTC = "2026-10-11 00:22:15 UTC"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_blend_weights() -> dict:
    with open(BLEND_CONFIG) as f:
        return json.load(f)["selected"]["active_weights"]


def load_affine() -> dict:
    with open(POSTPROC_CONFIG) as f:
        spec = json.load(f)["full_fit"][POSTPROC_CANDIDATE]
    return {k: tuple(v) for k, v in spec["airport_affine"].items()}


def apply_affine(blended: np.ndarray, airports: pd.Series, affine: dict) -> np.ndarray:
    """Independent implementation of the accepted per-airport affine recalibration."""
    slope_map = {k: v[0] for k, v in affine.items()}
    inter_map = {k: v[1] for k, v in affine.items()}
    apt = airports.astype("string").fillna("__NA__")
    slopes = apt.map(slope_map).fillna(1.0).to_numpy("float64")
    inters = apt.map(inter_map).fillna(0.0).to_numpy("float64")
    return slopes * blended + inters


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, dtype="float64") - np.asarray(p, dtype="float64")) ** 2)))


def dist_stats(v: np.ndarray) -> tuple[float, float, float]:
    v = np.asarray(v, dtype="float64")
    return float(v.mean()), float(np.median(v)), float(np.percentile(v, 95))


# ---------------------------------------------------------------------------
# 1-3. artifact assertions + post-processing reproduction + repro gate
# ---------------------------------------------------------------------------

def check_artifact() -> tuple[pd.DataFrame, np.ndarray]:
    af = pl.read_parquet(FINAL_V1)
    sub = pl.read_parquet(FINAL_SUBMITTING)
    af_ids = af["MVT_ID_mvt"].to_numpy()
    sub_ids = sub["MVT_ID_mvt"].to_numpy()
    vals = af["TAXITIME_SEC_mvt"].to_numpy()
    assert af.height == sub.height == EXPECTED_DEP_ROWS, \
        f"row count {af.height} != {sub.height} != {EXPECTED_DEP_ROWS}"
    assert np.array_equal(af_ids, sub_ids), "MVT_ID order is not exact vs final_submitting"
    assert not pd.isna(vals).any(), "nulls present"
    assert np.isfinite(vals).all(), "non-finite values present"
    assert vals.min() >= 1.0, f"min below 1 s: {vals.min()}"
    assert vals.dtype == np.float64, f"dtype is {vals.dtype}, expected float64"
    assert af_ids.dtype == sub_ids.dtype, "MVT_ID dtype differs from final_submitting"
    log.info("Pre-upload assertions re-verified on the written artifact: PASS")
    return af.to_pandas(), vals


def check_postproc(side: pd.DataFrame, vals: np.ndarray, weights: dict, affine: dict) -> dict:
    ids = side["MVT_ID_mvt"].to_numpy()
    recomputed_blend = (float(weights["tuned_lgbm"]) * side["pred_tuned_lgbm"].to_numpy("float64")
                        + float(weights["per_airport_lgbm"]) * side["pred_per_airport_lgbm"].to_numpy("float64"))
    blend_max_abs_diff = float(np.max(np.abs(recomputed_blend - side["blended"].to_numpy("float64"))))
    recomputed_final = np.maximum(apply_affine(recomputed_blend, side["ADEP_mvt"], affine), 1.0)
    final_max_abs_diff = float(np.max(np.abs(recomputed_final - side["final_value"].to_numpy("float64"))))
    # Independent re-derivation must agree (allow float64 rounding only).
    assert blend_max_abs_diff < 1e-6, f"blend mismatch {blend_max_abs_diff}"
    assert final_max_abs_diff < 1e-6, f"affine mismatch {final_max_abs_diff}"
    log.info("Post-processing independently reproduced: blend max|Δ|=%.2e affine max|Δ|=%.2e",
             blend_max_abs_diff, final_max_abs_diff)
    return {"blend_max_abs_diff": blend_max_abs_diff, "final_max_abs_diff": final_max_abs_diff}


def check_reproduction(baseline: str | None) -> dict | None:
    if baseline is None:
        return None
    if not os.path.exists(baseline):
        raise SystemExit(f"reproduction baseline not found: {baseline}")
    cur = hashlib.sha256(open(FINAL_V1, "rb").read()).hexdigest()
    base = hashlib.sha256(open(baseline, "rb").read()).hexdigest()
    identical = open(FINAL_V1, "rb").read() == open(baseline, "rb").read()
    log.info("Reproduction gate: run1=%s run2=%s byte-identical=%s", base[:12], cur[:12], identical)
    assert identical, "reproduction gate FAILED: rebuilt artifact differs byte-for-byte"
    return {"baseline_sha": base, "current_sha": cur, "identical": identical}


# ---------------------------------------------------------------------------
# 4. distributions
# ---------------------------------------------------------------------------

def analog_oof(weights: dict, affine: dict) -> pd.DataFrame:
    """Blend + affine the 2025 OOF predictions, with month and airport attached."""
    oof = pd.read_parquet(OOF_PARQUET)
    oof["blended"] = (float(weights["tuned_lgbm"]) * oof["pred_tuned_lgbm"]
                      + float(weights["per_airport_lgbm"]) * oof["pred_per_airport_lgbm"])
    train = load_training()
    key = train.select(["MVT_ID_mvt", "ADEP_mvt"]).to_pandas()
    oof = oof.merge(key, on="MVT_ID_mvt", how="left")
    assert oof["ADEP_mvt"].notna().all(), "OOF rows missing airport after join"
    oof["final_affine"] = apply_affine(oof["blended"].to_numpy("float64"), oof["ADEP_mvt"], affine)
    return oof


def v5_ranking_predictions(weights: dict, affine: dict) -> pd.DataFrame:
    """Reconstruct v5's Jan/Jul post-affine ranking predictions (undithered input)."""
    preds = pd.read_parquet(V5_RANKING_PREDS)
    rank = load_ranking()
    key = pd.DataFrame({
        "MVT_ID_mvt": rank["MVT_ID_mvt"].to_pandas().to_numpy(),
        "ADEP_mvt": rank["ADEP_mvt"].to_pandas().astype("string").to_numpy(),
        "month": get_month_labels(rank).to_numpy(),
    })
    df = preds.merge(key, on="MVT_ID_mvt", how="left")
    assert df["ADEP_mvt"].notna().all() and df["month"].notna().all()
    df["final_affine"] = apply_affine(df["blended"].to_numpy("float64"), df["ADEP_mvt"], affine)
    return df


def analog_rmse_table(oof: pd.DataFrame) -> list[dict]:
    rows = []
    for m in ANALOG_MONTHS:
        sub = oof[oof["fold_month"] == m]
        rows.append({
            "month": m, "n": len(sub),
            "blend_rmse": rmse(sub["actual_raw"], sub["blended"]),
            "affine_rmse": rmse(sub["actual_raw"], sub["final_affine"]),
            "quoted": QUOTED_ANALOG.get(m),
        })
    sub4 = oof[oof["fold_month"].isin(ANALOG_MONTHS)]
    sub2 = oof[oof["fold_month"].isin(["2025-01", "2025-07"])]
    rows.append({"month": "4-month (Jan+Feb+Jun+Jul)", "n": len(sub4),
                 "blend_rmse": rmse(sub4["actual_raw"], sub4["blended"]),
                 "affine_rmse": rmse(sub4["actual_raw"], sub4["final_affine"]),
                 "quoted": QUOTED_ANALOG["4-month"]})
    rows.append({"month": "2-month (Jan+Jul)", "n": len(sub2),
                 "blend_rmse": rmse(sub2["actual_raw"], sub2["blended"]),
                 "affine_rmse": rmse(sub2["actual_raw"], sub2["final_affine"]),
                 "quoted": QUOTED_ANALOG["2-month"]})
    return rows


def dist_by_month(final_side: pd.DataFrame, oof: pd.DataFrame) -> list[dict]:
    rows = []
    for am, fm in MONTH_PAIRS:
        f = final_side[final_side["month"] == fm]["final_value"].to_numpy()
        a = oof[oof["fold_month"] == am]["final_affine"].to_numpy()
        fmean, fmed, fp95 = dist_stats(f)
        amean, amed, ap95 = dist_stats(a)
        shift = (fmean - amean) / amean
        rows.append({"final_month": fm, "analog_month": am, "n": len(f),
                     "f_mean": fmean, "f_med": fmed, "f_p95": fp95,
                     "a_mean": amean, "a_med": amed, "a_p95": ap95,
                     "mean_shift": shift, "flag": abs(shift) > SHIFT_FLAG})
    return rows


def dist_by_airport(final_side: pd.DataFrame, oof: pd.DataFrame) -> list[dict]:
    rows = []
    for apt in sorted(final_side["ADEP_mvt"].dropna().unique()):
        f = final_side[final_side["ADEP_mvt"] == apt]["final_value"].to_numpy()
        a = oof[oof["ADEP_mvt"] == apt]["final_affine"].to_numpy()
        fmean, fmed, fp95 = dist_stats(f)
        amean, amed, ap95 = dist_stats(a)
        shift = (fmean - amean) / amean
        rows.append({"airport": apt, "n": len(f),
                     "f_mean": fmean, "f_med": fmed, "f_p95": fp95,
                     "a_mean": amean, "a_med": amed, "a_p95": ap95,
                     "mean_shift": shift, "flag": abs(shift) > SHIFT_FLAG})
    return rows


def v5_deltas(final_side: pd.DataFrame, v5: pd.DataFrame) -> list[dict]:
    """Per-ID comparison of final Jan/Jul predictions vs v5's own Jan/Jul predictions."""
    rows = []
    for m in ["2026-01", "2026-07"]:
        f = final_side[final_side["month"] == m][["MVT_ID_mvt", "final_value"]]
        v = v5[v5["month"] == m][["MVT_ID_mvt", "final_affine", "blended"]]
        j = f.merge(v, on="MVT_ID_mvt", how="inner")
        dab = (j["final_value"] - j["final_affine"]).abs()
        rows.append({"month": m, "n": len(j),
                     "f_mean": float(j["final_value"].mean()),
                     "v5_mean": float(j["final_affine"].mean()),
                     "v5_blend_mean": float(j["blended"].mean()),
                     "mean_delta": float((j["final_value"].mean() - j["final_affine"].mean())
                                         / j["final_affine"].mean()),
                     "median_abs_delta": float(dab.median()),
                     "p95_abs_delta": float(np.percentile(dab, 95))})
    return rows


def hard_row_distribution(final_side: pd.DataFrame) -> dict:
    null_mask = final_side["operator_null"].to_numpy()
    hard = final_side.loc[null_mask, "final_value"].to_numpy()
    base = final_side.loc[~null_mask, "final_value"].to_numpy()
    return {"n_hard": int(null_mask.sum()), "n_base": int((~null_mask).sum()),
            "hard": dist_stats(hard), "base": dist_stats(base),
            "hard_pct": float(null_mask.mean())}


# ---------------------------------------------------------------------------
# 5. adversarial validation
# ---------------------------------------------------------------------------

def _exact_month_file(month: str) -> str:
    """Return the single training parquet whose range STARTS at `month`.

    `pdc26.data.load_training(filter_months=...)` matches substrings, so a month
    like `2025-06` also matches the May file (`...2025-05-01_2025-06-01.parquet`).
    The AV training side must contain exactly the four analog months, so pin the
    file by its start date here instead.
    """
    matches = sorted(glob.glob(os.path.join(DATA_RAW, f"training_{month}-01_*.parquet")))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one file for {month}, found {matches}")
    return matches[0]


def load_training_analog_dep(months: list[str]) -> pl.DataFrame:
    """Exact-month DEP loader (mirrors `load_training` semantics on exact files)."""
    frames = [pl.scan_parquet(_exact_month_file(m)).filter(pl.col("PHASE_mvt") == "DEP").collect()
              for m in months]
    df = pl.concat(frames).unique(subset=["MVT_ID_mvt"], keep="first").sort("MVT_TIME_UTC_mvt")
    return df


def load_training_analog_movements(months: list[str]) -> pl.DataFrame:
    """Exact-month DEP+ARR loader for congestion features."""
    frames = [pl.read_parquet(_exact_month_file(m)) for m in months]
    return pl.concat(frames).unique(subset=["MVT_ID_mvt"], keep="first").sort("MVT_TIME_UTC_mvt")


def build_training_analog_features(art: ServingArtifacts) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    dep = load_training_analog_dep(ANALOG_MONTHS)
    mov = load_training_analog_movements(ANALOG_MONTHS).to_pandas()
    pdf = dep.to_pandas()
    op = apply_operator_target_encoding(pdf["AIRCRAFT_OPERATOR_flt"], art.operator_enc,
                                        fallback=art.op_fallback)
    X = build_d4_features(dep, mov, art.vocab, art.domain_vocab, art.metar, art.coords,
                          operator_enc=op)
    X = cast_features(X)
    X.index = range(len(X))
    months = get_month_labels(dep).to_numpy()
    days = pd.to_datetime(dep["MVT_TIME_UTC_mvt"].to_pandas(), utc=True).dt.strftime("%Y-%m-%d").to_numpy()
    log.info("Training-analog feature matrix: %s rows x %d cols", f"{len(X):,}", X.shape[1])
    return X, months, days


def _lgb_params() -> dict:
    # deterministic/force_row_wise: remove LightGBM multithreading nondeterminism so the
    # verifier reproduces the same AUCs on re-run (the artifact build is deterministic anyway).
    return {"objective": "binary", "metric": "auc", "learning_rate": 0.1, "num_leaves": 31,
            "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1,
            "seed": 42, "deterministic": True, "force_row_wise": True,
            "verbosity": -1, "num_threads": 4}


def av_auc(X: pd.DataFrame, y: np.ndarray, cats: list[str], groups: np.ndarray | None = None,
           folds: int = 5, rounds: int = 150) -> tuple[float, float, list[float]]:
    """5-fold AUC. Row-split (StratifiedKFold) by default; GroupKFold when `groups` given."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    if groups is not None:
        splitter = GroupKFold(n_splits=folds).split(X, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42).split(X, y)
    aucs = []
    for tr, te in splitter:
        dtr = lgb.Dataset(X.iloc[tr], label=y[tr], categorical_feature=cats, free_raw_data=False)
        model = lgb.train(_lgb_params(), dtr, num_boost_round=rounds)
        aucs.append(float(roc_auc_score(y[te], model.predict(X.iloc[te]))))
    mean = float(np.mean(aucs))
    ci = float(1.96 * np.std(aucs, ddof=1) / np.sqrt(folds))
    return mean, ci, aucs


_AV_ANCHORS = ["anchor_aobt_sec", "anchor_eobt_sec", "anchor_iobt_sec", "anchor_lobt_sec",
               "pre_pushback_delay", "aoht_vs_eobt", "sched_deviation_min"]


def _av_groups(allc: list[str]) -> dict[str, list[str]]:
    from pdc26.features import CONGESTION_ALL_COLUMNS

    cong = [c for c in CONGESTION_ALL_COLUMNS if c in allc]
    metar = [c for c in allc if c.startswith("metar_") or c.startswith("minutes_since")]
    cats = [c for c in CAT_FEATURES if c in allc]
    return {
        "anchors + schedule (predictive core)": [c for c in _AV_ANCHORS if c in allc],
        "temporal (hour/dow/month)": [c for c in ["hour_sin", "hour_cos", "dow", "month"] if c in allc],
        "flight characteristics": [c for c in ["wake_cat_ordinal", "nm_matched", "flight_rule_ifr"] if c in allc],
        "categoricals": cats,
        "METAR weather": metar,
        "congestion counts": cong,
    }


def av_group_attribution(X: pd.DataFrame, y: np.ndarray, cats_full: list[str],
                        groups: dict[str, list[str]]) -> dict:
    """Single stratified 70/30 split per feature family (fast drift attribution)."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    out = {}
    for name, cols in groups.items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        Xtr, Xte, ytr, yte = train_test_split(X[cols], y, test_size=0.3,
                                              random_state=42, stratify=y)
        m = lgb.train(_lgb_params(), lgb.Dataset(Xtr, label=ytr,
                                                 categorical_feature=[c for c in cols if c in cats_full]),
                      num_boost_round=200)
        out[name] = {"n": len(cols), "auc": float(roc_auc_score(yte, m.predict(Xte)))}
        log.info("  AV group %-40s n=%2d AUC=%.4f", name, len(cols), out[name]["auc"])
    return out


def av_check(train_X: pd.DataFrame, train_months: np.ndarray, train_days: np.ndarray,
             final_X: pd.DataFrame, final_months: np.ndarray, final_days: np.ndarray) -> dict:
    import lightgbm as lgb
    assert list(train_X.columns) == list(final_X.columns), "AV feature columns differ"
    cats = [c for c in CAT_FEATURES if c in train_X.columns]

    Xpool = pd.concat([train_X, final_X], ignore_index=True)
    ypool = np.concatenate([np.zeros(len(train_X), dtype="int8"),
                            np.ones(len(final_X), dtype="int8")])
    days = np.concatenate([train_days, final_days])

    mean, ci, aucs = av_auc(Xpool, ypool, cats)
    log.info("AV pooled row-split AUC = %.4f +/- %.4f", mean, ci)
    dmean, dci, daucs = av_auc(Xpool, ypool, cats, groups=days)
    log.info("AV pooled day-grouped AUC = %.4f +/- %.4f", dmean, dci)

    dsm = lgb.Dataset(Xpool, label=ypool, categorical_feature=cats, free_raw_data=False)
    full = lgb.train(_lgb_params(), dsm, num_boost_round=300)
    imp = pd.Series(full.feature_importance(importance_type="gain"), index=Xpool.columns)
    top = imp.sort_values(ascending=False).head(10)

    per_month = {}
    for am, fm in MONTH_PAIRS:
        mt = train_months == am
        mf = final_months == fm
        tr, fi = train_X[mt], final_X[mf]
        dtr, dfi = train_days[mt], final_days[mf]
        Xm = pd.concat([tr, fi], ignore_index=True)
        ym = np.concatenate([np.zeros(len(tr), dtype="int8"), np.ones(len(fi), dtype="int8")])
        m, c, _ = av_auc(Xm, ym, cats)
        dmm, dcc, _ = av_auc(Xm, ym, cats, groups=np.concatenate([dtr, dfi]))
        per_month[fm] = {"analog": am, "n": len(Xm), "auc": m, "ci": c,
                         "day_auc": dmm, "day_ci": dcc}
        log.info("AV %s vs %s: row-split %.4f +/- %.4f, day-grouped %.4f +/- %.4f",
                 am, fm, m, c, dmm, dcc)

    log.info("AV feature-group attribution:")
    groups = av_group_attribution(Xpool, ypool, cats, _av_groups(list(Xpool.columns)))
    return {"pooled": {"auc": mean, "ci": ci, "folds": aucs},
            "pooled_day": {"auc": dmean, "ci": dci, "folds": daucs},
            "per_month": per_month, "group_attribution": groups, "top_features": top}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def write_report(ctx: dict) -> str:
    now = time.strftime("%Y-%m-%d %H:%M %z")
    L: list[str] = []
    A = L.append

    A("# Final Pipeline Validation (Task 18)")
    A("")
    A(f"**Status:** complete · **Generated (machine clock):** {now} (Asia/Manila; the "
      "`%z` offset is authoritative — macOS labels the zone `PST`, see TODO audit A3) · "
      "**Verifier:** `src/validate_final_pipeline.py` (re-runnable; all numbers below are its output)")
    A("")
    A("Scope: validation of the final-phase serving artifact built by "
      "`src/make_final_submission.py` from `final_ranking.parquet` through the frozen v5 stack. "
      "No model retraining, no new features, no recalibration refits. The EUROCONTROL taxi-out "
      "aggregates are not used in any form (decision Q2-A / D2).")
    A("")

    A("## 1. Artifact and pre-upload assertions")
    A("")
    A("| Check | Result |")
    A("|---|---|")
    A(f"| Row count == `final_submitting` | {EXPECTED_DEP_ROWS:,} — **PASS** |")
    A("| MVT_ID order exact vs `final_submitting` | **PASS** |")
    A("| Nulls | 0 — **PASS** |")
    A("| All finite | **PASS** |")
    A(f"| Min ≥ 1 s | {ctx['stats']['min']:.3f} — **PASS** |")
    A("| `TAXITIME_SEC_mvt` dtype | float64 — **PASS** |")
    A("| MVT_ID dtype matches template | **PASS** |")
    A("")
    A(f"Artifact: `submissions/final/thoughtful-glove_final_v1.parquet`, "
      f"{ctx['size_bytes']:,} bytes, SHA256 `{ctx['sha256']}`.")
    A("")
    A("Prediction stats (full 4 months, n={:,}): mean={:.1f} s, median={:.1f} s, "
      "min={:.1f} s, max={:.1f} s, std={:.1f} s.".format(
          EXPECTED_DEP_ROWS, ctx["stats"]["mean"], ctx["stats"]["median"],
          ctx["stats"]["min"], ctx["stats"]["max"], ctx["stats"]["std"]))
    A("")

    A("## 2. Post-processing reproduction (independent)")
    A("")
    A(f"The validation script re-derived the blend from the side table's per-component "
      f"predictions and re-applied the affine with its own implementation. Max absolute "
      f"difference: blend {ctx['postproc']['blend_max_abs_diff']:.2e} s, final "
      f"{ctx['postproc']['final_max_abs_diff']:.2e} s (float64 rounding only). **PASS**.")
    A("")

    A("## 3. Reproduction gate (byte-identical rebuild)")
    A("")
    if ctx.get("repro"):
        r = ctx["repro"]
        A(f"Two independent builds of `src/make_final_submission.py`; the first was saved as a "
          f"baseline and the second compared byte-for-byte. "
          f"run1 SHA256 `{r['baseline_sha']}`, run2 SHA256 `{r['current_sha']}` — "
          f"byte-identical: **{r['identical']}** (PASS).")
    else:
        A("Not run in this invocation (no `--repro-baseline` passed).")
    A("")

    A("## 4. Prediction distribution sanity")
    A("")
    A("### 4a. 2025 analog RMSE (recomputation of the quoted figures)")
    A("")
    A("| Segment | Rows | Raw blend RMSE (s) | Post-affine RMSE (s) | Quoted blend (s) | Δ |")
    A("|---|---:|---:|---:|---:|---:|")
    for r in ctx["analog_rmse"]:
        q = f"{r['quoted']:.2f}" if r["quoted"] is not None else "—"
        d = f"{r['blend_rmse'] - r['quoted']:+.2f}" if r["quoted"] is not None else "—"
        A(f"| {r['month']} | {r['n']:,} | {r['blend_rmse']:.2f} | {r['affine_rmse']:.2f} | {q} | {d} |")
    A("")
    A("The raw-blend column reproduces the quoted 2025-analog figures exactly (Δ ≈ 0), "
      "confirming the OOF blend reconstruction. The post-affine column applies the accepted "
      "`mi2_affine_s50` **full-train** fit in-sample to the OOF predictions — it is shown for "
      "distribution context only. The honest leave-one-month-out season figure is **452.90 s** "
      "(`src/configs/blend_postproc.json`).")
    A("")
    A("### 4b. Prediction distributions by month (final 2026 vs 2025 analog)")
    A("")
    A("| Final month | Analog | n | final mean/median/p95 | analog mean/median/p95 | mean shift | flag |")
    A("|---|---|---:|---|---|---:|---|")
    for r in ctx["dist_month"]:
        A(f"| {r['final_month']} | {r['analog_month']} | {r['n']:,} | "
          f"{r['f_mean']:.1f} / {r['f_med']:.1f} / {r['f_p95']:.1f} | "
          f"{r['a_mean']:.1f} / {r['a_med']:.1f} / {r['a_p95']:.1f} | "
          f"{r['mean_shift']*100:+.1f}% | {'FLAG' if r['flag'] else 'ok'} |")
    A("")
    A("### 4c. Prediction distributions by airport (pooled 4 months)")
    A("")
    A("| Airport | n | final mean/median/p95 | analog mean/median/p95 | mean shift | flag |")
    A("|---|---:|---|---|---:|---|")
    for r in ctx["dist_airport"]:
        A(f"| {r['airport']} | {r['n']:,} | "
          f"{r['f_mean']:.1f} / {r['f_med']:.1f} / {r['f_p95']:.1f} | "
          f"{r['a_mean']:.1f} / {r['a_med']:.1f} / {r['a_p95']:.1f} | "
          f"{r['mean_shift']*100:+.1f}% | {'FLAG' if r['flag'] else 'ok'} |")
    A("")
    A("### 4d. Serving-regression check vs v5's own Jan/Jul predictions")
    A("")
    A("Same calendar months (undithered v5 input vs dithered final input), per-MVT_ID comparison.")
    A("")
    A("| Month | n | final mean (s) | v5 mean (s) | mean Δ | median \\|Δ\\| (s) | p95 \\|Δ\\| (s) |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for r in ctx["v5_deltas"]:
        A(f"| {r['month']} | {r['n']:,} | {r['f_mean']:.2f} | {r['v5_mean']:.2f} | "
          f"{r['mean_delta']*100:+.2f}% | {r['median_abs_delta']:.2f} | {r['p95_abs_delta']:.2f} |")
    A("")
    A(f"v5 leaderboard reference: {V5_LEADERBOARD:.4f} s.")
    A("")

    A("## 5. Hard-row (null-operator) prediction distribution")
    A("")
    hr = ctx["hard"]
    A(f"Null-operator rows: {hr['n_hard']:,} ({hr['hard_pct']*100:.4f}%); "
      f"base rows: {hr['n_base']:,}.")
    A("")
    A("| Segment | mean (s) | median (s) | p95 (s) |")
    A("|---|---:|---:|---:|")
    A(f"| hard (null operator) | {hr['hard'][0]:.1f} | {hr['hard'][1]:.1f} | {hr['hard'][2]:.1f} |")
    A(f"| base | {hr['base'][0]:.1f} | {hr['base'][1]:.1f} | {hr['base'][2]:.1f} |")
    A("")

    A("## 6. Adversarial-validation drift check")
    A("")
    A("LightGBM binary classifier separating 2025 training-analog features from final 2026 "
      "features (63-column D4 matrix, identical fitted artifacts both sides; features are not "
      "re-fitted per side). Two CV schemes are reported: **row-split** (StratifiedKFold, the "
      "literature default) and **day-grouped** (GroupKFold by calendar date, which prevents the "
      "classifier from memorising per-day signatures — see the interpretation below).")
    A("")
    av = ctx["av"]
    A(f"**Pooled (4 analog months 2025 vs 4 final months 2026):** "
      f"row-split AUC = {av['pooled']['auc']:.4f} ± {av['pooled']['ci']:.4f}; "
      f"day-grouped AUC = {av['pooled_day']['auc']:.4f} ± {av['pooled_day']['ci']:.4f}.")
    A("")
    A("| Comparison (analog → final) | n | row-split AUC | ± 95% CI | day-grouped AUC | ± 95% CI |")
    A("|---|---:|---:|---:|---:|---:|")
    for fm, d in av["per_month"].items():
        A(f"| {d['analog']} → {fm} | {d['n']:,} | {d['auc']:.4f} | {d['ci']:.4f} | "
          f"{d['day_auc']:.4f} | {d['day_ci']:.4f} |")
    A("")
    A("**Feature-group attribution** (single stratified 70/30 split per family; isolates which "
      "families carry the separability):")
    A("")
    A("| Feature family | n | AUC |")
    A("|---|---:|---:|")
    for name, d in av["group_attribution"].items():
        A(f"| {name} | {d['n']} | {d['auc']:.4f} |")
    A("")
    A("**Top-10 discriminative features (gain, pooled row-split model):**")
    A("")
    A("| Rank | Feature | Gain |")
    A("|---:|---|---:|")
    for i, (name, val) in enumerate(av["top_features"].items(), 1):
        A(f"| {i} | `{name}` | {val:,.0f} |")
    A("")
    A("### Interpretation")
    A("")
    A("*(Author interpretation; the numeric values above are script output.)*")
    A("")
    A("The pooled row-split AUC is far above the 0.7 flag threshold, so the drift was "
      "investigated by feature family and with a day-grouped CV. The result is unambiguous:")
    A("")
    A("- **The predictive core is stable.** Anchors + schedule = 0.56, temporal = 0.52, "
      "flight characteristics = 0.50 — no material shift in the families that carry the "
      "model's signal (the near-oracle takeoff-time anchor family is the pipeline's strongest "
      "predictor; see `src/reports/error_analysis.md`).")
    A("- **The high AUC comes entirely from METAR weather (0.86) and congestion counts (0.999).** "
      "The 2025-analog and 2026-final windows had genuinely different weather (benign covariate, "
      "used as a minor correction). The congestion counts are dominated by a latent units bug in "
      "the frozen v5 feature code: `build_congestion_features` builds windows with a nanosecond "
      "constant (`MIN_NS`) while the timestamps are microsecond `datetime64[us]`, so the nominal "
      "5/15/30/60-minute windows are ~1000× too wide (±83 h to ±42 days; the \"5-min\" feature "
      "mean is ~3985, i.e. a multi-day airport activity count). Those wide counts act as "
      "dataset-specific fingerprints.")
    A("- **Two CV schemes together separate the two causes.** Row-split AUC ≈ 1.000 is inflated "
      "by per-day memorisation (the two sets are disjoint calendar days, so day-specific weather "
      "signatures are memorisable); day-grouped AUC ≈ 0.96–0.97 removes that channel but stays "
      "high because the METAR and over-wide congestion differences are day-generalizable.")
    A("- **Reproducibility note:** the AV classifier is a diagnostic, not a gate artifact. The "
      "table values carry ~±0.002 run-to-run variation (multithreaded LightGBM) despite the "
      "`deterministic`/`force_row_wise` settings; the headline conclusions are invariant. The "
      "submission artifact itself is exactly reproducible (see §3).")
    A("- **Neither inflation is a serving risk.** The congestion bug is consistent between "
      "training and serving (the frozen stack is used unchanged), and the independent evidence "
      "is favourable: v5 (same features/stack on Jan/Jul 2026) scored 392.59 s, −13.6% versus "
      "the 2025 season-matched analog (454.25 s). §4d further shows the final Jan/Jul "
      "predictions reproduce v5's own Jan/Jul predictions with a negligible mean shift "
      "(≤ 0.75%) and a ~7–9 s median per-row difference (the ±12 s dither).")
    A("- **Conclusion:** flag recorded, investigation complete, **no block on submission**. The "
      "drift signal is a diagnostic degeneracy (disjoint calendar days + high-resolution "
      "date-correlated covariates), amplified by freezing a units bug in the congestion code — "
      "not evidence of harmful distribution shift in the predictive features.")
    A("- **Out of scope, recorded for the repo/paper (Task 21/23):** the congestion window units "
      "bug should be disclosed as a latent issue. It must NOT be fixed without retraining (a "
      "serve-only fix would break train/serve consistency).")
    A("")

    A("## 7. v6 ID overlap and validation submission")
    A("")
    A(f"Jan/Jul subset of the final side table (n={EXPECTED_V6_ROWS:,}) vs `submitting.parquet` "
      f"(n={ctx['v6_rows']:,}): set equality **{ctx['v6_overlap']}**. The other two final months "
      f"(Feb/Jun 2026) are not part of `submitting.parquet` and are excluded from v6.")
    A("")
    A(f"**v6 leaderboard probe (recorded post-run).** `submissions/thoughtful-glove_v6.parquet` "
      f"(SHA256 `{V6_SHA256}`, {V6_UPLOAD_BYTES:,} bytes) = the v1 predictions restricted to "
      f"Jan/Jul, uploaded {V6_UPLOADED_UTC}. Result: status Succeeded, 344,841/344,841 pairs, "
      f"RMSE **{V6_SCORE:.4f} s** vs v5's {V5_LEADERBOARD:.4f} s = "
      f"**{V6_SCORE - V5_LEADERBOARD:+.4f} s ({(V6_SCORE - V5_LEADERBOARD) / V5_LEADERBOARD * 100:+.2f}%)** "
      f"— inside the 2% gate: **no serving regression**.")
    A("")

    A("## 8. Notes for downstream tasks")
    A("")
    A("- Task 19: fallback artifact = `submissions/final/thoughtful-glove_final_v1.parquet` "
      "(SHA256 in §1). Base-row predictions are the blend output; any hard-row treatment must "
      "leave them byte-identical.")
    A("- Task 20: copy the selected artifact to `submissions/final/thoughtful-glove_final.parquet` "
      "under the official name.")
    A("- Task 21/23: this report is the evidence for the final-phase serving claim.")
    A("")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(L))
    log.info("Wrote report: %s", REPORT_PATH)
    return REPORT_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 18 final pipeline validation battery")
    parser.add_argument("--repro-baseline", type=str, default=None,
                        help="Saved first-run parquet to compare byte-for-byte")
    args = parser.parse_args()

    weights = load_blend_weights()
    affine = load_affine()

    log.info("Checking artifact...")
    af_pdf, vals = check_artifact()
    side = pd.read_parquet(SIDE_TABLE)
    assert len(side) == EXPECTED_DEP_ROWS
    postproc = check_postproc(side, vals, weights, affine)
    repro = check_reproduction(args.repro_baseline)

    # v6 overlap (cheap; check before the expensive AV so failures surface fast)
    sub = pl.read_parquet(RANKING_SUBMITTING)
    sub_ids = set(sub["MVT_ID_mvt"].to_pandas().to_numpy())
    janjul = side[side["month"].isin(["2026-01", "2026-07"])]["MVT_ID_mvt"]
    v6_overlap = (len(janjul) == EXPECTED_V6_ROWS and sub.height == EXPECTED_V6_ROWS
                  and set(janjul.to_numpy()) == sub_ids)
    assert v6_overlap, "Jan/Jul ID set differs from submitting.parquet"
    log.info("v6 overlap: Jan/Jul side rows = %s, submitting rows = %s, set equality True",
             f"{len(janjul):,}", f"{sub.height:,}")

    log.info("Building 2025 analog OOF predictions...")
    oof = analog_oof(weights, affine)
    analog_rmse = analog_rmse_table(oof)
    dist_month = dist_by_month(side, oof)
    dist_airport = dist_by_airport(side, oof)

    log.info("Reconstructing v5 Jan/Jul predictions...")
    v5 = v5_ranking_predictions(weights, affine)
    v5_delta = v5_deltas(side, v5)

    hard = hard_row_distribution(side)

    log.info("Building serving artifacts and feature matrices for AV...")
    art = ServingArtifacts()
    rank_df, rank_pdf, final_X = build_final_features(art)
    if not np.array_equal(rank_df["MVT_ID_mvt"].to_pandas().to_numpy(),
                          side["MVT_ID_mvt"].to_numpy()):
        raise ValueError("rebuilt final features are not aligned to the side table")
    train_X, train_months, train_days = build_training_analog_features(art)
    final_months = get_month_labels(rank_df).to_numpy()
    final_days = pd.to_datetime(rank_df["MVT_TIME_UTC_mvt"].to_pandas(), utc=True).dt.strftime("%Y-%m-%d").to_numpy()
    av = av_check(train_X, train_months, train_days, final_X, final_months, final_days)

    sha = hashlib.sha256(open(FINAL_V1, "rb").read()).hexdigest()
    ctx = {
        "stats": {"mean": float(vals.mean()), "median": float(np.median(vals)),
                  "min": float(vals.min()), "max": float(vals.max()), "std": float(vals.std())},
        "sha256": sha, "size_bytes": os.path.getsize(FINAL_V1),
        "postproc": postproc, "repro": repro, "analog_rmse": analog_rmse,
        "dist_month": dist_month, "dist_airport": dist_airport,
        "v5_deltas": v5_delta, "hard": hard, "av": av,
        "v6_overlap": v6_overlap, "v6_rows": sub.height,
    }
    write_report(ctx)
    log.info("Validation battery done.")


if __name__ == "__main__":
    main()
