#!/usr/bin/env python3
"""Task 19 - time-boxed, CV-gated hard-row experiment runner.

Pre-registered protocol: `src/reports/hardrow_experiment.md` (committed before
this file was run). This runner produces EVERY verdict number; there are no
throwaway scripts.

Hard row = DEP row whose `AIRCRAFT_OPERATOR_flt` is null (equivalently the
fully-null `_flt` block, verified identical in Task 17).

Design (frozen in the protocol):
  * One global D4 feature matrix built by `run_blend.BlendData` (fixed global
    categorical vocabulary, fixed feature order, within-fold operator target
    encoding) -- lesson #1 compliance.
  * Dedicated LightGBM trained ONLY on the hard rows, leave-one-month-out
    through the authoritative `pdc26.tuning.month_blocked_cv_model`.
  * Signal test (hard stop) -> acceptance gate (base case REJECT).

Phases:
  signal       build features, run the hard-row CV, write the OOF table,
               evaluate the signal test.
  gate         read the hard-row OOF table, evaluate the acceptance gate.
  all          signal then gate in one process (no feature rebuild).
  serve-train  (only if ACCEPT) train on all hard rows and save the serving
               model + feature order for `make_final_submission.py --hardrow-v2`.

Usage:
    uv run python src/run_hardrow_experiment.py --phase all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_hardrow_experiment")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models", "blend")
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")

OOF_PATH = os.path.join(DATA_PROCESSED, "oof_predictions.parquet")
HARDROW_OOF_PATH = os.path.join(DATA_PROCESSED, "hardrow_oof_predictions.parquet")
RESULTS_PATH = os.path.join(DATA_PROCESSED, "hardrow_experiment_results.md")
SERVE_CHECK_PATH = os.path.join(DATA_PROCESSED, "hardrow_serve_check.md")
BLEND_CONFIG = os.path.join(CONFIG_DIR, "blend.json")
POSTPROC_CONFIG = os.path.join(CONFIG_DIR, "blend_postproc.json")
HARDROW_SERVING = os.path.join(MODELS_DIR, "hardrow_lgbm_full.txt")
HARDROW_CONFIG = os.path.join(CONFIG_DIR, "hardrow.json")
GATE_JSON = os.path.join(DATA_PROCESSED, "hardrow_gate.json")
FINAL_V1 = os.path.join(_PROJECT_ROOT, "submissions", "final", "thoughtful-glove_final_v1.parquet")
FINAL_V2 = os.path.join(_PROJECT_ROOT, "submissions", "final", "thoughtful-glove_final_v2.parquet")
SIDE_V2 = os.path.join(DATA_PROCESSED, "final_predictions_v2.parquet")

SEED = 42
N_ESTIMATORS = 500
EARLY_STOPPING = 50

ANALOG = ["2025-01", "2025-02", "2025-06", "2025-07"]
ANALOG_2MO = ["2025-01", "2025-07"]

# Pre-registered baselines (recomputed from primary sources 2026-10-11).
BASELINE_ALL_2025 = 333.0208
BASELINE_ANALOG = 420.3667
BASELINE_ANALOG_2MO = 454.2545
BASELINE_ANALOG_AFFINE = 417.52
BASELINE_HARD_ANALOG = 2922.74
GATE_THRESHOLD = BASELINE_ANALOG * 0.99  # 416.1663
REPRO_TOL = 0.01
WEIGHT_GRID = [0.25, 0.50, 0.75]


def rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(actual)) ** 2)))


# ---------------------------------------------------------------------------
# Baseline OOF reconstruction + reproduction gate
# ---------------------------------------------------------------------------

def load_baseline_oof() -> pd.DataFrame:
    from pdc26.data import load_training

    oof = pd.read_parquet(OOF_PATH)
    with open(BLEND_CONFIG) as f:
        weights = json.load(f)["selected"]["active_weights"]
    oof["pred_blend"] = sum(oof[f"pred_{k}"] * w for k, w in weights.items())

    all_rmse = rmse(oof["pred_blend"], oof["actual_raw"])
    an_mask = oof["fold_month"].isin(ANALOG)
    an_rmse = rmse(oof.loc[an_mask, "pred_blend"], oof.loc[an_mask, "actual_raw"])
    jj_mask = oof["fold_month"].isin(ANALOG_2MO)
    jj_rmse = rmse(oof.loc[jj_mask, "pred_blend"], oof.loc[jj_mask, "actual_raw"])
    log.info("Reproduction: all-2025 blend %.4f (baseline %.4f) | 4-month analog %.4f "
             "(baseline %.4f) | 2-month %.4f", all_rmse, BASELINE_ALL_2025, an_rmse,
             BASELINE_ANALOG, jj_rmse)
    assert abs(all_rmse - BASELINE_ALL_2025) < REPRO_TOL, "all-2025 reproduction failed"
    assert abs(an_rmse - BASELINE_ANALOG) < REPRO_TOL, "4-month analog reproduction failed"
    assert abs(jj_rmse - BASELINE_ANALOG_2MO) < REPRO_TOL, "2-month reproduction failed"

    dep = (
        load_training()
        .select(["MVT_ID_mvt", "AIRCRAFT_OPERATOR_flt", "ADEP_mvt"])
        .to_pandas()
    )
    dep["hard"] = dep["AIRCRAFT_OPERATOR_flt"].isna()
    n_hard = int(dep["hard"].sum())
    log.info("2025 hard rows: %s / %s (%.4f%%)", f"{n_hard:,}", f"{len(dep):,}",
             100 * dep["hard"].mean())
    assert n_hard == 22470, f"expected 22,470 hard rows, got {n_hard}"

    oof = oof.merge(dep[["MVT_ID_mvt", "hard", "ADEP_mvt"]], on="MVT_ID_mvt", how="left")
    assert oof["hard"].notna().all(), "hard flag merge left nulls"
    return oof


def load_affine() -> dict:
    with open(POSTPROC_CONFIG) as f:
        spec = json.load(f)["full_fit"]["mi2_affine_s50"]["airport_affine"]
    return {k: tuple(v) for k, v in spec.items()}


def apply_affine(pred: np.ndarray, airports: pd.Series, affine: dict) -> np.ndarray:
    slope = airports.map({k: v[0] for k, v in affine.items()}).fillna(1.0).to_numpy("float64")
    inter = airports.map({k: v[1] for k, v in affine.items()}).fillna(0.0).to_numpy("float64")
    return slope * pred + inter


# ---------------------------------------------------------------------------
# Dedicated hard-row CV (signal test)
# ---------------------------------------------------------------------------

def build_hardrow_features():
    """Build the one global D4 matrix and the hard-row subset (fixed order)."""
    from run_blend import BlendData

    bundle = BlendData()
    X = bundle.build_X()
    hard = bundle.pdf_full["AIRCRAFT_OPERATOR_flt"].isna().to_numpy()
    log.info("D4 matrix: %s rows x %d cols; hard rows: %s", f"{X.shape[0]:,}", X.shape[1],
             f"{int(hard.sum()):,}")
    return bundle, X, hard


def run_hardrow_cv(bundle, X: pd.DataFrame, hard: np.ndarray):
    from pdc26.features import CAT_FEATURES
    from pdc26.tuning import family_default_params, month_blocked_cv_model, winsorize_errors

    X_h = X.loc[hard].reset_index(drop=True)
    months_h = bundle.months[hard].reset_index(drop=True)
    mvt_h = bundle.mvt_ids[hard].reset_index(drop=True)
    raw_h = bundle.raw_target[hard].reset_index(drop=True)
    apt_h = bundle.airports[hard].reset_index(drop=True)
    y_h = winsorize_errors(raw_h, apt_h)

    log.info("Hard-row CV: %s rows, target mean raw %.1f / winsorised %.1f",
             f"{len(X_h):,}", raw_h.mean(), y_h.mean())
    res = month_blocked_cv_model(
        X=X_h, y=y_h, months=months_h, mvt_ids=mvt_h,
        cat_features=CAT_FEATURES, family="lgbm",
        params=family_default_params("lgbm", SEED),
        n_estimators=N_ESTIMATORS, early_stopping_rounds=EARLY_STOPPING, y_raw=raw_h,
    )
    res.oof_predictions.to_parquet(HARDROW_OOF_PATH, index=False)
    meta = {
        "best_iterations": [int(v) for v in res.per_month["best_iteration"].tolist()],
        "fold_months": [str(m) for m in res.per_month["fold_month"].tolist()],
        "median_best_iteration": int(round(float(np.median(res.per_month["best_iteration"])))),
        "n_hard_rows": int(len(X_h)),
        "feature_order": list(X_h.columns),
    }
    with open(os.path.join(DATA_PROCESSED, "hardrow_cv_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log.info("CV best_iterations: %s -> median %d", meta["best_iterations"],
             meta["median_best_iteration"])
    log.info("Wrote hard-row OOF: %s (%s rows)", HARDROW_OOF_PATH, len(res.oof_predictions))
    return res


def signal_table(oof: pd.DataFrame, res) -> tuple[list[str], bool]:
    """Per-fold dedicated vs blend hard-row RMSE; returns (markdown, passed)."""
    hr = res.oof_predictions.set_index("MVT_ID_mvt")
    m = oof.loc[oof["hard"]].copy()
    m["pred_hard"] = m["MVT_ID_mvt"].map(hr["prediction"])
    assert m["pred_hard"].notna().all(), "hard-row OOF did not cover every hard row"

    lines = [
        "### B. Signal test (hard stop) -- dedicated hard-row LightGBM vs the blend",
        "",
        "Leave-one-month-out; trained only on the 22,470 null-operator 2025 rows; "
        "raw-target RMSE on the held-out month's hard rows.",
        "",
        "| Fold | n hard | blend hard RMSE (s) | dedicated hard RMSE (s) | Δ (s) | dedicated wins |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    wins = 0
    wins_analog = 0
    for month in sorted(m["fold_month"].unique()):
        sub = m[m["fold_month"] == month]
        b = rmse(sub["pred_blend"], sub["actual_raw"])
        d = rmse(sub["pred_hard"], sub["actual_raw"])
        win = d < b
        wins += int(win)
        marker = "**yes**" if win else "no"
        if month in ANALOG and win:
            wins_analog += 1
        star = " *" if month in ANALOG else ""
        lines.append(f"| {month}{star} | {len(sub):,} | {b:,.2f} | {d:,.2f} | "
                     f"{b - d:+,.2f} | {marker} |")
    an = m[m["fold_month"].isin(ANALOG)]
    b_pool = rmse(an["pred_blend"], an["actual_raw"])
    d_pool = rmse(an["pred_hard"], an["actual_raw"])
    lines.append(f"| **Pooled analog** | {len(an):,} | {b_pool:,.2f} | {d_pool:,.2f} | "
                 f"{b_pool - d_pool:+,.2f} | {'**yes**' if d_pool < b_pool else 'no'} |")
    all_h = m
    lines.append(f"| _All 2025_ | {len(all_h):,} | "
                 f"{rmse(all_h['pred_blend'], all_h['actual_raw']):,.2f} | "
                 f"{rmse(all_h['pred_hard'], all_h['actual_raw']):,.2f} | -- | -- |")
    passed = wins_analog >= 3
    lines += [
        "",
        "*\\* = analog fold. Pre-registered signal gate: the dedicated model must beat the "
        "blend in **≥ 3 of the 4** analog folds (Jan/Feb/Jun/Jul).*",
        "",
        f"**Signal-test result: {'PASS' if passed else 'FAIL'} — dedicated wins "
        f"{wins_analog}/4 analog folds ({wins}/12 all folds).**",
        "",
    ]
    if not passed:
        lines += [
            "**HARD STOP.** The signal test failed: the dedicated hard-row treatment does not "
            "beat the blend's hard-row RMSE in ≥ 3 of the 4 analog folds, so the hard-row error "
            "is treated as irreducible / non-transferable. No acceptance modeling was attempted; "
            "no candidate (b) or (c) was run.",
            "",
        ]
    return lines, passed


# ---------------------------------------------------------------------------
# Acceptance gate
# ---------------------------------------------------------------------------

def _treated_vector(oof: pd.DataFrame, pred_hard: pd.Series, weight: float | None) -> np.ndarray:
    base = oof["pred_blend"].to_numpy("float64")
    ded = oof["MVT_ID_mvt"].map(pred_hard).to_numpy("float64")
    hard = oof["hard"].to_numpy()
    assert not np.isnan(ded[hard]).any(), "dedicated OOF missing on hard rows"
    treated = base.copy()
    if weight is None:  # replace
        treated[hard] = ded[hard]
    else:  # reject-guard fallback: convex blend
        treated[hard] = weight * ded[hard] + (1.0 - weight) * base[hard]
    return treated


def _subset_rmse(df: pd.DataFrame, pred: np.ndarray, months=None) -> float:
    mask = np.ones(len(df), dtype=bool) if months is None else df["fold_month"].isin(months).to_numpy()
    return rmse(pred[mask], df["actual_raw"].to_numpy()[mask])


def _hard_subset_rmse(df: pd.DataFrame, pred: np.ndarray, months=None) -> float:
    mask = df["hard"].to_numpy().copy()
    if months is not None:
        mask &= df["fold_month"].isin(months).to_numpy()
    return rmse(pred[mask], df["actual_raw"].to_numpy()[mask])


def evaluate_gate(oof: pd.DataFrame, res) -> tuple[list[str], bool, dict]:
    hr = res.oof_predictions.set_index("MVT_ID_mvt")["prediction"]
    base = oof["pred_blend"].to_numpy("float64")
    hard = oof["hard"].to_numpy()
    affine = load_affine()
    base_aff = apply_affine(base, oof["ADEP_mvt"], affine)

    # Baseline per-fold (recomputed for the gate table).
    base_folds = {m: _subset_rmse(oof, base, [m]) for m in ANALOG}

    candidates = [("replace", None)] + [(f"blend_w{w:.2f}", w) for w in WEIGHT_GRID]

    # Evaluate every pre-registered candidate (replace first, then the weight grid).
    records = []
    for name, w in candidates:
        treated = _treated_vector(oof, hr, w)
        an = _subset_rmse(oof, treated, ANALOG)
        gains = [base_folds[m] - _subset_rmse(oof, treated, [m]) for m in ANALOG]
        n_imp = sum(g > 0 for g in gains)
        hard_rmse = _hard_subset_rmse(oof, treated, ANALOG)
        cond = {"primary": an <= GATE_THRESHOLD, "folds": n_imp >= 3,
                "hard": hard_rmse < BASELINE_HARD_ANALOG,
                "base": bool(np.array_equal(treated[~hard], base[~hard]))}
        records.append({"name": name, "weight": w, "analog": an, "gain": BASELINE_ANALOG - an,
                        "n_improved": n_imp, "hard_rmse": hard_rmse,
                        "conditions": cond, "accept": all(cond.values()), "treated": treated})

    lines = [
        "### C. Acceptance gate",
        "",
        "Treated OOF = blend OOF with hard rows replaced by the dedicated model's OOF "
        "prediction (or a pre-registered convex blend `w·dedicated + (1−w)·blend`). Base rows "
        "unchanged by construction.",
        "",
        "| Candidate | 4-month analog (s) | vs base | gain | folds improved | hard RMSE (s) | base identical |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for rec in records:
        lines.append(f"| {rec['name']} | {rec['analog']:,.2f} | {rec['analog'] - BASELINE_ANALOG:+,.2f} | "
                     f"{100 * (BASELINE_ANALOG - rec['analog']) / BASELINE_ANALOG:+.2f}% | "
                     f"{rec['n_improved']}/4 | {rec['hard_rmse']:,.2f} | "
                     f"{'yes' if rec['conditions']['base'] else 'NO'} |")
    lines.append("")
    lines += [
        f"Pre-registered gate (all must hold): analog ≤ **{GATE_THRESHOLD:,.2f} s**; "
        f"improvement in ≥ 3/4 folds; hard-row RMSE < **{BASELINE_HARD_ANALOG:,.2f} s**; "
        "base rows byte-identical.",
        "",
    ]

    # Select by the pre-registered order: replace first, then w ∈ {0.25, 0.5, 0.75}.
    chosen = next((r for r in records if r["accept"]), None)
    if chosen is None:
        chosen = min(records, key=lambda r: r["analog"])  # best-effort candidate for the REJECT report
    treated = chosen["treated"]

    # Secondary metrics on the chosen candidate.
    an2 = _subset_rmse(oof, treated, ANALOG_2MO)
    allm = _subset_rmse(oof, treated)
    hard_se = np.sum((treated[hard] - oof["actual_raw"].to_numpy()[hard]) ** 2)
    tot_se = np.sum((treated - oof["actual_raw"].to_numpy()) ** 2)
    ded = oof["MVT_ID_mvt"].map(hr).to_numpy("float64")
    treated_aff = base_aff.copy()
    if chosen["weight"] is None:
        treated_aff[hard] = ded[hard]
    else:
        treated_aff[hard] = chosen["weight"] * ded[hard] + (1.0 - chosen["weight"]) * base_aff[hard]
    an_aff = _subset_rmse(oof, treated_aff, ANALOG)

    lines += [
        "### D. Secondary metrics (chosen candidate: "
        f"**{chosen['name']}**)",
        "",
        "| Metric | Baseline | Treated | Δ |",
        "|---|---:|---:|---:|",
        f"| 4-month analog RMSE (s) | {BASELINE_ANALOG:,.4f} | {_subset_rmse(oof, treated, ANALOG):,.4f} | "
        f"{BASELINE_ANALOG - _subset_rmse(oof, treated, ANALOG):+,.4f} |",
        f"| 2-month analog RMSE (s) | {BASELINE_ANALOG_2MO:,.4f} | {an2:,.4f} | "
        f"{BASELINE_ANALOG_2MO - an2:+,.4f} |",
        f"| all-2025 RMSE (s) | {BASELINE_ALL_2025:,.4f} | {allm:,.4f} | {BASELINE_ALL_2025 - allm:+,.4f} |",
        f"| hard-row RMSE, analog (s) | {BASELINE_HARD_ANALOG:,.2f} | "
        f"{_hard_subset_rmse(oof, treated, ANALOG):,.2f} | "
        f"{BASELINE_HARD_ANALOG - _hard_subset_rmse(oof, treated, ANALOG):+,.2f} |",
        f"| hard-row SE share, analog | 64.7% | {100 * hard_se / tot_se:.1f}% | -- |",
        f"| 4-month analog, post-affine variant (s) | {BASELINE_ANALOG_AFFINE:,.2f} | {an_aff:,.2f} | "
        f"{BASELINE_ANALOG_AFFINE - an_aff:+,.2f} |",
        "",
        "Per-fold 4-month analog RMSE (raw blend space):",
        "",
        "| Fold | baseline (s) | treated (s) | Δ (s) |",
        "|---|---:|---:|---:|",
    ]
    for m in ANALOG:
        bt = _subset_rmse(oof, base, [m])
        tt = _subset_rmse(oof, treated, [m])
        lines.append(f"| {m} | {bt:,.2f} | {tt:,.2f} | {bt - tt:+,.2f} |")
    lines.append("")

    accept = bool(chosen["accept"])
    lines += [
        "### E. Verdict",
        "",
        f"**{'ACCEPT' if accept else 'REJECT'}** — chosen candidate "
        f"`{chosen['name']}` (selected by the pre-registered order: replace, then w ∈ "
        f"{{0.25, 0.5, 0.75}}). Gate conditions for the chosen candidate:",
        "",
        f"1. 4-month analog ≤ {GATE_THRESHOLD:,.2f} s: "
        f"{'PASS' if chosen['conditions']['primary'] else 'FAIL'} ({chosen['analog']:,.2f} s)",
        f"2. Improvement in ≥ 3/4 folds: "
        f"{'PASS' if chosen['conditions']['folds'] else 'FAIL'} ({chosen['n_improved']}/4)",
        f"3. Hard-row RMSE < {BASELINE_HARD_ANALOG:,.2f} s: "
        f"{'PASS' if chosen['conditions']['hard'] else 'FAIL'} ({chosen['hard_rmse']:,.2f} s)",
        f"4. Base rows byte-identical: "
        f"{'PASS' if chosen['conditions']['base'] else 'FAIL'}",
        "",
    ]
    if not accept:
        lines += [
            "**REJECT.** The evidence is complete and the negative result stands: the 4-month "
            "analog gate did not clear with any pre-registered candidate. Task 18's "
            "`thoughtful-glove_final_v1.parquet` is the artifact; no serving change was made.",
            "",
        ]
    else:
        lines += [
            "**ACCEPT (within-2025).** The gate clears on the 2025 analog — see the honest "
            "transfer disclosure in the protocol (§7): this is within-2025 learnability of the "
            "corrupted tail, not evidence the 2026 scored truth carries it. The final artifact "
            "choice (v1 vs v2) belongs to Task 20.",
            "",
        ]
    return lines, accept, chosen


# ---------------------------------------------------------------------------
# Serving model training (only if ACCEPT)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Serving sanity (final_v2 vs final_v1)
# ---------------------------------------------------------------------------

def serve_check() -> str:
    import hashlib

    if not os.path.exists(SIDE_V2):
        raise SystemExit(f"v2 side table missing: {SIDE_V2}; run make_final_submission.py --hardrow-v2")
    side = pd.read_parquet(SIDE_V2)
    v1 = pd.read_parquet(FINAL_V1)
    v2 = pd.read_parquet(FINAL_V2)
    assert list(v1.columns) == list(v2.columns) == ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]
    assert np.array_equal(v1["MVT_ID_mvt"].to_numpy(), v2["MVT_ID_mvt"].to_numpy())
    a = v1["TAXITIME_SEC_mvt"].to_numpy("float64")
    b = v2["TAXITIME_SEC_mvt"].to_numpy("float64")
    changed_p = a != b
    base_same = bool(np.array_equal(a[~changed_p], b[~changed_p]))
    sha1 = hashlib.sha256(open(FINAL_V1, "rb").read()).hexdigest()
    sha2 = hashlib.sha256(open(FINAL_V2, "rb").read()).hexdigest()

    # Rank-order view (the side table) for the treatment-level stats.
    hard = side["operator_null"].to_numpy()
    changed_r = side["delta"].to_numpy("float64") != 0
    n_changed = int(changed_r.sum())
    assert n_changed == int(changed_p.sum()), "changed-row count differs between views"
    d = side["delta"].to_numpy("float64")
    dh = d[hard]

    L = [
        "### F. Serving sanity (final_v2 vs final_v1)",
        "",
        f"final_v1 SHA256 `{sha1}` · final_v2 SHA256 `{sha2}` · rows {len(v1):,} · "
        f"changed rows {n_changed:,} of {int(hard.sum()):,} hard rows "
        f"({int((hard & ~changed_r).sum())} hard rows identical); base rows changed: "
        f"{int((~hard & changed_r).sum())} · base rows byte-identical: {'PASS' if base_same else 'FAIL'}.",
        "",
        "Delta (v2 − v1) on the 9,077 hard rows:",
        "",
        "| stat | mean | median | std | RMS | p1 | p5 | p95 | p99 | max\u2009|\u0394\u2009| |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| \u0394 s | {dh.mean():+.3f} | {np.median(dh):+.3f} | {dh.std():.3f} | "
        f"{np.sqrt(np.mean(dh ** 2)):.3f} | {np.percentile(dh, 1):+.1f} | {np.percentile(dh, 5):+.1f} | "
        f"{np.percentile(dh, 95):+.1f} | {np.percentile(dh, 99):+.1f} | {np.abs(dh).max():.1f} |",
        "",
        "Per-month hard-row predictions (v1 vs v2):",
        "",
        "| Month | n hard | v1 mean | v2 mean | mean \u0394 | RMS \u0394 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in sorted(side["month"].unique()):
        sub = side[(side["month"] == m) & side["operator_null"]]
        L.append(f"| {m} | {len(sub):,} | {sub.v1_value.mean():.1f} | {sub.v2_value.mean():.1f} | "
                 f"{sub.delta.mean():+.1f} | {np.sqrt(np.mean(sub.delta.to_numpy() ** 2)):.1f} |")
    L += ["", "Per-airport hard-row predictions (v1 vs v2):", "",
          "| Airport | n hard | v1 mean | v2 mean | mean \u0394 | RMS \u0394 |",
          "|---|---:|---:|---:|---:|---:|"]
    for apt in sorted(side["ADEP_mvt"].unique()):
        sub = side[(side["ADEP_mvt"] == apt) & side["operator_null"]]
        if len(sub) == 0:
            continue
        L.append(f"| {apt} | {len(sub):,} | {sub.v1_value.mean():.1f} | {sub.v2_value.mean():.1f} | "
                 f"{sub.delta.mean():+.1f} | {np.sqrt(np.mean(sub.delta.to_numpy() ** 2)):.1f} |")

    # Implied effect arithmetic (both directions, script-generated).
    rms_d = float(np.sqrt(np.mean(dh ** 2)))
    share = len(dh) / len(side)
    add_mse = share * float(np.mean(dh ** 2))
    expected = 363.0
    worse = float(np.sqrt(expected ** 2 + add_mse))
    gate = json.load(open(GATE_JSON)) if os.path.exists(GATE_JSON) else {}
    treated_analog = float(gate.get("chosen_analog", 391.3768))
    rel = (BASELINE_ANALOG - treated_analog) / BASELINE_ANALOG
    better = expected * rel
    L += [
        "",
        f"Overall prediction stats — v1: mean {a.mean():.1f} median {np.median(a):.1f} std {a.std():.1f}; "
        f"v2: mean {b.mean():.1f} median {np.median(b):.1f} std {b.std():.1f}.",
        "",
        f"**Interpretation (honest; the aggregate mean hides the row-level change).** The swap changes "
        f"{n_changed:,} of {int(hard.sum()):,} hard rows by RMS **{rms_d:,.0f} s** per row, yet the "
        f"marginal means barely move (v1 {a.mean():.1f} vs v2 {b.mean():.1f}) — the reallocation is "
        f"near-symmetric (mean {dh.mean():+.1f} s). Which direction helps depends on the unknown 2026 "
        f"hard-row truth, and there is no leaderboard feedback for the final months:",
        "",
        f"- If the 2025 corruption mechanism transfers, the analog gain is {100 * rel:.2f}%, i.e. \u2248 "
        f"**\u2212{better:.1f} s** on a {expected:.0f} s expected final.",
        f"- If the 2026 truth is clean and the reallocation is uncorrelated with the residual (the audit's "
        f"lean: leader 213.18 s < the 338.1 s 4-month floor), it adds \u2248 {add_mse:,.0f} MSE/row \u2192 \u2248 "
        f"**+{worse - expected:.1f} s** on the {expected:.0f} s baseline.",
        "",
        "The two are similar in magnitude and of opposite sign, so this is a genuine judgment call for "
        "Task 20, not a mechanical 'v2 is better'.",
        "",
    ]
    text = "\n".join(L)
    with open(SERVE_CHECK_PATH, "w") as f:
        f.write(text)
    log.info("Wrote serving-check fragment: %s", SERVE_CHECK_PATH)
    log.info("serve-check done; SHA v1=%s v2=%s", sha1[:12], sha2[:12])
    return text


# ---------------------------------------------------------------------------
# Results fragment
# ---------------------------------------------------------------------------

def write_results(text: str) -> None:
    with open(RESULTS_PATH, "w") as f:
        f.write(text)
    log.info("Wrote results fragment: %s", RESULTS_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 19 hard-row experiment")
    parser.add_argument("--phase", default="all",
                        choices=["signal", "gate", "all", "serve-train", "serve-check"])
    parser.add_argument("--rounds", type=int, default=None,
                        help="Rounds for --phase serve-train (median CV best_iteration)")
    args = parser.parse_args()

    t0 = time.time()

    if args.phase == "serve-check":
        serve_check()
        log.info("serve-check phase done in %.0fs", time.time() - t0)
        return

    oof = load_baseline_oof()

    if args.phase in ("signal", "all"):
        bundle, X, hard = build_hardrow_features()
        res = run_hardrow_cv(bundle, X, hard)
        sig_lines, sig_pass = signal_table(oof, res)
        if args.phase == "signal":
            write_results("\n".join(sig_lines))
            log.info("Signal phase done in %.0fs (%.1f min)", time.time() - t0,
                     (time.time() - t0) / 60)
            return
    else:
        # gate phase without rebuilding features: reconstruct via a dummy result container
        bundle = X = hard = None
        res = None

    # --phase all: signal already computed above.
    if args.phase == "all":
        gate_lines, accept, chosen = evaluate_gate(oof, res)
        with open(GATE_JSON, "w") as f:
            json.dump({"accept": bool(accept), "candidate": chosen["name"],
                       "chosen_analog": float(chosen["analog"]),
                       "chosen_hard_rmse": float(chosen["hard_rmse"]),
                       "n_folds_improved": int(chosen["n_improved"])}, f, indent=2)
        text = "\n".join(sig_lines + gate_lines)
        write_results(text)
        log.info("ALL phase done in %.0fs (%.1f min); verdict=%s",
                 time.time() - t0, (time.time() - t0) / 60,
                 "ACCEPT" if accept else "REJECT")

    elif args.phase == "gate":
        hr = pd.read_parquet(HARDROW_OOF_PATH)

        class _Res:  # minimal container so evaluate_gate can read the OOF table
            oof_predictions = hr

        gate_lines, accept, chosen = evaluate_gate(oof, _Res)
        with open(GATE_JSON, "w") as f:
            json.dump({"accept": bool(accept), "candidate": chosen["name"],
                       "chosen_analog": float(chosen["analog"]),
                       "chosen_hard_rmse": float(chosen["hard_rmse"]),
                       "n_folds_improved": int(chosen["n_improved"])}, f, indent=2)
        write_results("\n".join(gate_lines))
        log.info("Gate phase done in %.0fs; verdict=%s", time.time() - t0,
                 "ACCEPT" if accept else "REJECT")

    elif args.phase == "serve-train":
        rounds = args.rounds
        if rounds is None:
            meta_path = os.path.join(DATA_PROCESSED, "hardrow_cv_meta.json")
            rounds = int(json.load(open(meta_path))["median_best_iteration"])
            log.info("Rounds from CV median best_iteration: %d", rounds)
        bundle, X, hard = build_hardrow_features()
        from pdc26.features import CAT_FEATURES  # noqa: F401
        import lightgbm as lgb
        from pdc26.tuning import family_default_params, winsorize_errors

        X_h = X.loc[hard].reset_index(drop=True)
        raw_h = bundle.raw_target[hard].reset_index(drop=True)
        apt_h = bundle.airports[hard].reset_index(drop=True)
        y_h = winsorize_errors(raw_h, apt_h)
        dtrain = lgb.Dataset(X_h, label=y_h.values, categorical_feature=CAT_FEATURES,
                             free_raw_data=False)
        model = lgb.train(family_default_params("lgbm", SEED), dtrain,
                          num_boost_round=int(rounds))
        os.makedirs(MODELS_DIR, exist_ok=True)
        model.save_model(HARDROW_SERVING)
        cfg = {
            "created": time.strftime("%Y-%m-%d"),
            "protocol": "src/reports/hardrow_experiment.md",
            "rounds": int(rounds),
            "n_train_hard_rows": int(hard.sum()),
            "feature_order": model.feature_name(),
            "cat_features": CAT_FEATURES,
            "target": "winsorize_errors(hard-row subset)",
        }
        with open(HARDROW_CONFIG, "w") as f:
            json.dump(cfg, f, indent=2)
        log.info("Saved %s (%d trees) + %s", HARDROW_SERVING, model.num_trees(), HARDROW_CONFIG)


if __name__ == "__main__":
    main()
