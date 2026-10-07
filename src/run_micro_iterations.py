#!/usr/bin/env python3
"""Task 4.1 - micro-iteration CV verdicts on the blend OOF table.

Every candidate is evaluated with leave-one-month-out (LOT-out) on the 2025 OOF
table: post-processing parameters are fit on the other 11 months and applied to
the held-out month. This mirrors the month-blocked CV design and gives an honest
generalisation estimate (fitting and scoring on the same rows would flatter the
tail-dominated LIRF calibration).

Candidates (one change each, per task discipline):
  baseline_global   recorded blend weights (reproduction gate)
  mi1_weights       per-airport 2-component blend weights
  mi2_affine        per-airport affine recalibration of the global blend
  mi2_affine_s50    same, slope shrunk 50% toward 1 (robustness variant)
  mi2_affine_rwy    per airport-runway affine recalibration
  mi3_weights_affine  per-airport weights + per-airport affine

Writes src/reports/micro_iterations.md and, for every candidate, full-12-month
fit parameters into src/configs/blend_postproc.json (with the LOT-out verdict).

Usage:
    uv run python src/run_micro_iterations.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from run_error_analysis import add_derived, load_and_join

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_micro_iterations")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")
POSTPROC_CONFIG = os.path.join(CONFIG_DIR, "blend_postproc.json")
REPORT_PATH = os.path.join(REPORT_DIR, "micro_iterations.md")

SEASON_MONTHS = ["2025-01", "2025-07"]
# Post-processing is "not degrading" the season-matched metric if within this
# many seconds; feature/model changes need the >=0.1% overall gain rule.
SEASON_TOL = 0.01
FEATURE_GAIN_PCT = 0.1


# ---------------------------------------------------------------------------
# Fit / apply helpers
# ---------------------------------------------------------------------------

def _fit_weights(train: pd.DataFrame, key: str) -> dict:
    """Optimal w per group for pred = w*tuned + (1-w)*per_airport (RMSE-optimal)."""
    params: dict = {}
    for g, sub in train.groupby(key, observed=True):
        y = sub["actual_raw"].to_numpy()
        tv = sub["pred_tuned_lgbm"].to_numpy()
        pv = sub["pred_per_airport_lgbm"].to_numpy()
        d = tv - pv
        denom = float(np.sum(d * d))
        w = float(np.clip(np.sum((y - pv) * d) / denom, 0.0, 1.0)) if denom > 0 else 0.5
        params[g] = w
    return params


def _apply_weights(df: pd.DataFrame, key: str, params: dict, default_w: float) -> np.ndarray:
    w = df[key].map(params).fillna(default_w).to_numpy(dtype=np.float64)
    return w * df["pred_tuned_lgbm"].to_numpy() + (1.0 - w) * df["pred_per_airport_lgbm"].to_numpy()


def _fit_affine(train: pd.DataFrame, key: str, base: np.ndarray, shrink: float) -> dict:
    """Per-group RMSE-optimal affine y ~ a*base + b, slope shrunk toward 1."""
    params: dict = {}
    work = pd.DataFrame({
        "_g": train[key].to_numpy(),
        "_base": base,
        "_y": train["actual_raw"].to_numpy(),
    })
    for g, sub in work.groupby("_g", observed=True):
        bv = sub["_base"].to_numpy()
        yv = sub["_y"].to_numpy()
        var = float(np.var(bv))
        slope = float(np.cov(bv, yv)[0, 1] / var) if var > 0 else 1.0
        slope = 1.0 + shrink * (slope - 1.0)
        intercept = float(np.mean(yv) - slope * np.mean(bv))
        params[g] = (slope, intercept)
    return params


def _apply_affine(df: pd.DataFrame, key: str, base: np.ndarray, params: dict) -> np.ndarray:
    slope_map = {g: a for g, (a, _b) in params.items()}
    inter_map = {g: b for g, (_a, b) in params.items()}
    slopes = df[key].map(slope_map).fillna(1.0).to_numpy(dtype=np.float64)
    intercepts = df[key].map(inter_map).fillna(0.0).to_numpy(dtype=np.float64)
    return slopes * base + intercepts


# ---------------------------------------------------------------------------
# Candidate transforms: fit on train rows, predict test rows
# ---------------------------------------------------------------------------

def make_candidates(global_w: dict) -> dict:
    default_w = float(global_w.get("tuned_lgbm", 0.5))

    def t_global(_train, test):
        return test["pred_blend"].to_numpy()

    def t_weights(train, test):
        params = _fit_weights(train, "airport")
        return _apply_weights(test, "airport", params, default_w)

    def t_affine(train, test, shrink=1.0):
        params = _fit_affine(train, "airport", train["pred_blend"].to_numpy(), shrink)
        return _apply_affine(test, "airport", test["pred_blend"].to_numpy(), params)

    def t_affine_rwy(train, test):
        params = _fit_affine(train, "airport_runway", train["pred_blend"].to_numpy(), 1.0)
        return _apply_affine(test, "airport_runway", test["pred_blend"].to_numpy(), params)

    def t_weights_affine(train, test):
        w = _fit_weights(train, "airport")
        base_tr = _apply_weights(train, "airport", w, default_w)
        base_te = _apply_weights(test, "airport", w, default_w)
        a = _fit_affine(train, "airport", base_tr, 1.0)
        return _apply_affine(test, "airport", base_te, a)

    return {
        "baseline_global": {"fn": t_global, "n_params": 0, "kind": "baseline",
                            "desc": "recorded global blend (reproduction gate)"},
        "mi1_weights": {"fn": t_weights, "n_params": 10, "kind": "postproc",
                        "desc": "per-airport 2-component blend weights"},
        "mi2_affine": {"fn": t_affine, "n_params": 20, "kind": "postproc",
                       "desc": "per-airport affine recalibration of the blend"},
        "mi2_affine_s50": {"fn": lambda tr, te: t_affine(tr, te, shrink=0.5), "n_params": 20,
                           "kind": "postproc", "desc": "per-airport affine, slope shrunk 50%"},
        "mi2_affine_rwy": {"fn": t_affine_rwy, "n_params": None, "kind": "postproc",
                           "desc": "per airport-runway affine recalibration"},
        "mi3_weights_affine": {"fn": t_weights_affine, "n_params": 30, "kind": "postproc",
                               "desc": "per-airport weights + per-airport affine"},
    }


def lot_out(pdf: pd.DataFrame, fn, months: list[str]) -> np.ndarray:
    pred = np.full(len(pdf), np.nan, dtype=np.float64)
    month_arr = pdf["month"].to_numpy()
    for m in months:
        te_mask = month_arr == m
        pred[te_mask] = fn(pdf[~te_mask], pdf[te_mask])
    assert not np.isnan(pred).any(), "LOT-out produced NaNs"
    return pred


def rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = datetime.now()
    pdf, blend = load_and_join()
    pdf = add_derived(pdf)
    global_w = blend["selected"]["active_weights"]
    baseline_overall = float(blend["selected"]["rmse"])

    actual = pdf["actual_raw"].to_numpy()
    months = sorted(pdf["month"].unique())
    season_mask = pdf["month"].isin(SEASON_MONTHS).to_numpy()

    candidates = make_candidates(global_w)
    results = {}
    per_month = {}
    log.info("Evaluating %d candidates with leave-one-month-out...", len(candidates))

    for name, spec in candidates.items():
        pred = lot_out(pdf, spec["fn"], months)
        overall = rmse(pred, actual)
        season = rmse(pred[season_mask], actual[season_mask])
        results[name] = {
            "overall": overall, "season": season,
            "delta_overall": baseline_overall - overall,
            "delta_season": rmse(pdf.loc[season_mask, "pred_blend"].to_numpy(), actual[season_mask]) - season,
            "n_params": spec["n_params"], "kind": spec["kind"], "desc": spec["desc"],
        }
        pm = {}
        month_arr = pdf["month"].to_numpy()
        for m in months:
            mm = month_arr == m
            pm[m] = rmse(pred[mm], actual[mm])
        per_month[name] = pm
        log.info("  %-20s overall %.3f (d %+.3f)  season %.3f", name, overall,
                 results[name]["delta_overall"], season)

    # baseline reference (must reproduce recorded RMSE)
    base_overall = results["baseline_global"]["overall"]
    assert abs(base_overall - baseline_overall) < 0.01, (
        f"baseline LOT-out {base_overall} != recorded {baseline_overall}")
    base_season = results["baseline_global"]["season"]

    # Verdicts per the approved acceptance rule.
    for name, r in results.items():
        if name == "baseline_global":
            r["verdict"] = "reference"
            continue
        improves = r["delta_overall"] > 0
        season_ok = r["delta_season"] >= -SEASON_TOL
        if r["kind"] == "postproc":
            r["verdict"] = "ACCEPT" if (improves and season_ok) else "reject"
        else:
            gain_pct = 100.0 * r["delta_overall"] / base_overall
            r["verdict"] = "ACCEPT" if (gain_pct >= FEATURE_GAIN_PCT and season_ok) else "reject"

    # Full-12-month fits for serving (only for accepted, non-baseline candidates).
    full_fit = {}
    for name, spec in candidates.items():
        if name == "baseline_global" or results[name]["verdict"] != "ACCEPT":
            continue
        train = pdf
        if name == "mi1_weights":
            full_fit[name] = {"type": "weights", "default_w": global_w,
                              "airport_weights": _fit_weights(train, "airport")}
        elif name in ("mi2_affine", "mi2_affine_s50"):
            shrink = 0.5 if name.endswith("s50") else 1.0
            full_fit[name] = {"type": "affine", "shrink": shrink, "base": "global_blend",
                              "airport_affine": _fit_affine(train, "airport",
                                                            train["pred_blend"].to_numpy(), shrink)}
        elif name == "mi2_affine_rwy":
            full_fit[name] = {"type": "affine", "shrink": 1.0, "base": "global_blend",
                              "airport_runway_affine": _fit_affine(
                                  train, "airport_runway", train["pred_blend"].to_numpy(), 1.0)}
        elif name == "mi3_weights_affine":
            w = _fit_weights(train, "airport")
            base = _apply_weights(train, "airport", w, float(global_w.get("tuned_lgbm", 0.5)))
            full_fit[name] = {"type": "weights_affine", "default_w": global_w,
                              "airport_weights": w,
                              "airport_affine": _fit_affine(train, "airport", base, 1.0)}

    out = {
        "created": datetime.now().strftime("%Y-%m-%d"),
        "source": "leave-one-month-out on data/processed/oof_predictions.parquet",
        "baseline": {"overall": base_overall, "season": base_season,
                     "recorded_overall": baseline_overall},
        "candidates": results,
        "accepted": {k: v for k, v in results.items() if v["verdict"] == "ACCEPT"},
        "full_fit": full_fit,
    }
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(POSTPROC_CONFIG, "w") as f:
        json.dump(out, f, indent=2, default=float)
    log.info("Wrote post-processing config: %s", POSTPROC_CONFIG)

    write_report(results, per_month, months, base_overall, base_season, season_mask,
                 pdf, out)
    accepted = [k for k, v in results.items() if v["verdict"] == "ACCEPT"]
    log.info("Accepted candidates: %s", accepted or "none")
    log.info("Done in %.1f min", (datetime.now() - t0).total_seconds() / 60)


def write_report(results, per_month, months, base_overall, base_season, season_mask,
                 pdf, out) -> None:
    now = datetime.now().astimezone()
    L = ["# Micro-Iterations — Task 4.1 (leave-one-month-out CV verdicts)", "",
         f"Generated: {now:%Y-%m-%d %H:%M %Z} · Source: 2025 OOF table, LOT-out "
         f"({len(months)} folds) · Baseline (reproduction gate): overall "
         f"**{base_overall:.2f}s**, season-matched **{base_season:.2f}s**.", "",
         "## Verdicts", "",
         "| Candidate | Change | Params | Overall RMSE | Δ vs base | Season RMSE | Δ season | Verdict |",
         "|---|---|---:|---:|---:|---:|---:|---|"]
    order = ["baseline_global", "mi1_weights", "mi2_affine", "mi2_affine_s50",
             "mi2_affine_rwy", "mi3_weights_affine"]
    for name in order:
        r = results[name]
        L.append(f"| `{name}` | {r['desc']} | "
                 f"{r['n_params'] if r['n_params'] is not None else 'n/a'} | "
                 f"{r['overall']:.2f} | {r['delta_overall']:+.3f} | "
                 f"{r['season']:.2f} | {r['delta_season']:+.3f} | **{r['verdict']}** |")
    L.append("")
    L.append("_Acceptance rule (approved): post-processing kept if overall improves and "
             "season-matched does not degrade (> −0.01s); feature/model changes need ≥0.1% "
             "overall gain. LOT-out is the honest estimate; in-sample fits are not used for "
             "selection._")
    L.append("")
    L.append("## Per-month RMSE (LOT-out)")
    L.append("")
    header = "| Candidate | " + " | ".join(months) + " |"
    L.append(header)
    L.append("|" + "|".join(["---"] * (len(months) + 1)) + "|")
    for name in order:
        row = [f"`{name}`"] + [f"{per_month[name][m]:.1f}" for m in months]
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    L.append("## Full-12-month fits written for accepted candidates")
    L.append("")
    if out["accepted"]:
        for k in out["accepted"]:
            L.append(f"- `{k}`: {results[k]['desc']} (Δ overall {results[k]['delta_overall']:+.3f}s, "
                     f"Δ season {results[k]['delta_season']:+.3f}s)")
    else:
        L.append("_None — no candidate passed the acceptance rule; the recorded blend stands._")
    L.append("")
    L.append("## Interpretation")
    L.append("")
    L.append("- The blend's squared error is dominated by a small number of metadata-missing "
             "(no operator ID) special flights at LIRF (see `error_analysis.md` §6).")
    L.append("- Per-airport affine recalibration tests whether the model systematically "
             "compresses the heavy upper tail; LOT-out decides if any such expansion generalises.")
    L.append("- If no candidate passes, the blend is already RMSE-calibrated and the tail is "
             "treated as irreducible with the ranking-available features.")
    L.append("")
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(L))
    log.info("Wrote report: %s", REPORT_PATH)


if __name__ == "__main__":
    main()
