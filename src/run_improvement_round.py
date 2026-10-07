#!/usr/bin/env python3
"""Task 4.2 improvement round - post-processing calibration battery (LOT-out).

The user asked for one more model/feature/post-processing round before the final
submission. This script is the cheapest, highest-signal first pass: a battery of
*sering-time-only* recalibrations of the existing blend, each scored with
leave-one-month-out (LOT-out) on the 2025 OOF table so the estimate is honest.

Candidates
  baseline_global      identity (reproduction gate)
  v4_affine_s50        per-airport affine, slope shrunk 50% (the accepted v4 rule)
  affine_s25/s75       slope shrinkage variants
  iso_airport          per-airport isotonic regression (monotone)
  iso_global           global isotonic regression
  decile_affine_ap     per (airport x predicted-decile) affine
  qmap_airport         per-airport empirical quantile mapping
  cap_ap_p995          cap predictions at the per-airport training P99.5

Acceptance rule (from the approved Task 4.1 plan): keep a post-processing change
only if overall OOF improves AND season-matched (Jan+Jul) does not degrade.

Usage:
    uv run python src/run_improvement_round.py
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
from run_micro_iterations import _apply_affine, _fit_affine

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("run_improvement_round")

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")
OUT_CONFIG = os.path.join(CONFIG_DIR, "improvement_round.json")
OUT_REPORT = os.path.join(REPORT_DIR, "improvement_round.md")

SEASON_MONTHS = ["2025-01", "2025-07"]
SEASON_TOL = 0.01


def rmse(pred, actual):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(actual)) ** 2)))


# --- isotonic -------------------------------------------------------------
def fit_isotonic(train, key):
    from sklearn.isotonic import IsotonicRegression
    params = {}
    for g, sub in train.groupby(key, observed=True):
        iso = IsotonicRegression(out_of_bounds="clip").fit(sub["pred_blend"].to_numpy(),
                                                           sub["actual_raw"].to_numpy())
        params[g] = (iso.X_thresholds_, iso.y_thresholds_)
    return params


def apply_isotonic(test, key, params):
    out = test["pred_blend"].to_numpy().copy()
    keys = test[key].to_numpy()
    for g, (xs, ys) in params.items():
        m = keys == g
        if m.any():
            out[m] = np.interp(out[m], xs, ys)
    return out


# --- quantile mapping -----------------------------------------------------
def fit_qmap(train, key, grid=201):
    qs = np.linspace(0.0, 1.0, grid)
    params = {}
    for g, sub in train.groupby(key, observed=True):
        p = sub["pred_blend"].to_numpy()
        y = sub["actual_raw"].to_numpy()
        params[g] = (np.quantile(p, qs), np.quantile(y, qs))
    return params


def apply_qmap(test, key, params):
    out = test["pred_blend"].to_numpy().copy()
    keys = test[key].to_numpy()
    for g, (pq, yq) in params.items():
        m = keys == g
        if m.any():
            out[m] = np.interp(out[m], pq, yq)
    return out


# --- decile affine --------------------------------------------------------
def fit_decile_affine(train, key, shrink, nbins=10):
    params = {}
    for g, sub in train.groupby(key, observed=True):
        p = sub["pred_blend"].to_numpy()
        y = sub["actual_raw"].to_numpy()
        if len(sub) < nbins * 20:
            edges = np.array([-np.inf, np.inf])
            var = float(np.var(p))
            slope = float(np.cov(p, y)[0, 1] / var) if var > 0 else 1.0
            slope = 1.0 + shrink * (slope - 1.0)
            params[g] = (edges, np.array([slope]), np.array([float(np.mean(y) - slope * np.mean(p))]))
            continue
        edges = np.quantile(p, np.linspace(0, 1, nbins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        bin_id = np.digitize(p, edges[1:-1])
        slopes, inters = [], []
        for b in range(nbins):
            bm = bin_id == b
            if bm.sum() < 10:
                slopes.append(1.0); inters.append(0.0); continue
            pv, yv = p[bm], y[bm]
            var = float(np.var(pv))
            s = float(np.cov(pv, yv)[0, 1] / var) if var > 0 else 1.0
            s = 1.0 + shrink * (s - 1.0)
            slopes.append(s)
            inters.append(float(np.mean(yv) - s * np.mean(pv)))
        params[g] = (edges, np.array(slopes), np.array(inters))
    return params


def apply_decile_affine(test, key, params):
    out = test["pred_blend"].to_numpy().astype(np.float64).copy()
    keys = test[key].to_numpy()
    for g, (edges, slopes, inters) in params.items():
        m = keys == g
        if not m.any():
            continue
        pv = out[m]
        if len(slopes) == 1:
            out[m] = slopes[0] * pv + inters[0]
        else:
            bin_id = np.digitize(pv, edges[1:-1])
            bin_id = np.clip(bin_id, 0, len(slopes) - 1)
            out[m] = slopes[bin_id] * pv + inters[bin_id]
    return out


# --- tail cap -------------------------------------------------------------
def fit_cap(train, key, q=0.995):
    return {g: float(np.quantile(sub["pred_blend"].to_numpy(), q))
            for g, sub in train.groupby(key, observed=True)}


def apply_cap(test, key, params):
    caps = test[key].map(params)
    cap = caps.fillna(np.inf).to_numpy(dtype=np.float64)
    return np.minimum(test["pred_blend"].to_numpy(), cap)


def main():
    t0 = datetime.now()
    pdf, blend = load_and_join()
    pdf = add_derived(pdf)
    baseline_overall = float(blend["selected"]["rmse"])
    actual = pdf["actual_raw"].to_numpy()
    months = sorted(pdf["month"].unique())
    season_mask = pdf["season_matched"].to_numpy()

    def lot(fit_fn, apply_fn):
        pred = np.full(len(pdf), np.nan)
        ma = pdf["month"].to_numpy()
        for m in months:
            te = ma == m
            params = fit_fn(pdf[~te])
            pred[te] = apply_fn(pdf[te], params)
        return pred

    cands = {
        "baseline_global": (lambda tr: None, lambda te, pr: te["pred_blend"].to_numpy()),
        "v4_affine_s50": (lambda tr: _fit_affine(tr, "airport", tr["pred_blend"].to_numpy(), 0.5),
                          lambda te, pr: _apply_affine(te, "airport", te["pred_blend"].to_numpy(), pr)),
        "affine_s25": (lambda tr: _fit_affine(tr, "airport", tr["pred_blend"].to_numpy(), 0.25),
                       lambda te, pr: _apply_affine(te, "airport", te["pred_blend"].to_numpy(), pr)),
        "affine_s75": (lambda tr: _fit_affine(tr, "airport", tr["pred_blend"].to_numpy(), 0.75),
                       lambda te, pr: _apply_affine(te, "airport", te["pred_blend"].to_numpy(), pr)),
        "iso_airport": (lambda tr: fit_isotonic(tr, "airport"),
                        lambda te, pr: apply_isotonic(te, "airport", pr)),
        "iso_global": (lambda tr: fit_isotonic(tr.assign(_g="all"), "_g"),
                       lambda te, pr: apply_isotonic(te.assign(_g="all"), "_g", pr)),
        "decile_affine_ap": (lambda tr: fit_decile_affine(tr, "airport", 0.5),
                             lambda te, pr: apply_decile_affine(te, "airport", pr)),
        "qmap_airport": (lambda tr: fit_qmap(tr, "airport"),
                         lambda te, pr: apply_qmap(te, "airport", pr)),
        "cap_ap_p995": (lambda tr: fit_cap(tr, "airport", 0.995),
                        lambda te, pr: apply_cap(te, "airport", pr)),
    }

    base_season = rmse(pdf.loc[season_mask, "pred_blend"].to_numpy(), actual[season_mask])
    results = {}
    for name, (fit_fn, apply_fn) in cands.items():
        pred = lot(fit_fn, apply_fn)
        overall = rmse(pred, actual)
        season = rmse(pred[season_mask], actual[season_mask])
        results[name] = {
            "overall": overall,
            "delta_overall": baseline_overall - overall,
            "season": season,
            "delta_season": base_season - season,
        }
        log.info("  %-18s overall %.3f (d %+.3f)  season %.3f (d %+.3f)",
                 name, overall, results[name]["delta_overall"], season, results[name]["delta_season"])

    assert abs(results["baseline_global"]["overall"] - baseline_overall) < 0.01, "reproduction gate failed"

    for name, r in results.items():
        if name == "baseline_global":
            r["verdict"] = "reference"
        else:
            r["verdict"] = ("ACCEPT" if (r["delta_overall"] > 0 and r["delta_season"] >= -SEASON_TOL)
                            else "reject")

    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(OUT_CONFIG, "w") as f:
        json.dump({"created": datetime.now().strftime("%Y-%m-%d"),
                   "source": "LOT-out on data/processed/oof_predictions.parquet",
                   "baseline_overall": baseline_overall, "baseline_season": base_season,
                   "candidates": results}, f, indent=2, default=float)

    lines = ["# Improvement Round — post-processing calibration battery (Task 4.2)", "",
             f"Generated: {datetime.now().astimezone():%Y-%m-%d %H:%M %Z} · LOT-out on the 2025 OOF "
             f"({len(months)} folds) · Baseline overall **{baseline_overall:.3f}s**, "
             f"season-matched **{base_season:.3f}s**.", "",
             "| Candidate | Overall RMSE | Δ overall | Season RMSE | Δ season | Verdict |",
             "|---|---:|---:|---:|---:|---|"]
    for name, r in results.items():
        lines.append(f"| `{name}` | {r['overall']:.3f} | {r['delta_overall']:+.3f} | "
                     f"{r['season']:.3f} | {r['delta_season']:+.3f} | **{r['verdict']}** |")
    lines += ["", "_Acceptance rule: overall improves AND season-matched does not degrade._", ""]
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(OUT_REPORT, "w") as f:
        f.write("\n".join(lines))

    accepted = [k for k, v in results.items() if v["verdict"] == "ACCEPT"]
    log.info("Accepted: %s", accepted or "NONE")
    log.info("Done in %.1f min", (datetime.now() - t0).total_seconds() / 60)


if __name__ == "__main__":
    main()
