#!/usr/bin/env python3
"""Blend the top Task 3.3 model configurations and produce submission v3 (Task 3.4).

Phases (each independently runnable):
  oof     12-fold month-blocked CV for every component; writes the wide
          per-MVT_ID OOF table that downstream error analysis consumes.
  weights Simplex-constrained blend-weight search on the OOF predictions only
          (never the leaderboard); writes the blend config + report.
  serve   Retrains the selected components on full 2025, scores the ranking
          set through the D4 serving path, blends, and writes the submission.
  all     oof -> weights -> serve.

Components (D4 features, E-winsorize-errors target, 12-fold CV, seed 42):
  tuned_lgbm        global LightGBM, src/configs/best.json params (CV 337.54s)
  per_airport_lgbm  10 per-airport LightGBM, default params  (CV 335.98s)
  xgb               global XGBoost, default params           (CV 355.00s)
  cat               global CatBoost, default params          (CV 362.24s)

Usage:
    uv run python src/run_blend.py --phase oof [--smoke]
    uv run python src/run_blend.py --phase weights
    uv run python src/run_blend.py --phase serve --version 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import polars as pl

from pdc26 import SEED
from pdc26.airports import load_coords
from pdc26.data import (
    load_ranking,
    load_ranking_movements,
    load_training,
    load_training_movements,
)
from pdc26.features import (
    CAT_FEATURES,
    apply_operator_target_encoding,
    fit_operator_target_encoding,
)
from pdc26.tuning import (
    build_d4_features,
    encode_operator_within_fold,
    family_default_params,
    fit_d4_vocabs,
    load_metar,
    month_blocked_cv_model,
    winsorize_errors,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_blend")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_RAW = os.path.join(_PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models", "blend")
SUBMISSIONS_DIR = os.path.join(_PROJECT_ROOT, "submissions")
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")
LOGS_DIR = os.path.join(_PROJECT_ROOT, "logs")

OOF_PARQUET = os.path.join(DATA_PROCESSED, "oof_predictions.parquet")
OOF_META = os.path.join(CONFIG_DIR, "blend_oof_meta.json")
BLEND_CONFIG = os.path.join(CONFIG_DIR, "blend.json")
POSTPROC_CONFIG = os.path.join(CONFIG_DIR, "blend_postproc.json")
BLEND_REPORT = os.path.join(REPORT_DIR, "blend_results.md")
QUALITY_REPORT = os.path.join(DATA_PROCESSED, "blend_ranking_predictions.parquet")

TARGET_TREATMENT = "E-winsorize-errors"
N_ESTIMATORS = 500
EARLY_STOPPING = 50
SMOKE_ROUNDS = 100
SMOKE_MONTHS = ["2025-01", "2025-02"]
SMOKE_AIRPORTS = 2

# Prior Task 3.3 full-12-fold raw RMSE (s), used as a reproduction gate.
EXPECTED_OOF_RMSE = {
    "tuned_lgbm": 337.54,
    "per_airport_lgbm": 335.98,
    "xgb": 355.00,
    "cat": 362.24,
}
GATE_TOLERANCE = 0.01  # 1% relative


@dataclass
class Component:
    """A blend component: one config, either global or per-airport."""

    name: str
    family: str
    params: dict
    mode: str  # "global" | "per_airport"
    n_estimators: int = N_ESTIMATORS
    early_stopping: int = EARLY_STOPPING

    def describe(self) -> dict:
        return {
            "family": self.family,
            "mode": self.mode,
            "feature_set": "D4",
            "target_treatment": TARGET_TREATMENT,
            "seed": SEED,
            "n_estimators": self.n_estimators,
            "early_stopping_rounds": self.early_stopping,
            "params": self.params,
        }


def build_components() -> dict[str, Component]:
    """Assemble the four component configs from best.json + family defaults."""
    with open(os.path.join(CONFIG_DIR, "best.json")) as f:
        best = json.load(f)
    return {
        "tuned_lgbm": Component(
            name="tuned_lgbm", family="lgbm", params=best["params"], mode="global",
            n_estimators=int(best.get("n_estimators", N_ESTIMATORS)),
            early_stopping=int(best.get("early_stopping_rounds", EARLY_STOPPING)),
        ),
        "per_airport_lgbm": Component(
            name="per_airport_lgbm", family="lgbm",
            params=family_default_params("lgbm", SEED), mode="per_airport",
        ),
        "xgb": Component(
            name="xgb", family="xgb",
            params=family_default_params("xgb", SEED), mode="global",
        ),
        "cat": Component(
            name="cat", family="cat",
            params=family_default_params("cat", SEED), mode="global",
        ),
    }


class BlendData:
    """All pre-loaded pieces needed to build the D4 feature matrix."""

    def __init__(self):
        log.info("Loading training DEP data...")
        self.dep = load_training()
        log.info("Loading all-phase movements...")
        self.movements = load_training_movements().to_pandas()
        log.info("Loading METAR + coordinates...")
        self.metar = load_metar()
        self.coords = load_coords()

        log.info("Fitting vocabularies...")
        self.vocab, self.domain_vocab = fit_d4_vocabs(self.dep)

        self.pdf_full = self.dep.to_pandas()
        self.raw_target = self.dep["TAXITIME_SEC_mvt"].to_pandas().astype("float64")
        self.airports = self.dep["ADEP_mvt"].to_pandas().astype("string").fillna("__NA__")
        self.months = get_month_labels_safe(self.dep)
        self.mvt_ids = self.dep["MVT_ID_mvt"].to_pandas()

        for s in (self.raw_target, self.airports, self.months, self.mvt_ids):
            s.index = range(len(s))

        self.y_winsor = winsorize_errors(self.raw_target, self.airports)
        self.fold_masks = {m: (self.months.values == m) for m in sorted(self.months.unique())}

        log.info("Computing within-fold operator target encoding...")
        self.operator_enc = encode_operator_within_fold(
            self.pdf_full["AIRCRAFT_OPERATOR_flt"], self.fold_masks, self.y_winsor
        )

    def build_X(self) -> pd.DataFrame:
        X = build_d4_features(
            self.dep, self.movements, self.vocab, self.domain_vocab,
            self.metar, self.coords, operator_enc=self.operator_enc,
        )
        return cast_features(X)

    def row_subset(self, keep: np.ndarray) -> SimpleNamespace:
        """Lightweight per-row view for smoke runs (OOF path needs no polars)."""
        return SimpleNamespace(
            y_winsor=self.y_winsor[keep].reset_index(drop=True),
            months=self.months[keep].reset_index(drop=True),
            mvt_ids=self.mvt_ids[keep].reset_index(drop=True),
            raw_target=self.raw_target[keep].reset_index(drop=True),
            airports=self.airports[keep].reset_index(drop=True),
        )


def get_month_labels_safe(df: pl.DataFrame) -> pd.Series:
    """Month labels from MVT_TIME_UTC_mvt (thin wrapper; keeps import local)."""
    from pdc26.features import get_month_labels

    labels = get_month_labels(df)
    labels.index = range(len(labels))
    return labels


def cast_features(X: pd.DataFrame) -> pd.DataFrame:
    """Cast numeric feature columns to float32 (matches Task 3.3 training)."""
    for col in X.select_dtypes(include=["float64", "int64"]).columns:
        X[col] = X[col].astype("float32")
    return X


# ---------------------------------------------------------------------------
# Phase oof
# ---------------------------------------------------------------------------

def _effective_params(comp: Component, smoke: bool) -> dict:
    params = {k: v for k, v in comp.params.items()}
    if smoke and comp.family == "cat":
        params["iterations"] = SMOKE_ROUNDS
    return params


def _effective_rounds(comp: Component, smoke: bool) -> int:
    return SMOKE_ROUNDS if smoke else comp.n_estimators


def _oof_component(bundle, X, comp, smoke):
    """Run one component's CV and return (oof_df, meta)."""
    params = _effective_params(comp, smoke)
    rounds = _effective_rounds(comp, smoke)

    if comp.mode == "global":
        res = month_blocked_cv_model(
            X=X, y=bundle.y_winsor, months=bundle.months, mvt_ids=bundle.mvt_ids,
            cat_features=CAT_FEATURES, family=comp.family, params=params,
            n_estimators=rounds, early_stopping_rounds=comp.early_stopping,
            y_raw=bundle.raw_target,
        )
        meta = {
            "oof_rmse": float(res.overall_rmse_raw),
            "best_iterations": [int(v) for v in res.per_month["best_iteration"].tolist()],
        }
        return res.oof_predictions, meta

    frames, per_apt = [], {}
    airports = sorted(bundle.airports.unique())
    if smoke:
        airports = airports[:SMOKE_AIRPORTS]
    for apt in airports:
        idx = np.where(bundle.airports.values == apt)[0]
        log.info("  per-airport %s: %s rows", apt, f"{len(idx):,}")
        res = month_blocked_cv_model(
            X=X.iloc[idx].reset_index(drop=True),
            y=bundle.y_winsor.iloc[idx].reset_index(drop=True),
            months=bundle.months.iloc[idx].reset_index(drop=True),
            mvt_ids=bundle.mvt_ids.iloc[idx].reset_index(drop=True),
            cat_features=CAT_FEATURES, family=comp.family, params=params,
            n_estimators=rounds, early_stopping_rounds=comp.early_stopping,
            y_raw=bundle.raw_target.iloc[idx].reset_index(drop=True),
        )
        frames.append(res.oof_predictions)
        per_apt[apt] = [int(v) for v in res.per_month["best_iteration"].tolist()]

    oof = pd.concat(frames, ignore_index=True)
    rmse = float(np.sqrt(np.mean((oof["prediction"] - oof["actual_raw"]) ** 2)))
    meta = {"oof_rmse": rmse, "per_airport_best_iterations": per_apt}
    return oof, meta


def merge_oof(view, pred_frames: dict) -> pd.DataFrame:
    """Merge component OOF frames into one wide per-MVT_ID table."""
    base = pd.DataFrame({
        "MVT_ID_mvt": view.mvt_ids.values,
        "fold_month": view.months.astype(str).values,
        "actual_raw": view.raw_target.values.astype("float64"),
    })
    for name, oof in pred_frames.items():
        s = oof.set_index("MVT_ID_mvt")["prediction"]
        base[f"pred_{name}"] = base["MVT_ID_mvt"].map(s).astype("float64")
    return base


def oof_paths(smoke: bool) -> tuple[str, str]:
    if smoke:
        return (
            os.path.join(DATA_PROCESSED, "oof_predictions_smoke.parquet"),
            os.path.join(CONFIG_DIR, "blend_oof_meta_smoke.json"),
        )
    return OOF_PARQUET, OOF_META


def blend_paths(smoke: bool) -> tuple[str, str]:
    if smoke:
        return (
            os.path.join(CONFIG_DIR, "blend_smoke.json"),
            os.path.join(REPORT_DIR, "blend_results_smoke.md"),
        )
    return BLEND_CONFIG, BLEND_REPORT


def run_oof(bundle, comps, smoke: bool, ignore_gate: bool) -> bool:
    t0 = time.time()
    X = bundle.build_X()
    log.info("D4 feature matrix: %s rows x %d cols", f"{X.shape[0]:,}", X.shape[1])

    view = bundle
    if smoke:
        keep = bundle.months.isin(SMOKE_MONTHS).values
        X = X.loc[keep].reset_index(drop=True)
        view = bundle.row_subset(keep)
        log.info("SMOKE: restricted to %s (%s rows), %s airports, %d rounds",
                 SMOKE_MONTHS, f"{len(X):,}", SMOKE_AIRPORTS, SMOKE_ROUNDS)

    meta: dict = {"smoke": smoke, "target_treatment": TARGET_TREATMENT, "components": {}}
    pred_frames: dict = {}
    for name, comp in comps.items():
        log.info("=" * 60)
        log.info("OOF component: %s (family=%s, mode=%s)", name, comp.family, comp.mode)
        oof, cmeta = _oof_component(view, X, comp, smoke)
        cmeta["config"] = comp.describe()
        meta["components"][name] = cmeta
        pred_frames[name] = oof
        log.info("  %s OOF raw RMSE = %.2f", name, cmeta["oof_rmse"])

    wide = merge_oof(view, pred_frames)

    gate_ok = True
    if not smoke:
        pred_cols = [c for c in wide.columns if c.startswith("pred_")]
        assert len(wide) == len(view.mvt_ids), "OOF row count mismatch"
        assert wide[pred_cols].notna().all().all(), "OOF merge left null predictions"
        for name, cmeta in meta["components"].items():
            exp = EXPECTED_OOF_RMSE.get(name)
            if exp is None:
                continue
            dev = abs(cmeta["oof_rmse"] - exp) / exp
            status = "OK" if dev <= GATE_TOLERANCE else "FAIL"
            log.info("  gate %-17s got %.2f  expected %.2f  dev %.3f%%  -> %s",
                     name, cmeta["oof_rmse"], exp, dev * 100, status)
            if dev > GATE_TOLERANCE:
                gate_ok = False
        meta["gate_ok"] = gate_ok

    path, meta_path = oof_paths(smoke)
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    wide.to_parquet(path, index=False)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Wrote OOF table: %s (%s rows x %d cols)", path, f"{len(wide):,}", wide.shape[1])
    log.info("Wrote OOF meta:  %s", meta_path)

    elapsed = time.time() - t0
    log.info("Phase oof done in %.0fs (%.1f min)", elapsed, elapsed / 60)
    if not gate_ok and not ignore_gate:
        raise SystemExit("ERROR: OOF reproduction gate failed (>1% deviation). "
                         "Diff the implementation or re-run with --ignore-gate.")
    return gate_ok


# ---------------------------------------------------------------------------
# Phase weights
# ---------------------------------------------------------------------------

def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _simplex_weights(P: np.ndarray, y: np.ndarray, seed: int = SEED,
                     n_random: int = 8) -> tuple[np.ndarray, float]:
    """Minimize RMSE over the probability simplex (weights >= 0, sum = 1)."""
    from scipy.optimize import minimize

    k = P.shape[1]

    def obj(w: np.ndarray) -> float:
        return float(np.sqrt(np.mean((P @ w - y) ** 2)))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bnds = [(0.0, 1.0)] * k
    starts = [np.full(k, 1.0 / k)]
    for i in range(k):
        e = np.zeros(k)
        e[i] = 1.0
        starts.append(e)
    rng = np.random.default_rng(seed)
    for _ in range(n_random):
        starts.append(rng.dirichlet(np.ones(k)))

    best_w, best_r = None, np.inf
    for w0 in starts:
        res = minimize(obj, w0, method="SLSQP", bounds=bnds, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-14})
        w = np.clip(res.x, 0.0, None)
        s = float(w.sum())
        if s <= 0:
            continue
        w = w / s
        r = _rmse(y, P @ w)
        if r < best_r:
            best_r, best_w = r, w
    if best_w is None:
        raise RuntimeError("Simplex search failed to find any feasible weights")
    return best_w, best_r


def run_weights(smoke: bool) -> dict:
    path, meta_path = oof_paths(smoke)
    oof = pd.read_parquet(path)
    with open(meta_path) as f:
        meta = json.load(f)

    names = [c[len("pred_"):] for c in oof.columns if c.startswith("pred_")]
    pred_cols = [f"pred_{n}" for n in names]
    complete = oof[pred_cols].notna().all(axis=1).to_numpy()
    n_incomplete = int((~complete).sum())
    if n_incomplete:
        if not smoke:
            raise ValueError(f"{n_incomplete} OOF rows have missing component predictions")
        log.warning("SMOKE: dropping %d incomplete-case rows", n_incomplete)
        oof = oof.loc[complete].reset_index(drop=True)
    y = oof["actual_raw"].to_numpy("float64")
    P_all = oof[pred_cols].to_numpy("float64")
    log.info("OOF matrix: %s rows x %d components (%s)", f"{len(y):,}", len(names), names)

    singles = {n: _rmse(y, P_all[:, i]) for i, n in enumerate(names)}
    best_single = min(singles, key=singles.get)
    for n in names:
        log.info("  single %-17s OOF raw RMSE = %.4f", n, singles[n])
    log.info("  best single: %s (%.4f)", best_single, singles[best_single])

    resid = P_all - y[:, None]
    corr = np.corrcoef(resid.T)
    corr_map = {names[i]: {names[j]: float(corr[i, j]) for j in range(len(names))}
                for i in range(len(names))}

    avg_w = np.full(len(names), 1.0 / len(names))
    avg_rmse = _rmse(y, P_all @ avg_w)
    log.info("  simple average (all %d): %.4f", len(names), avg_rmse)

    subset_results = []
    best: dict = {"weights": None, "rmse": np.inf, "subset": None}
    for r in range(1, len(names) + 1):
        for combo in combinations(range(len(names)), r):
            P = P_all[:, list(combo)]
            w, rr = _simplex_weights(P, y)
            entry = {
                "subset": [names[i] for i in combo],
                "weights": {names[combo[j]]: float(w[j]) for j in range(r)},
                "rmse": float(rr),
            }
            subset_results.append(entry)
            if rr < best["rmse"]:
                best = {"weights": entry["weights"], "rmse": float(rr),
                        "subset": entry["subset"]}
    subset_results.sort(key=lambda d: d["rmse"])

    active = {n: w for n, w in best["weights"].items() if w > 1e-6}
    beats = best["rmse"] < singles[best_single] - 1e-9
    log.info("=" * 60)
    log.info("Best subset: %s -> OOF raw RMSE = %.4f", best["subset"], best["rmse"])
    log.info("Active weights: %s", {k: round(v, 4) for k, v in active.items()})
    log.info("Blend beats best single (%s, %.4f): %s (gain %.4f s)",
             best_single, singles[best_single], beats, singles[best_single] - best["rmse"])

    # Full-retrain round counts from CV best_iteration medians.
    rounds: dict = {}
    for name, cmeta in meta["components"].items():
        if "best_iterations" in cmeta:
            rounds[name] = int(round(float(np.median(cmeta["best_iterations"]))))
        elif "per_airport_best_iterations" in cmeta:
            rounds[name] = {apt: int(round(float(np.median(v))))
                            for apt, v in cmeta["per_airport_best_iterations"].items()}

    payload = {
        "created": time.strftime("%Y-%m-%d"),
        "smoke": smoke,
        "target_treatment": TARGET_TREATMENT,
        "component_configs": {n: meta["components"][n].get("config", {}) for n in names},
        "oof_rmse": singles,
        "residual_correlation": corr_map,
        "simple_average": {"weights": {n: float(w) for n, w in zip(names, avg_w)},
                           "rmse": float(avg_rmse)},
        "subset_search": subset_results,
        "selected": {
            "weights": {n: float(w) for n, w in best["weights"].items()},
            "active_weights": {n: float(w) for n, w in active.items()},
            "n_active": len(active),
            "rmse": float(best["rmse"]),
            "subset": best["subset"],
            "beats_best_single": bool(beats),
            "best_single": {"name": best_single, "rmse": float(singles[best_single])},
        },
        "full_retrain_rounds": rounds,
    }
    cfg_path, rep_path = blend_paths(smoke)
    with open(cfg_path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Wrote blend config: %s", cfg_path)

    _write_blend_report(payload, meta, rep_path)
    return payload


def _write_blend_report(payload: dict, meta: dict, report_path: str) -> str:
    os.makedirs(REPORT_DIR, exist_ok=True)
    sel = payload["selected"]
    cfgs = payload["component_configs"]
    lines = [
        "# Blend Results — Task 3.4",
        "",
        "Task: `11-blend_submission` · Team: `thoughtful-glove`",
        f"Generated: {payload['created']} · Seed: {SEED}",
        "",
        "---",
        "",
        "## Configuration",
        "",
        "- **Feature set:** D4 (baseline + METAR + congestion + domain, 62 features)",
        f"- **Target:** {TARGET_TREATMENT}",
        "- **CV:** 12-fold month-blocked (leave-one-month-out, Jan–Dec 2025)",
        "- **Metric:** raw unclipped RMSE (competition metric)",
        "- **Weights fit on OOF predictions only — never leaderboard feedback.**",
        "",
        "## Components (single-model OOF raw RMSE)",
        "",
        "| Component | Family | Mode | OOF RMSE (s) | Median best_iter |",
        "|-----------|--------|------|-------------:|-----------------:|",
    ]
    for n in payload["oof_rmse"]:
        r = payload["full_retrain_rounds"].get(n)
        r_str = (f"{r}" if isinstance(r, int)
                 else ", ".join(f"{k}:{v}" for k, v in sorted(r.items())) if isinstance(r, dict)
                 else "—")
        lines.append(f"| {n} | {cfgs.get(n, {}).get('family', '?')} | "
                     f"{cfgs.get(n, {}).get('mode', '?')} | "
                     f"{payload['oof_rmse'][n]:.2f} | {r_str} |")

    lines += [
        "",
        "## Residual Correlation (OOF errors)",
        "",
        "| | " + " | ".join(payload["oof_rmse"].keys()) + " |",
        "|---|" + "|".join(["---:"] * len(payload["oof_rmse"])) + "|",
    ]
    for a, row in payload["residual_correlation"].items():
        lines.append(f"| {a} | " + " | ".join(f"{row[b]:.3f}" for b in row) + " |")

    lines += [
        "",
        "## Blend Search (simplex-constrained, all subsets)",
        "",
        "| Rank | Subset | OOF RMSE (s) | Weights |",
        "|-----:|--------|-------------:|---------|",
    ]
    for i, e in enumerate(payload["subset_search"][:10], 1):
        wstr = ", ".join(f"{k}={v:.3f}" for k, v in e["weights"].items())
        lines.append(f"| {i} | {' + '.join(e['subset'])} | {e['rmse']:.4f} | {wstr} |")

    lines += [
        "",
        f"- **Simple average (all {len(payload['oof_rmse'])}):** "
        f"{payload['simple_average']['rmse']:.4f} s",
        f"- **Best single component:** {sel['best_single']['name']} "
        f"({sel['best_single']['rmse']:.4f} s)",
        f"- **Selected blend:** {' + '.join(sel['subset'])} "
        f"({sel['rmse']:.4f} s)",
        f"- **Blend beats best single:** {sel['beats_best_single']} "
        f"(gain {sel['best_single']['rmse'] - sel['rmse']:.4f} s)",
        "",
        "## Decision",
        "",
    ]
    if sel["n_active"] >= 2 and sel["beats_best_single"]:
        lines.append(f"Weighted blend of **{sel['n_active']} components** selected "
                     f"(OOF {sel['rmse']:.2f} s ≤ best single {sel['best_single']['rmse']:.2f} s).")
        lines.append("")
        lines.append("Weights: " + ", ".join(f"`{k}` = {v:.4f}"
                                             for k, v in sel["active_weights"].items()))
    else:
        lines.append(f"Blend does not meaningfully beat the best single component; "
                     f"**keep single model `{sel['best_single']['name']}`** "
                     f"({sel['best_single']['rmse']:.2f} s) and document.")
    lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    log.info("Wrote blend report: %s", report_path)
    return report_path


# ---------------------------------------------------------------------------
# Phase serve
# ---------------------------------------------------------------------------

def _train_lgbm_full(X: pd.DataFrame, y: pd.Series, params: dict, rounds: int):
    import lightgbm as lgb

    p = {k: v for k, v in params.items() if v is not None}
    dtrain = lgb.Dataset(X, label=y.values, categorical_feature=CAT_FEATURES,
                         free_raw_data=False)
    return lgb.train(p, dtrain, num_boost_round=int(rounds))


def _train_xgb_full(X: pd.DataFrame, y: pd.Series, params: dict, rounds: int):
    import xgboost as xgb

    p = dict(params)
    for k in ("metric", "objective", "force_row_wise", "num_threads", "nthread"):
        p.pop(k, None)
    p["verbosity"] = 0
    dtrain = xgb.DMatrix(X, label=y.values)
    return xgb.train(p, dtrain, num_boost_round=int(rounds))


def _train_cat_full(X: pd.DataFrame, y: pd.Series, params: dict, rounds: int):
    from catboost import CatBoostRegressor, Pool

    ctb: dict = {}
    for key in ("learning_rate", "depth", "l2_leaf_reg", "iterations",
                "random_seed", "loss_function", "eval_metric", "verbose"):
        if key in params and params[key] is not None:
            ctb[key] = params[key]
    ctb["iterations"] = int(rounds)
    ctb.setdefault("random_seed", SEED)
    ctb.setdefault("loss_function", "RMSE")
    ctb.setdefault("verbose", False)
    model = CatBoostRegressor(**ctb)
    model.fit(Pool(X, y.values))
    return model


def _train_by_family(family: str, X: pd.DataFrame, y: pd.Series,
                     params: dict, rounds: int):
    if family == "lgbm":
        return _train_lgbm_full(X, y, params, rounds)
    if family == "xgb":
        return _train_xgb_full(X, y, params, rounds)
    if family == "cat":
        return _train_cat_full(X, y, params, rounds)
    raise ValueError(f"Unknown family: {family}")


def _save_model(model, family: str, stem: str) -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    if family == "lgbm":
        path = os.path.join(MODELS_DIR, f"{stem}.txt")
    elif family == "xgb":
        path = os.path.join(MODELS_DIR, f"{stem}.json")
    else:
        path = os.path.join(MODELS_DIR, f"{stem}.cbm")
    model.save_model(path)
    return path


def _per_airport_rounds(spec, default: int) -> int:
    if isinstance(spec, dict) and spec:
        return int(round(float(np.median(list(spec.values())))))
    if isinstance(spec, int):
        return spec
    return default


def train_components_full(bundle, X: pd.DataFrame, comps: dict, rounds_spec: dict,
                          active_names: list[str], smoke: bool) -> dict:
    """Retrain every active component on the full training set."""
    models: dict = {}
    for name in active_names:
        comp = comps[name]
        log.info("=" * 60)
        log.info("Full retrain: %s (family=%s, mode=%s)", name, comp.family, comp.mode)
        t0 = time.time()
        if comp.mode == "global":
            rounds = SMOKE_ROUNDS if smoke else int(rounds_spec.get(name, comp.n_estimators))
            model = _train_by_family(comp.family, X, bundle.y_winsor, comp.params, rounds)
            _save_model(model, comp.family, f"{name}_full")
            models[name] = model
            log.info("  trained %d rounds -> saved", rounds)
        else:
            per = rounds_spec.get(name, {})
            default_r = _per_airport_rounds(per, comp.n_estimators)
            models[name] = {}
            for apt in sorted(bundle.airports.unique()):
                idx = np.where(bundle.airports.values == apt)[0]
                rounds = SMOKE_ROUNDS if smoke else int(per.get(apt, default_r))
                m = _train_by_family(
                    comp.family,
                    X.iloc[idx].reset_index(drop=True),
                    bundle.y_winsor.iloc[idx].reset_index(drop=True),
                    comp.params, rounds,
                )
                _save_model(m, comp.family, f"{name}_{apt}_full")
                models[name][apt] = m
            log.info("  trained %d per-airport models", len(models[name]))
        log.info("  %s retrain done in %.0fs", name, time.time() - t0)
    return models


def build_ranking_features(bundle):
    """Build the D4 feature matrix for the ranking set using training-fitted pieces."""
    log.info("Loading ranking data + movements...")
    rank_df = load_ranking()
    rank_movements = load_ranking_movements().to_pandas()
    rank_pdf = rank_df.to_pandas()

    log.info("Fitting operator encoding on full training target...")
    enc = fit_operator_target_encoding(
        bundle.pdf_full["AIRCRAFT_OPERATOR_flt"], bundle.y_winsor)
    fallback = float(bundle.y_winsor.mean())
    op_rank = apply_operator_target_encoding(
        rank_pdf["AIRCRAFT_OPERATOR_flt"], enc, fallback=fallback)

    log.info("Building D4 ranking features...")
    X_rank = build_d4_features(
        rank_df, rank_movements, bundle.vocab, bundle.domain_vocab,
        bundle.metar, bundle.coords, operator_enc=op_rank,
    )
    X_rank = cast_features(X_rank)
    X_rank.index = range(len(X_rank))
    log.info("Ranking feature matrix: %s rows x %d cols", f"{len(X_rank):,}", X_rank.shape[1])
    return rank_df, rank_pdf, X_rank


def _predict(model, family: str, X: pd.DataFrame) -> np.ndarray:
    if family == "xgb":
        import xgboost as xgb

        return model.predict(xgb.DMatrix(X))
    return model.predict(X)


def score_ranking_components(models: dict, comps: dict, X_rank: pd.DataFrame,
                             rank_pdf: pd.DataFrame, active_names: list[str]) -> dict:
    """Score each active component on the ranking rows."""
    rank_airport = rank_pdf["ADEP_mvt"].astype("string").fillna("__NA__").values
    preds: dict = {}
    for name in active_names:
        comp = comps[name]
        log.info("Scoring component %s on ranking rows...", name)
        if comp.mode == "global":
            preds[name] = _predict(models[name], comp.family, X_rank)
        else:
            out = np.full(len(X_rank), np.nan, dtype="float64")
            for apt, m in models[name].items():
                mask = (rank_airport == apt)
                if mask.any():
                    out[mask] = _predict(m, comp.family, X_rank.loc[mask].reset_index(drop=True))
            n_missing = int(np.isnan(out).sum())
            if n_missing > 0:
                raise ValueError(f"{name}: {n_missing} ranking rows have no per-airport model")
            preds[name] = out
        log.info("  %s: mean=%.1f median=%.1f min=%.1f max=%.1f",
                 name, np.mean(preds[name]), np.median(preds[name]),
                 np.min(preds[name]), np.max(preds[name]))
    return preds


def write_submission(preds: dict, weights: dict, version: int, rank_df,
                     smoke: bool = False) -> str:
    """Blend component predictions and write the validated submission parquet."""
    names = list(weights.keys())
    blended = np.zeros(len(preds[names[0]]), dtype="float64")
    for n in names:
        blended += float(weights[n]) * preds[n]
    log.info("Blended %d components: mean=%.1f median=%.1f min=%.1f max=%.1f",
             len(names), blended.mean(), np.median(blended), blended.min(), blended.max())

    submitting = pl.read_parquet(os.path.join(DATA_RAW, "submitting.parquet"))
    sub_ids = submitting["MVT_ID_mvt"].to_pandas()
    rank_ids = rank_df["MVT_ID_mvt"].to_pandas()
    pred_map = pd.Series(blended, index=rank_ids.values)
    vals = sub_ids.map(pred_map).to_numpy("float64")

    assert len(vals) == submitting.height, \
        f"row count mismatch: {len(vals)} vs {submitting.height}"
    assert not np.isnan(vals).any(), "null predictions after join"
    assert np.isfinite(vals).all(), "non-finite predictions"
    vals = np.maximum(vals, 1.0)

    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    out = pd.DataFrame({"MVT_ID_mvt": sub_ids.values, "TAXITIME_SEC_mvt": vals})
    if smoke:
        path = os.path.join(DATA_PROCESSED, f"submission_smoke_v{version}.parquet")
    else:
        path = os.path.join(SUBMISSIONS_DIR, f"thoughtful-glove_v{version}.parquet")
    out.to_parquet(path, index=False)
    log.info("Saved %s (%s rows)", path, f"{len(out):,}")
    log.info("Prediction stats: mean=%.1f median=%.1f min=%.1f max=%.1f std=%.1f",
             vals.mean(), np.median(vals), vals.min(), vals.max(), vals.std())
    log.info("Upload: mc cp %s opensky/prc-2026-thoughtful-glove/", path)

    side = pd.DataFrame({"MVT_ID_mvt": rank_ids.values})
    for n in names:
        side[f"pred_{n}"] = preds[n]
    side["blended"] = blended
    side_path = (os.path.join(DATA_PROCESSED, "blend_ranking_predictions_smoke.parquet")
                 if smoke else QUALITY_REPORT)
    side.to_parquet(side_path, index=False)
    log.info("Saved per-component ranking predictions: %s", side_path)
    return path


def run_reblend(version: int, candidate: str) -> str:
    """Apply an accepted post-processing transform to existing ranking predictions.

    Reads the per-component ranking predictions written by `run_serve`
    (blend_ranking_predictions.parquet), applies the accepted affine recalibration
    from blend_postproc.json, and writes the validated submission parquet. No
    retraining and no feature rebuild: the component predictions already exist.
    """
    import hashlib

    with open(POSTPROC_CONFIG) as f:
        pp = json.load(f)
    if candidate not in pp.get("full_fit", {}):
        raise SystemExit(f"candidate '{candidate}' has no accepted full fit in {POSTPROC_CONFIG}")
    spec = pp["full_fit"][candidate]
    if spec.get("type") != "affine" or spec.get("base") != "global_blend":
        raise SystemExit(f"reblend supports affine-on-global-blend only; got type={spec.get('type')}")
    log.info("Reblend v%d using %s: %s", version, candidate,
             pp["candidates"][candidate]["desc"])

    comp = pd.read_parquet(QUALITY_REPORT)
    rank_df = load_ranking()
    rank_ids = rank_df["MVT_ID_mvt"].to_pandas()
    rank_apt = rank_df["ADEP_mvt"].to_pandas().astype("string").fillna("__NA__")
    key = pd.DataFrame({"MVT_ID_mvt": rank_ids.values, "airport": rank_apt.values})
    comp = comp.merge(key, on="MVT_ID_mvt", how="left")
    assert comp["airport"].notna().all(), "some ranking rows lack an airport"
    assert comp["MVT_ID_mvt"].is_unique, "ranking predictions have duplicate MVT_IDs"

    params = {k: tuple(v) for k, v in spec["airport_affine"].items()}
    slope_map = {k: v[0] for k, v in params.items()}
    inter_map = {k: v[1] for k, v in params.items()}
    slopes = comp["airport"].map(slope_map).fillna(1.0).to_numpy("float64")
    inters = comp["airport"].map(inter_map).fillna(0.0).to_numpy("float64")
    blended = slopes * comp["blended"].to_numpy("float64") + inters
    log.info("Post-processed blend: mean=%.1f median=%.1f min=%.1f max=%.1f",
             blended.mean(), np.median(blended), blended.min(), blended.max())

    submitting = pl.read_parquet(os.path.join(DATA_RAW, "submitting.parquet"))
    sub_ids = submitting["MVT_ID_mvt"].to_pandas()
    pred_map = pd.Series(blended, index=comp["MVT_ID_mvt"].values)
    vals = sub_ids.map(pred_map).to_numpy("float64")
    assert len(vals) == submitting.height, \
        f"row count mismatch: {len(vals)} vs {submitting.height}"
    assert not pd.isna(vals).any(), "null predictions after join"
    assert np.isfinite(vals).all(), "non-finite predictions"
    vals = np.maximum(vals, 1.0)

    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    out = pd.DataFrame({"MVT_ID_mvt": sub_ids.values, "TAXITIME_SEC_mvt": vals})
    path = os.path.join(SUBMISSIONS_DIR, f"thoughtful-glove_v{version}.parquet")
    out.to_parquet(path, index=False)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    log.info("Saved %s (%s rows)", path, f"{len(out):,}")
    log.info("Prediction stats: mean=%.1f median=%.1f min=%.1f max=%.1f std=%.1f",
             vals.mean(), np.median(vals), vals.min(), vals.max(), vals.std())
    log.info("SHA256 %s", sha)
    log.info("Upload: mc cp %s opensky/prc-2026-thoughtful-glove/", path)
    return path


def run_serve(version: int, smoke: bool, bundle: BlendData | None = None) -> str:
    comps = build_components()
    cfg_path, _ = blend_paths(smoke)
    with open(cfg_path) as f:
        blend = json.load(f)
    sel = blend["selected"]
    weights = sel["active_weights"]
    active_names = list(weights.keys())
    rounds_spec = blend["full_retrain_rounds"]
    log.info("Serving with weights: %s", {k: round(v, 4) for k, v in weights.items()})

    t0 = time.time()
    bundle = bundle if bundle is not None else BlendData()
    X = bundle.build_X()

    train_bundle, X_train = bundle, X
    if smoke:
        keep = bundle.months.isin(SMOKE_MONTHS).values
        X_train = X.loc[keep].reset_index(drop=True)
        train_bundle = bundle.row_subset(keep)
        log.info("SMOKE serve: training on %s rows, %d rounds", f"{len(X_train):,}", SMOKE_ROUNDS)

    models = train_components_full(train_bundle, X_train, comps, rounds_spec,
                                   active_names, smoke)
    rank_df, rank_pdf, X_rank = build_ranking_features(bundle)
    preds = score_ranking_components(models, comps, X_rank, rank_pdf, active_names)
    path = write_submission(preds, weights, version, rank_df, smoke=smoke)

    elapsed = time.time() - t0
    log.info("Phase serve done in %.0fs (%.1f min)", elapsed, elapsed / 60)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Task 3.4 blend + submission")
    parser.add_argument("--phase", default="all",
                        choices=["oof", "weights", "serve", "reblend", "all"])
    parser.add_argument("--candidate", type=str, default="mi2_affine_s50",
                        help="Accepted post-processing candidate for --phase reblend")
    parser.add_argument("--version", type=int, default=3,
                        help="Submission version number (default 3)")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke run: 2 months, 2 airports, 100 rounds")
    parser.add_argument("--ignore-gate", action="store_true",
                        help="Do not abort if the OOF reproduction gate fails")
    args = parser.parse_args()

    os.makedirs(DATA_PROCESSED, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    t0 = time.time()
    bundle = None
    if args.phase in ("oof", "all", "serve"):
        bundle = BlendData()

    if args.phase in ("oof", "all"):
        run_oof(bundle, build_components(), args.smoke, args.ignore_gate)

    if args.phase in ("weights", "all"):
        run_weights(args.smoke)

    if args.phase in ("serve", "all"):
        run_serve(args.version, args.smoke, bundle)

    if args.phase == "reblend":
        run_reblend(args.version, args.candidate)

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("run_blend.py --phase %s done in %.0fs (%.1f min)",
             args.phase, elapsed, elapsed / 60)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
