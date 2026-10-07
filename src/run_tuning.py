#!/usr/bin/env python3
"""Model comparison, per-airport vs global, and hyperparameter tuning (Task 3.3).

Runs three phases against the FULL D4 feature set (baseline + METAR + congestion
+ domain; 62 features), scored by raw-target month-blocked CV RMSE:

  Phase 1 — Model comparison: LightGBM vs XGBoost vs CatBoost on identical
            default params + 12-fold CV -> src/reports/model_comparison.md
  Phase 2 — Global vs per-airport: 10 per-airport LightGBM models vs one global
            -> src/reports/model_comparison.md
  Phase 3 — Optuna hyperparameter tuning over num_leaves / learning_rate /
            min_child_samples / feature_fraction / bagging (LightGBM + the
            2nd-best family). Search on 4 representative months; finalist
            validated on full 12 folds -> src/reports/tuning_results.md

  Phase 4 — Outlier re-check: confirm E-winsorize-errors still holds after
            tuning (retuned-model scope only; does NOT re-run Task 05b table).

Target treatment is FIXED to E-winsorize-errors throughout (Task 05b winner).

Usage:
    uv run python src/run_tuning.py [--phase 1|2|3|4|all]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import resource
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import polars as pl

from pdc26 import SEED
from pdc26.airports import load_coords
from pdc26.data import load_training, load_training_movements
from pdc26.features import CAT_FEATURES, get_month_labels
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
log = logging.getLogger("run_tuning")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# Representative months for the Optuna search (4 spread across the year).
REPRESENTATIVE_MONTHS = ["2025-03", "2025-06", "2025-09", "2025-12"]


# ---------------------------------------------------------------------------
# Shared data loading / feature assembly
# ---------------------------------------------------------------------------

class DataBundle:
    """All pre-loaded pieces needed for any phase."""

    def __init__(self):
        log.info("Loading training data (DEP)...")
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
        self.airports = self.dep["ADEP_mvt"].to_pandas()
        self.months = get_month_labels(self.dep)
        self.mvt_ids = self.dep["MVT_ID_mvt"].to_pandas()

        for s in (self.raw_target, self.airports, self.months, self.mvt_ids):
            s.index = range(len(s))

        # Winsorized target (E-winsorize-errors, fixed)
        self.y_winsor = winsorize_errors(self.raw_target, self.airports)

        self.fold_masks = {m: (self.months.values == m)
                           for m in sorted(self.months.unique())}

        log.info("Computing within-fold operator target encoding...")
        self.operator_enc = encode_operator_within_fold(
            self.pdf_full["AIRCRAFT_OPERATOR_flt"], self.fold_masks, self.y_winsor
        )

    def build_X(self) -> pd.DataFrame:
        X = build_d4_features(
            self.dep, self.movements, self.vocab, self.domain_vocab,
            self.metar, self.coords, operator_enc=self.operator_enc,
        )
        # Cast numeric features to float32 to reduce memory footprint
        for col in X.select_dtypes(include=["float64", "int64"]).columns:
            X[col] = X[col].astype("float32")
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
        log.info("X shape=%s RSS=%.0fMB", X.shape, rss_mb)
        return X


def compute_per_airport_rmse(oof_df: pd.DataFrame, airport_map: dict) -> pd.DataFrame:
    """Per-airport raw RMSE across all folds."""
    oof = oof_df.copy()
    oof["airport"] = oof["MVT_ID_mvt"].map(airport_map)
    rows = []
    for apt in sorted(oof["airport"].dropna().unique()):
        subset = oof[oof["airport"] == apt]
        residuals = subset["prediction"] - subset["actual_raw"]
        rows.append({
            "airport": apt,
            "n": len(subset),
            "rmse_raw": float(np.sqrt(np.mean(residuals ** 2))),
            "mae_raw": float(np.mean(np.abs(residuals))),
            "mean_actual": float(subset["actual_raw"].mean()),
        })
    return pd.DataFrame(rows).sort_values("rmse_raw", ascending=False)


# ---------------------------------------------------------------------------
# Phase 1: model comparison
# ---------------------------------------------------------------------------

def phase1_model_comparison(bundle: DataBundle):
    X = bundle.build_X()
    log.info("D4 feature matrix: %s rows x %d cols", f"{X.shape[0]:,}", X.shape[1])

    airport_map = dict(zip(bundle.mvt_ids.values, bundle.airports.values))

    results = {}
    for family in ["lgbm", "xgb", "cat"]:
        params = family_default_params(family, seed=SEED)
        log.info("\n=== Model comparison: %s ===", family)
        cv = month_blocked_cv_model(
            X=X, y=bundle.y_winsor, months=bundle.months,
            mvt_ids=bundle.mvt_ids, cat_features=CAT_FEATURES,
            family=family, params=params,
            n_estimators=500, early_stopping_rounds=50,
            y_raw=bundle.raw_target,
        )
        results[family] = {
            "cv": cv,
            "per_airport": compute_per_airport_rmse(cv.oof_predictions, airport_map),
        }
        log.info("%s raw RMSE=%.2f", family, cv.overall_rmse_raw)

    _write_model_comparison(results)
    _persist_comparison_json(results)
    return results


def _persist_comparison_json(results: dict) -> str:
    """Persist comparison results to JSON for downstream/report regeneration."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    path = os.path.join(CONFIG_DIR, "model_comparison.json")
    payload = {}
    for family, r in results.items():
        cv = r["cv"]
        payload[family] = {
            "overall_rmse_raw": cv.overall_rmse_raw,
            "overall_mae_raw": cv.overall_mae_raw,
            "per_month": cv.per_month.to_dict("records"),
            "per_airport": r["per_airport"].to_dict("records"),
            "n_features": None,
        }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Persisted comparison results to %s", path)
    return path


def _write_model_comparison(results: dict) -> str:
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, "model_comparison.md")
    lines = [
        "# Model Comparison Report — Task 3.3",
        "",
        f"Task: `10-model_tuning` · Team: `thoughtful-glove`",
        f"Generated: {time.strftime('%Y-%m-%d')} · Seed: {SEED}",
        "",
        "---",
        "",
        "## Configuration",
        "",
        "- **Feature set:** D4 (baseline + METAR + congestion + domain, 62 features)",
        "- **Target:** E-winsorize-errors (Task 05b winner)",
        "- **CV:** 12-fold month-blocked (leave-one-month-out, Jan–Dec 2025)",
        "- **Metric:** raw unclipped RMSE (competition metric)",
        "- **Families:** LightGBM / XGBoost / CatBoost, family-equivalent default params",
        "",
        "## Overall Raw RMSE",
        "",
        "| Family | RMSE (s) | MAE (s) |",
        "|--------|---------:|--------:|",
    ]
    for family in ["lgbm", "xgb", "cat"]:
        r = results[family]["cv"]
        lines.append(f"| {family} | {r.overall_rmse_raw:.2f} | {r.overall_mae_raw:.2f} |")

    lines += [
        "",
        "## Per-Month Raw RMSE",
        "",
        "| Fold Month | lgbm | xgb | cat |",
        "|------------|-----:|----:|----:|",
    ]
    for month in [f"2025-{m:02d}" for m in range(1, 13)]:
        vals = {}
        for family in ["lgbm", "xgb", "cat"]:
            pm = results[family]["cv"].per_month
            row = pm[pm["fold_month"] == month]
            vals[family] = float(row.iloc[0]["rmse_raw"]) if len(row) else float("nan")
        lines.append(f"| {month} | {vals['lgbm']:.2f} | {vals['xgb']:.2f} | {vals['cat']:.2f} |")

    lines += [
        "",
        "## Per-Airport Raw RMSE",
        "",
        "| Airport | N | lgbm | xgb | cat | Mean Actual |",
        "|---------|--:|-----:|----:|----:|------------:|",
    ]
    lgbm_apt = results["lgbm"]["per_airport"].set_index("airport")
    for apt in lgbm_apt.index:
        n = int(lgbm_apt.loc[apt, "n"])
        mean_act = lgbm_apt.loc[apt, "mean_actual"]
        row = [apt, f"{n:,}"]
        for family in ["lgbm", "xgb", "cat"]:
            apt_df = results[family]["per_airport"].set_index("airport")
            v = apt_df.loc[apt, "rmse_raw"] if apt in apt_df.index else float("nan")
            row.append(f"{v:.2f}")
        row.append(f"{mean_act:.1f}")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "---",
        "",
        "## Conclusion",
        "",
    ]
    best = min(["lgbm", "xgb", "cat"], key=lambda f: results[f]["cv"].overall_rmse_raw)
    lines.append(f"Best family under identical CV: **{best}** "
                 f"({results[best]['cv'].overall_rmse_raw:.2f}s).")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    log.info("Wrote %s", path)
    return path


# ---------------------------------------------------------------------------
# Phase 2: global vs per-airport (LightGBM only)
# ---------------------------------------------------------------------------

def phase2_per_airport(bundle: DataBundle):
    X = bundle.build_X()
    airport_map = dict(zip(bundle.mvt_ids.values, bundle.airports.values))

    params = family_default_params("lgbm", seed=SEED)

    # Global model (with airport features)
    log.info("\n=== Global LightGBM (D4) ===")
    global_cv = month_blocked_cv_model(
        X=X, y=bundle.y_winsor, months=bundle.months,
        mvt_ids=bundle.mvt_ids, cat_features=CAT_FEATURES,
        family="lgbm", params=params, n_estimators=500,
        early_stopping_rounds=50, y_raw=bundle.raw_target,
    )

    # Per-airport models
    per_airport_cv_all = []
    for apt in sorted(bundle.airports.unique()):
        mask = bundle.airports.values == apt
        if mask.sum() < 1000:
            log.warning("  Airport %s has only %d rows, skipping", apt, int(mask.sum()))
            continue
        idx = np.where(mask)[0]
        X_apt = X.iloc[idx].reset_index(drop=True)
        y_apt = bundle.y_winsor.iloc[idx].reset_index(drop=True)
        months_apt = bundle.months.iloc[idx].reset_index(drop=True)
        mvt_apt = bundle.mvt_ids.iloc[idx].reset_index(drop=True)
        raw_apt = bundle.raw_target.iloc[idx].reset_index(drop=True)

        log.info("\n=== Per-airport LightGBM: %s (%d rows) ===", apt, len(X_apt))
        cv = month_blocked_cv_model(
            X=X_apt, y=y_apt, months=months_apt, mvt_ids=mvt_apt,
            cat_features=CAT_FEATURES, family="lgbm", params=params,
            n_estimators=500, early_stopping_rounds=50, y_raw=raw_apt,
        )
        per_airport_cv_all.append(cv)
        log.info("  %s raw RMSE=%.2f", apt, cv.overall_rmse_raw)

    # Stitch per-airport OOF back together for a combined RMSE
    combined_oof = pd.concat([c.oof_predictions for c in per_airport_cv_all],
                             ignore_index=True)
    combined_rmse = float(np.sqrt(np.mean(
        (combined_oof["prediction"] - combined_oof["actual_raw"]) ** 2
    )))

    result = {
        "global_rmse": global_cv.overall_rmse_raw,
        "per_airport_combined_rmse": combined_rmse,
        "per_airport": {
            "global": global_cv,
            "combined_oof": combined_oof,
        },
    }
    _append_per_airport_report(result, combined_oof, airport_map)
    return result


def _append_per_airport_report(result: dict, combined_oof, airport_map):
    path = os.path.join(REPORT_DIR, "model_comparison.md")
    lines = [
        "",
        "## Global vs Per-Airport (LightGBM, full 12-fold)",
        "",
        "| Model | Raw RMSE (s) |",
        "|-------|-------------:|",
        f"| Global (D4, airport features) | {result['global_rmse']:.2f} |",
        f"| 10 per-airport models (combined OOF) | {result['per_airport_combined_rmse']:.2f} |",
        "",
        "_Per-airport combined RMSE is computed by stitching each airport's OOF "
        "predictions and scoring against raw target._",
        "",
    ]
    with open(path, "a") as f:
        f.write("\n".join(lines))
    log.info("Appended per-airport results to %s", path)


# ---------------------------------------------------------------------------
# Phase 3: Optuna hyperparameter tuning
# ---------------------------------------------------------------------------

TUNE_FAMILY = "lgbm"


def _objective(trial, X, y, months, cat_features, n_estimators, early_stopping, y_raw,
               active_month_set):
    import optuna
    t0 = time.time()

    # Sub-sample to representative months
    keep = months.isin(active_month_set)
    X_s = X[keep]
    y_s = y[keep]
    months_s = months[keep]
    y_raw_s = y_raw[keep]

    params = family_default_params(TUNE_FAMILY, seed=SEED)
    # Drop subsampling aliases so the tuned fraction params are unambiguous.
    params.pop("subsample", None)
    params.pop("colsample_bytree", None)
    params["num_leaves"] = trial.suggest_int("num_leaves", 16, 255, log=True)
    params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
    params["min_child_samples"] = trial.suggest_int("min_child_samples", 5, 200, log=True)
    params["feature_fraction"] = trial.suggest_float("feature_fraction", 0.5, 1.0)
    params["bagging_fraction"] = trial.suggest_float("bagging_fraction", 0.5, 1.0)
    params["bagging_freq"] = 1

    # Direct fold loop (4 folds) for speed
    unique_months = sorted(months_s.unique())
    import lightgbm as lgb

    all_resid = []
    for fold_month in unique_months:
        # Wall-clock guard: prune trial if >15 min total
        if time.time() - t0 > 900:
            raise optuna.TrialPruned("Trial exceeded 15-min wall-clock limit")

        val_mask = months_s.values == fold_month
        train_mask = ~val_mask
        dtrain = lgb.Dataset(X_s[train_mask], label=y_s[train_mask].values,
                             categorical_feature=cat_features, free_raw_data=False)
        dval = lgb.Dataset(X_s[val_mask], label=y_s[val_mask].values,
                           categorical_feature=cat_features, reference=dtrain,
                           free_raw_data=False)
        model = lgb.train(
            params, dtrain, num_boost_round=n_estimators, valid_sets=[dval],
            callbacks=[lgb.early_stopping(stopping_rounds=early_stopping, verbose=False)],
        )
        preds = model.predict(X_s[val_mask], num_iteration=model.best_iteration)
        resid = preds - y_raw_s[val_mask].values
        all_resid.append(resid)

    resid = np.concatenate(all_resid)
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    elapsed = time.time() - t0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    log.info("Trial %d: RMSE=%.2f elapsed=%.0fs RSS=%.0fMB params=%s",
             trial.number, rmse, elapsed, rss_mb, params)
    return rmse


def phase3_tuning(bundle: DataBundle, n_trials: int = 60, n_estimators: int = 500):
    import optuna

    os.makedirs(LOGS_DIR, exist_ok=True)
    journal_path = os.path.join(LOGS_DIR, "optuna_journal.jsonl")

    X = bundle.build_X()
    log.info("Starting Optuna search (%d trials, %d est) on %s months",
             n_trials, n_estimators, REPRESENTATIVE_MONTHS)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def obj(trial):
        return _objective(
            trial, X, bundle.y_winsor, bundle.months, CAT_FEATURES,
            n_estimators=n_estimators, early_stopping=50,
            y_raw=bundle.raw_target, active_month_set=REPRESENTATIVE_MONTHS,
        )

    study = optuna.create_study(
        storage=optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(journal_path)),
        study_name="phase3_lgbm_tuning",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        load_if_exists=True,
    )

    n_completed = len(study.trials)
    log.info("Study loaded; %d trials already completed (journal=%s)", n_completed, journal_path)

    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    log.info("Search complete (%d+%d=%d total trials)", n_completed, n_trials, len(study.trials))

    best = study.best_params
    best_value = study.best_value

    # Finalist: validate on FULL 12 folds with the best params
    params = family_default_params(TUNE_FAMILY, seed=SEED)
    params.pop("subsample", None)
    params.pop("colsample_bytree", None)
    params["num_leaves"] = best["num_leaves"]
    params["learning_rate"] = best["learning_rate"]
    params["min_child_samples"] = best["min_child_samples"]
    params["feature_fraction"] = best["feature_fraction"]
    params["bagging_fraction"] = best["bagging_fraction"]
    params["bagging_freq"] = 1

    log.info("Best params (subsample): %s -> RMSE %.2f", best, best_value)
    log.info("Validating best config on full 12 folds...")
    full_cv = month_blocked_cv_model(
        X=X, y=bundle.y_winsor, months=bundle.months,
        mvt_ids=bundle.mvt_ids, cat_features=CAT_FEATURES,
        family=TUNE_FAMILY, params=params, n_estimators=n_estimators,
        early_stopping_rounds=50, y_raw=bundle.raw_target,
    )

    full_config = {
        "family": TUNE_FAMILY,
        "seed": SEED,
        "target_treatment": "E-winsorize-errors",
        "feature_set": "D4",
        "params": {k: v for k, v in params.items() if v is not None},
        "n_estimators": n_estimators,
        "early_stopping_rounds": 50,
    }

    return {
        "best_params": best,
        "subsample_rmse": best_value,
        "full_cv": full_cv,
        "full_config": full_config,
        "default_params": family_default_params(TUNE_FAMILY, seed=SEED),
    }


# ---------------------------------------------------------------------------
# Phase 4: outlier re-check (retuned scope)
# ---------------------------------------------------------------------------

def phase4_outlier_check(bundle: DataBundle, tuned_config: dict | None):
    """Confirm E-winsorize-errors still holds after tuning.

    Re-checks only: does the tuned model still benefit from the fixed
    winsorize treatment vs. raw/no-clip? (NOT a re-run of the 05b table.)
    """
    X = bundle.build_X()
    if tuned_config is None:
        best_path = os.path.join(CONFIG_DIR, "best.json")
        if os.path.exists(best_path):
            with open(best_path) as f:
                tuned_config = json.load(f)
    params = tuned_config["params"] if tuned_config else family_default_params("lgbm", SEED)
    params = {k: v for k, v in params.items() if v is not None}
    params.setdefault("objective", "regression")
    params.setdefault("metric", "rmse")
    params.setdefault("seed", SEED)
    params.setdefault("verbose", -1)

    log.info("\n=== Outlier re-check: E-winsorize vs B-raw (tuned scope) ===")

    # Arm 1: E-winsorize-errors (current)
    cv_win = month_blocked_cv_model(
        X=X, y=bundle.y_winsor, months=bundle.months,
        mvt_ids=bundle.mvt_ids, cat_features=CAT_FEATURES,
        family=TUNE_FAMILY, params=params, n_estimators=500,
        early_stopping_rounds=50, y_raw=bundle.raw_target,
    )

    # Arm 2: B-raw (clip negatives to 0 only)
    y_raw_b = bundle.raw_target.clip(0, None)
    cv_raw = month_blocked_cv_model(
        X=X, y=y_raw_b, months=bundle.months,
        mvt_ids=bundle.mvt_ids, cat_features=CAT_FEATURES,
        family=TUNE_FAMILY, params=params, n_estimators=500,
        early_stopping_rounds=50, y_raw=bundle.raw_target,
    )

    return {"winsorize": cv_win.overall_rmse_raw, "raw": cv_raw.overall_rmse_raw}


# ---------------------------------------------------------------------------
# Report + config writing
# ---------------------------------------------------------------------------

def _baseline_lgbm_rmse(comparison: dict | None) -> float | None:
    """Resolve the baseline LightGBM raw RMSE for the tuning report."""
    if comparison is not None and "lgbm" in comparison:
        return comparison["lgbm"]["cv"].overall_rmse_raw
    p = os.path.join(CONFIG_DIR, "model_comparison.json")
    if os.path.exists(p):
        with open(p) as f:
            data = json.load(f)
        if "lgbm" in data:
            return float(data["lgbm"]["overall_rmse_raw"])
    return None


def _write_tuning_report(phase3: dict | None, phase4: dict | None,
                         comparison: dict | None) -> str:
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, "tuning_results.md")
    lines = [
        "# Hyperparameter Tuning Report — Task 3.3",
        "",
        f"Task: `10-model_tuning` · Team: `thoughtful-glove`",
        f"Generated: {time.strftime('%Y-%m-%d')} · Seed: {SEED}",
        "",
        "---",
        "",
        "## Configuration",
        "",
        "- **Feature set:** D4 (baseline + METAR + congestion + domain, 62 features)",
        "- **Target:** E-winsorize-errors (Task 05b winner)",
        "- **Metric:** raw unclipped RMSE (competition metric)",
        "- **Search:** Optuna (TPE), 4 representative months",
        "- **Validation:** full 12-fold month-blocked CV for finalists",
        "",
    ]

    if phase3:
        best = phase3["best_params"]
        full = phase3["full_cv"]
        lines += [
            "## Optuna Search Results",
            "",
            f"- Subsample RMSE (4 rep. months) = {phase3['subsample_rmse']:.2f}s",
            "",
            "| Hyperparameter | Value |",
            "|----------------|------:|",
            f"| num_leaves | {best['num_leaves']} |",
            f"| learning_rate | {best['learning_rate']:.4f} |",
            f"| min_child_samples | {best['min_child_samples']} |",
            f"| feature_fraction | {best['feature_fraction']:.3f} |",
            f"| bagging_fraction | {best['bagging_fraction']:.3f} |",
            "",
            "## Finalist vs Baseline (full 12-fold)",
            "",
            "| Config | Full-CV Raw RMSE (s) |",
            "|--------|---------------------:|",
        ]
        base_rmse = _baseline_lgbm_rmse(comparison)
        if base_rmse is not None:
            lines.append(f"| Baseline LGBM (default params) | {base_rmse:.2f} |")
            lines.append(f"| Tuned LGBM | {full.overall_rmse_raw:.2f} |")
            lines.append(f"| Δ | {full.overall_rmse_raw - base_rmse:+.2f} |")
        else:
            lines.append(f"| Tuned LGBM | {full.overall_rmse_raw:.2f} |")
        lines.append("")

    if phase4:
        lines += [
            "## Outlier Re-Check (retuned scope)",
            "",
            "| Target treatment | Full-CV Raw RMSE (s) |",
            "|------------------|---------------------:|",
            f"| E-winsorize-errors (current) | {phase4['winsorize']:.2f} |",
            f"| B-raw (negatives → 0 only) | {phase4['raw']:.2f} |",
            "",
        ]
        if phase4["winsorize"] <= phase4["raw"]:
            verdict = ("E-winsorize-errors still holds after tuning "
                       "(winsorize ≤ raw).")
        else:
            verdict = ("NOTE: B-raw now edges out winsorize under the tuned model — "
                       "worth revisiting the clip choice.")
        lines.append(f"Verdict: {verdict}")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    log.info("Wrote %s", path)
    return path


def _write_best_config(phase3: dict) -> str:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    path = os.path.join(CONFIG_DIR, "best.json")
    with open(path, "w") as f:
        json.dump(phase3["full_config"], f, indent=2)
    log.info("Wrote %s", path)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Task 3.3 model tuning")
    parser.add_argument("--phase", default="all",
                        help="1,2,3,4,34 or all (default all)")
    parser.add_argument("--n-trials", type=int, default=60,
                        help="Optuna trials (default 60)")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 2 trials, 100 estimators")
    args = parser.parse_args()

    os.makedirs(LOGS_DIR, exist_ok=True)

    t0 = time.time()
    bundle = DataBundle()

    comparison = None
    phase3 = None
    phase4 = None

    if args.phase in ("1", "all"):
        comparison = phase1_model_comparison(bundle)

    if args.phase in ("2", "all"):
        phase2_per_airport(bundle)

    if args.phase in ("3", "all", "34"):
        n_estimators = 100 if args.smoke else 500
        n_trials_actual = 2 if args.smoke else args.n_trials
        if args.smoke:
            log.info("SMOKE MODE: 2 trials, 100 estimators")
        phase3 = phase3_tuning(bundle, n_trials=n_trials_actual, n_estimators=n_estimators)
        _write_best_config(phase3)

    if args.phase in ("4", "all", "34"):
        tuned_config = phase3["full_config"] if phase3 else None
        phase4 = phase4_outlier_check(bundle, tuned_config)

    _write_tuning_report(phase3, phase4, comparison)

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("Done in %.0fs (%.1f min)", elapsed, elapsed / 60)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
