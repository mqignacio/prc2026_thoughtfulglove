#!/usr/bin/env python3
"""Task 4.1 - OOF error analysis for the Task 3.4 blend.

Joins the wide OOF prediction table (data/processed/oof_predictions.parquet)
with segmentation keys from the training DEP rows, forms the blended prediction
using the weights in src/configs/blend.json, and decomposes the raw-target
squared error by airport / runway / hour / aircraft type / month / operator /
stand-zone. Writes src/reports/error_analysis.md and figures under
assets/figures/.

The Stage 0 reproduction gate (blend OOF RMSE == blend.json selected.rmse) is
asserted at load time, so every number downstream is anchored to the recorded
CV result.

Usage:
    uv run python src/run_error_analysis.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from pdc26.airports import AIRPORT_TZ
from pdc26.data import load_training
from pdc26.features import _extract_stand_zone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_error_analysis")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
FIGURES_DIR = os.path.join(_PROJECT_ROOT, "assets", "figures")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")

OOF_PATH = os.path.join(DATA_PROCESSED, "oof_predictions.parquet")
BLEND_CONFIG = os.path.join(CONFIG_DIR, "blend.json")
REPORT_PATH = os.path.join(REPORT_DIR, "error_analysis.md")

SEG_KEYS = [
    "ADEP_mvt",
    "RUNWAY_mvt",
    "MVT_TIME_UTC_mvt",
    "AIRCRAFT_TYPE_mvt",
    "STAND_mvt",
    "AIRCRAFT_OPERATOR_flt",
]
TOP_N = 15
MIN_SEG_N = 2000  # minimum rows for a segment to be rankable
TAIL_FRACTIONS = [0.0003, 0.001, 0.01, 0.05]
SEASON_MONTHS = ["2025-01", "2025-07"]


# ---------------------------------------------------------------------------
# Load + join + reproduction gate
# ---------------------------------------------------------------------------

def load_and_join() -> tuple[pd.DataFrame, dict]:
    """Load OOF predictions, join segment keys, compute blend pred + errors."""
    with open(BLEND_CONFIG) as f:
        blend = json.load(f)
    weights = blend["selected"]["active_weights"]
    expected_rmse = float(blend["selected"]["rmse"])
    log.info("Blend active weights: %s", weights)

    oof = pl.read_parquet(OOF_PATH).with_columns(
        pl.col("MVT_ID_mvt").cast(pl.Int64).alias("mvt_id")
    )
    n_oof = oof.height

    dep = (
        load_training()
        .select(["MVT_ID_mvt"] + SEG_KEYS)
        .with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64).alias("mvt_id"))
        .drop("MVT_ID_mvt")
    )

    joined = oof.join(dep, on="mvt_id", how="inner")
    assert joined.height == n_oof, f"join dropped rows: {joined.height} != {n_oof}"
    log.info("Joined %s OOF rows with segment keys", f"{joined.height:,}")

    joined = joined.with_columns(
        sum(pl.col(f"pred_{k}") * w for k, w in weights.items()).alias("pred_blend")
    ).with_columns(
        (pl.col("pred_blend") - pl.col("actual_raw")).alias("err"),
        ((pl.col("pred_blend") - pl.col("actual_raw")) ** 2).alias("se"),
        (pl.col("pred_blend") - pl.col("actual_raw")).abs().alias("abs_err"),
    )

    recomputed = float(joined["se"].mean()) ** 0.5
    log.info("Reproduction gate: recomputed blend OOF RMSE %.4f (expected %.4f)",
             recomputed, expected_rmse)
    assert abs(recomputed - expected_rmse) < 0.01, (
        f"blend RMSE reproduction failed: {recomputed} vs {expected_rmse}"
    )

    pdf = joined.select(
        ["mvt_id", "fold_month", "actual_raw", "pred_blend",
         "pred_tuned_lgbm", "pred_per_airport_lgbm", "err", "se", "abs_err", *SEG_KEYS]
    ).to_pandas()
    return pdf, blend


def add_derived(pdf: pd.DataFrame) -> pd.DataFrame:
    """Add segmentation columns, using the model's own stand-zone helper verbatim."""
    pdf["airport"] = pdf["ADEP_mvt"].astype("string").fillna("__NA__")
    pdf["runway"] = pdf["RUNWAY_mvt"].astype("string").fillna("__NA__")
    pdf["airport_runway"] = pdf["airport"] + "/" + pdf["runway"]
    pdf["aircraft_type"] = pdf["AIRCRAFT_TYPE_mvt"].astype("string").fillna("__NA__")
    pdf["operator"] = pdf["AIRCRAFT_OPERATOR_flt"].astype("string").fillna("__NA__")

    # Exact parity with the model's stand_zone: apply the helper to unique values.
    raw_stand = pdf["STAND_mvt"].astype("string").fillna("__NA__")
    zone_map = {v: _extract_stand_zone(v) for v in pd.unique(raw_stand)}
    pdf["stand_zone"] = raw_stand.map(zone_map)

    mvt_time = pd.to_datetime(pdf["MVT_TIME_UTC_mvt"], utc=True)
    pdf["utc_hour"] = mvt_time.dt.hour.astype("int16")
    local_hour = np.full(len(pdf), np.nan, dtype=np.float64)
    apt_vals = pdf["airport"].to_numpy()
    for apt, tzname in AIRPORT_TZ.items():
        mask = apt_vals == apt
        if not mask.any():
            continue
        lh = mvt_time[mask].dt.tz_convert(ZoneInfo(tzname))
        local_hour[mask] = (lh.dt.hour + lh.dt.minute / 60.0).to_numpy()
    pdf["local_hour"] = local_hour
    pdf["local_hour_int"] = np.clip(np.floor(local_hour), 0, 23).astype("int16")
    pdf["month"] = pdf["fold_month"]
    pdf["season_matched"] = pdf["fold_month"].isin(SEASON_MONTHS)
    return pdf


# ---------------------------------------------------------------------------
# Decomposition helpers
# ---------------------------------------------------------------------------

def decompose(df: pd.DataFrame, keys: list[str], total_se: float, min_n: int = 1) -> pd.DataFrame:
    """Aggregate error stats per key group, sorted by squared-error contribution."""
    g = (
        df.groupby(keys, dropna=False, observed=True)
        .agg(
            n=("se", "count"),
            se_sum=("se", "sum"),
            abse_sum=("abs_err", "sum"),
            err_sum=("err", "sum"),
            actual_mean=("actual_raw", "mean"),
            pred_mean=("pred_blend", "mean"),
        )
        .reset_index()
    )
    g = g[g["n"] >= min_n].copy()
    g["rmse"] = np.sqrt(g["se_sum"] / g["n"])
    g["mae"] = g["abse_sum"] / g["n"]
    g["bias"] = g["err_sum"] / g["n"]
    g["se_share_pct"] = 100.0 * g["se_sum"] / total_se
    return g.sort_values("se_sum", ascending=False).reset_index(drop=True)


def calibration(df: pd.DataFrame) -> pd.DataFrame:
    """Pred-decile calibration: mean actual vs mean predicted per bin."""
    rank = df["pred_blend"].rank(method="first")
    bins = pd.qcut(rank, 10, labels=False, duplicates="drop")
    work = df.assign(pred_bin=bins)
    g = (
        work.groupby("pred_bin", observed=True)
        .agg(
            n=("se", "count"),
            pred_mean=("pred_blend", "mean"),
            actual_mean=("actual_raw", "mean"),
            rmse=("se", lambda s: float(np.sqrt(np.mean(s.to_numpy())))),
            bias=("err", "mean"),
        )
        .reset_index()
    )
    return g


def tail_analysis(df: pd.DataFrame, total_se: float) -> dict:
    """Global + per-airport concentration of squared error in the extreme tail."""
    order = np.argsort(-df["se"].to_numpy())
    se_sorted = df["se"].to_numpy()[order]
    global_rows = {}
    for frac in TAIL_FRACTIONS:
        k = max(1, int(len(df) * frac))
        global_rows[frac] = 100.0 * se_sorted[:k].sum() / total_se

    per_airport = []
    for apt, sub in df.groupby("airport", observed=True):
        ss = np.sort(sub["se"].to_numpy())[::-1]
        k = max(1, int(len(sub) * 0.001))
        per_airport.append({
            "airport": apt,
            "n": len(sub),
            "tail_n_0.1pct": k,
            "se_share_0.1pct": 100.0 * ss[:k].sum() / sub["se"].sum(),
            "rmse": float(np.sqrt(sub["se"].mean())),
        })
    return {
        "global": global_rows,
        "per_airport": pd.DataFrame(per_airport).sort_values("se_share_0.1pct", ascending=False),
    }


def postproc_headroom(df: pd.DataFrame, total_se: float) -> pd.DataFrame:
    """In-sample diagnostics for post-processing micro-iterations (upper bounds).

    For each airport: optimal 2-component blend weight, RMSE-optimal affine
    recalibration, and the best upper prediction cap. All are fit and scored on
    the same OOF rows (in-sample); the honest leave-one-month-out verdict is
    produced separately by src/run_micro_iterations.py.
    """
    global_w = None
    tune = df["pred_tuned_lgbm"].to_numpy()
    per_air = df["pred_per_airport_lgbm"].to_numpy()
    y = df["actual_raw"].to_numpy()
    p = df["pred_blend"].to_numpy()

    rows = []
    for apt, sub in df.groupby("airport", observed=True):
        yv = sub["actual_raw"].to_numpy()
        tv = sub["pred_tuned_lgbm"].to_numpy()
        pv = sub["pred_per_airport_lgbm"].to_numpy()
        bv = sub["pred_blend"].to_numpy()

        # Optimal weight for pred = w*tuned + (1-w)*per_airport
        d = tv - pv
        w = float(np.clip(np.sum((yv - pv) * d) / max(np.sum(d * d), 1e-9), 0.0, 1.0))
        rmse_opt_w = float(np.sqrt(np.mean((pv + w * d - yv) ** 2)))
        rmse_global_w = float(np.sqrt(np.mean((bv - yv) ** 2)))

        # Affine recalibration y ~ a*p + b
        var = float(np.var(bv))
        slope = float(np.cov(bv, yv)[0, 1] / var) if var > 0 else 1.0
        intercept = float(np.mean(yv) - slope * np.mean(bv))
        rmse_affine = float(np.sqrt(np.mean((slope * bv + intercept - yv) ** 2)))

        # Best upper cap over a quantile grid
        grid = np.linspace(0.95, 0.9999, 40)
        best_cap_rmse, best_q = rmse_global_w, 1.0
        for q in grid:
            cap = float(np.quantile(bv, q))
            rmse_cap = float(np.sqrt(np.mean((np.minimum(bv, cap) - yv) ** 2)))
            if rmse_cap < best_cap_rmse:
                best_cap_rmse, best_q = rmse_cap, float(q)

        rows.append({
            "airport": apt,
            "n": len(sub),
            "bias": float(np.mean(bv - yv)),
            "rmse_global_w": rmse_global_w,
            "w_opt": w,
            "rmse_opt_w": rmse_opt_w,
            "gain_w": rmse_global_w - rmse_opt_w,
            "slope": slope,
            "intercept": intercept,
            "rmse_affine": rmse_affine,
            "gain_affine": rmse_global_w - rmse_affine,
            "cap_q": best_q,
            "rmse_cap": best_cap_rmse,
            "gain_cap": rmse_global_w - best_cap_rmse,
        })
    out = pd.DataFrame(rows).sort_values("gain_affine", ascending=False).reset_index(drop=True)
    return out


def tail_drilldown(df: pd.DataFrame, total_se: float) -> dict:
    """Characterise the extreme-tail rows and the missing-operator subgroup."""
    out: dict = {}
    k = max(1, int(len(df) * 0.0003))
    order = np.argsort(-df["se"].to_numpy())[:k]
    tail = df.iloc[order]
    out["tail_n"] = k
    out["tail_by_airport"] = (
        tail.groupby("airport", observed=True).size().sort_values(ascending=False)
        .rename("rows").reset_index()
    )
    out["tail_by_airport_runway"] = (
        tail.groupby(["airport", "runway"], observed=True).size().sort_values(ascending=False)
        .head(10).rename("rows").reset_index()
    )
    out["tail_operator_na_pct"] = 100.0 * float((tail["operator"] == "__NA__").mean())

    na_mask = df["operator"] == "__NA__"
    out["na_rows"] = int(na_mask.sum())
    out["na_se_share"] = 100.0 * float(df.loc[na_mask, "se"].sum()) / total_se
    na = df[na_mask]
    na_tbl = (
        na.groupby("airport", observed=True)
        .agg(rows=("se", "count"), se_sum=("se", "sum"), rmse=("se", lambda s: float(np.sqrt(np.mean(s.to_numpy())))))
        .reset_index()
    )
    na_tbl["se_share_pct"] = 100.0 * na_tbl["se_sum"] / total_se
    out["na_by_airport"] = na_tbl.sort_values("se_sum", ascending=False).reset_index(drop=True)
    return out


def segment_drilldown(df: pd.DataFrame, segments: list[tuple[str, str]], total_se: float) -> pd.DataFrame:
    """For each airport/runway segment: operator-NA share of segment SE, and
    the segment's share of the global 0.1% tail rows. Reveals whether a segment's
    error is driven by special (metadata-missing) flights."""
    top_order = np.argsort(-df["se"].to_numpy())
    k = max(1, int(len(df) * 0.001))
    tail_ids = set(df.iloc[top_order[:k]]["mvt_id"].to_numpy())
    rows = []
    for apt, rw in segments:
        sub = df[(df["airport"] == apt) & (df["runway"] == rw)]
        if len(sub) == 0:
            continue
        na = sub["operator"] == "__NA__"
        rows.append({
            "airport": apt,
            "runway": rw,
            "n": len(sub),
            "rmse": float(np.sqrt(sub["se"].mean())),
            "bias": float(sub["err"].mean()),
            "se_share_pct": 100.0 * float(sub["se"].sum()) / total_se,
            "na_se_share_pct": 100.0 * float(sub.loc[na, "se"].sum()) / max(float(sub["se"].sum()), 1e-9),
            "rmse_excl_na": float(np.sqrt(sub.loc[~na, "se"].mean())) if (~na).any() else float("nan"),
            "tail_rows_here": int(sub["mvt_id"].isin(tail_ids).sum()),
        })
    return pd.DataFrame(rows)


def component_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Per-airport RMSE of each component and their residual correlation."""
    rows = []
    y = df["actual_raw"].to_numpy()
    for apt, sub in df.groupby("airport", observed=True):
        yv = sub["actual_raw"].to_numpy()
        tv = sub["pred_tuned_lgbm"].to_numpy()
        pv = sub["pred_per_airport_lgbm"].to_numpy()
        rows.append({
            "airport": apt,
            "n": len(sub),
            "rmse_tuned": float(np.sqrt(np.mean((tv - yv) ** 2))),
            "rmse_per_airport": float(np.sqrt(np.mean((pv - yv) ** 2))),
            "rmse_blend": float(np.sqrt(np.mean((sub["pred_blend"].to_numpy() - yv) ** 2))),
            "resid_corr": float(np.corrcoef(tv - yv, pv - yv)[0, 1]),
        })
    return pd.DataFrame(rows).sort_values("rmse_blend", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style():
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figures(d_airport, d_month, d_hour, cal, tail, pdf) -> list[str]:
    _style()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    paths = []

    # 1. Per-airport RMSE (sorted)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    d = d_airport.sort_values("rmse", ascending=False)
    ax.bar(d["airport"], d["rmse"], color="#3b6ea5")
    ax.axhline(float(np.sqrt(pdf["se"].mean())), color="#c0504d", ls="--", lw=1.2,
               label=f"overall {np.sqrt(pdf['se'].mean()):.0f}s")
    ax.set_ylabel("Raw-target RMSE (s)")
    ax.set_xlabel("Departure airport")
    ax.set_title("Blend OOF RMSE by departure airport (2025)")
    ax.legend(frameon=False)
    p = os.path.join(FIGURES_DIR, "error_analysis_per_airport.png")
    fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)

    # 2. Calibration: mean actual vs mean predicted per decile
    fig, ax = plt.subplots(figsize=(5.2, 5))
    lo = float(min(cal["pred_mean"].min(), cal["actual_mean"].min()))
    hi = float(max(cal["pred_mean"].max(), cal["actual_mean"].max()))
    ax.plot([lo, hi], [lo, hi], color="#888888", ls="--", lw=1, label="y = pred")
    ax.plot(cal["pred_mean"], cal["actual_mean"], "o-", color="#3b6ea5", label="observed")
    ax.set_xlabel("Mean blended prediction (s)")
    ax.set_ylabel("Mean actual taxi-out (s)")
    ax.set_title("Error-vs-predicted calibration (pred deciles)")
    ax.legend(frameon=False)
    p = os.path.join(FIGURES_DIR, "error_analysis_calibration.png")
    fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)

    # 3. RMSE and SE share by local hour
    fig, ax1 = plt.subplots(figsize=(8, 4.2))
    h = d_hour.sort_values("local_hour_int")
    ax1.plot(h["local_hour_int"], h["rmse"], "o-", color="#3b6ea5", label="RMSE")
    ax1.set_xlabel("Local hour of day")
    ax1.set_ylabel("Raw-target RMSE (s)", color="#3b6ea5")
    ax2 = ax1.twinx()
    ax2.bar(h["local_hour_int"], h["se_share_pct"], alpha=0.25, color="#c0504d", label="SE share")
    ax2.set_ylabel("Share of total squared error (%)", color="#c0504d")
    ax1.set_title("Blend error by local hour (DEP)")
    p = os.path.join(FIGURES_DIR, "error_analysis_hour.png")
    fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)

    # 4. Pareto: cumulative SE share vs row fraction (log x)
    se_sorted = np.sort(pdf["se"].to_numpy())[::-1]
    cum = np.cumsum(se_sorted) / se_sorted.sum() * 100.0
    frac = np.arange(1, len(cum) + 1) / len(cum) * 100.0
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(frac, cum, color="#3b6ea5")
    for f0 in (0.03, 1.0, 10.0):
        ax.axvline(f0, color="#c0504d", ls=":", lw=0.9)
        ax.annotate(f"{f0:g}% rows", (f0, 5), rotation=90, fontsize=8, color="#c0504d")
    ax.set_xscale("log")
    ax.set_xlabel("Rows, largest error first (% of OOF, log scale)")
    ax.set_ylabel("Cumulative share of squared error (%)")
    ax.set_title("Squared-error concentration (Pareto)")
    p = os.path.join(FIGURES_DIR, "error_analysis_pareto.png")
    fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)

    # 5. Tail concentration by airport
    fig, ax = plt.subplots(figsize=(8, 4.2))
    t = tail["per_airport"].sort_values("se_share_0.1pct", ascending=False)
    ax.bar(t["airport"], t["se_share_0.1pct"], color="#8064a2")
    ax.axhline(10.0, color="#c0504d", ls="--", lw=1, label="10% (uniform baseline)")
    ax.set_ylabel("Share of airport SE from top 0.1% rows (%)")
    ax.set_xlabel("Departure airport")
    ax.set_title("Within-airport squared-error tail concentration")
    ax.legend(frameon=False)
    p = os.path.join(FIGURES_DIR, "error_analysis_tail_by_airport.png")
    fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)

    return paths


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def md_table(df: pd.DataFrame, cols: list[str], headers: list[str], fmts: dict | None = None) -> str:
    """Render a markdown table. Column dtypes (not row values) decide formatting;
    iterrows() upcasts integers to float in mixed rows, so never trust row dtypes."""
    fmts = fmts or {}
    dtypes = {c: df[c].dtype for c in cols}
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, str) or dtypes[c] == object:
                cells.append(str(v))
            elif c in fmts:
                cells.append(fmts[c].format(float(v)))
            elif pd.api.types.is_integer_dtype(dtypes[c]):
                cells.append(f"{int(v):,}")
            else:
                cells.append(f"{float(v):.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(pdf, blend, tables, cal, tail, pp, agree, drill, seg_drill, fig_paths) -> None:
    total_se = pdf["se"].sum()
    rmse = float(np.sqrt(pdf["se"].mean()))
    mae = float(pdf["abs_err"].mean())
    bias = float(pdf["err"].mean())
    sm = pdf[pdf["season_matched"]]
    rmse_sm = float(np.sqrt(sm["se"].mean()))
    n = len(pdf)
    now = datetime.now().astimezone()

    L: list[str] = []
    L.append("# Error Analysis — Task 4.1 (blend OOF, 2025)")
    L.append("")
    L.append(f"Team: `thoughtful-glove` · Generated: {now:%Y-%m-%d %H:%M %Z} · "
             f"Machine: Mac Mini M4 Pro")
    L.append("")
    L.append(f"Source: `data/processed/oof_predictions.parquet` ({n:,} rows) joined to "
             "training DEP segment keys; blend = "
             + " + ".join(f"{k}·{w:.4f}" for k, w in blend["selected"]["active_weights"].items())
             + f" (from `src/configs/blend.json`).")
    L.append("")
    L.append("## 0. Reproduction gate")
    L.append("")
    L.append(f"- Recomputed blend OOF raw RMSE: **{rmse:.2f}s** (recorded `selected.rmse` = "
             f"{float(blend['selected']['rmse']):.2f}s; asserted |Δ| < 0.01s).")
    L.append(f"- Season-matched (Jan+Jul 2025) OOF raw RMSE: **{rmse_sm:.2f}s** "
             f"({len(sm):,} rows).")
    L.append(f"- Join coverage: {n:,}/{n:,} OOF rows matched to a training DEP row (100%).")
    L.append("")
    L.append("## 1. Overall error structure")
    L.append("")
    L.append(f"- Overall RMSE **{rmse:.2f}s**, MAE **{mae:.2f}s**, mean bias **{bias:+.2f}s** "
             "(negative = model under-predicts on average).")
    L.append(f"- Total squared error: {total_se:,.0f}s².")
    L.append("")
    L.append("**Squared-error concentration (whole OOF).**")
    L.append("")
    L.append("| Largest-error rows | Share of total squared error |")
    L.append("|---|---:|")
    for frac in TAIL_FRACTIONS:
        k = max(1, int(n * frac))
        L.append(f"| top {frac*100:.2f}% ({k:,} rows) | {tail['global'][frac]:.1f}% |")
    L.append("")
    L.append("**Within-airport tail concentration (top 0.1% rows per airport).**")
    L.append("")
    t = tail["per_airport"]
    L.append(md_table(t, ["airport", "n", "rmse", "tail_n_0.1pct", "se_share_0.1pct"],
                      ["Airport", "Rows", "RMSE (s)", "Tail rows (0.1%)", "Airport SE from tail (%)"],
                      {"rmse": "{:.2f}", "se_share_0.1pct": "{:.1f}"}))
    L.append("")
    L.append("## 2. Decomposition by segment")
    L.append("")
    L.append("### 2a. By departure airport (all 10)")
    L.append("")
    L.append(md_table(tables["airport"],
                      ["airport", "n", "rmse", "mae", "bias", "se_share_pct"],
                      ["Airport", "Rows", "RMSE (s)", "MAE (s)", "Bias (s)", "SE share (%)"],
                      {"rmse": "{:.2f}", "mae": "{:.2f}", "bias": "{:+.1f}", "se_share_pct": "{:.2f}"}))
    L.append("")
    L.append("### 2b. By fold month (2025)")
    L.append("")
    L.append(md_table(tables["month"],
                      ["month", "n", "rmse", "mae", "bias", "se_share_pct"],
                      ["Month", "Rows", "RMSE (s)", "MAE (s)", "Bias (s)", "SE share (%)"],
                      {"rmse": "{:.2f}", "mae": "{:.2f}", "bias": "{:+.1f}", "se_share_pct": "{:.2f}"}))
    L.append("")
    L.append(f"### 2c. By airport × runway (top {TOP_N} by SE share, n ≥ {MIN_SEG_N:,})")
    L.append("")
    L.append(md_table(tables["runway"].head(TOP_N),
                      ["airport", "runway", "n", "rmse", "bias", "se_share_pct"],
                      ["Airport", "Runway", "Rows", "RMSE (s)", "Bias (s)", "SE share (%)"],
                      {"rmse": "{:.2f}", "bias": "{:+.1f}", "se_share_pct": "{:.2f}"}))
    L.append("")
    L.append("### 2d. By local hour of day (24 buckets)")
    L.append("")
    L.append(md_table(tables["hour_local"].sort_values("local_hour_int"),
                      ["local_hour_int", "n", "rmse", "bias", "se_share_pct"],
                      ["Local hour", "Rows", "RMSE (s)", "Bias (s)", "SE share (%)"],
                      {"local_hour_int": "{:.0f}", "rmse": "{:.2f}", "bias": "{:+.1f}",
                       "se_share_pct": "{:.2f}"}))
    L.append("")
    L.append(f"### 2e. By aircraft type (top {TOP_N} by SE share, n ≥ {MIN_SEG_N:,})")
    L.append("")
    L.append(md_table(tables["aircraft"].head(TOP_N),
                      ["aircraft_type", "n", "rmse", "bias", "se_share_pct"],
                      ["Aircraft type", "Rows", "RMSE (s)", "Bias (s)", "SE share (%)"],
                      {"rmse": "{:.2f}", "bias": "{:+.1f}", "se_share_pct": "{:.2f}"}))
    L.append("")
    L.append(f"### 2f. By operator (top {TOP_N} by SE share, n ≥ {MIN_SEG_N:,})")
    L.append("")
    L.append(md_table(tables["operator"].head(TOP_N),
                      ["operator", "n", "rmse", "bias", "se_share_pct"],
                      ["Operator", "Rows", "RMSE (s)", "Bias (s)", "SE share (%)"],
                      {"rmse": "{:.2f}", "bias": "{:+.1f}", "se_share_pct": "{:.2f}"}))
    L.append("")
    L.append(f"### 2g. By stand zone (top {TOP_N} by SE share, n ≥ {MIN_SEG_N:,})")
    L.append("")
    L.append(md_table(tables["stand"].head(TOP_N),
                      ["stand_zone", "n", "rmse", "bias", "se_share_pct"],
                      ["Stand zone", "Rows", "RMSE (s)", "Bias (s)", "SE share (%)"],
                      {"rmse": "{:.2f}", "bias": "{:+.1f}", "se_share_pct": "{:.2f}"}))
    L.append("")
    L.append(f"### 2h. By UTC hour of day (24 buckets)")
    L.append("")
    L.append(md_table(tables["hour_utc"].sort_values("utc_hour"),
                      ["utc_hour", "n", "rmse", "bias", "se_share_pct"],
                      ["UTC hour", "Rows", "RMSE (s)", "Bias (s)", "SE share (%)"],
                      {"utc_hour": "{:.0f}", "rmse": "{:.2f}", "bias": "{:+.1f}",
                       "se_share_pct": "{:.2f}"}))
    L.append("")
    L.append("## 3. Error vs predicted (calibration)")
    L.append("")
    L.append(md_table(cal,
                      ["pred_bin", "n", "pred_mean", "actual_mean", "bias", "rmse"],
                      ["Pred decile", "Rows", "Mean pred (s)", "Mean actual (s)", "Bias (s)", "RMSE (s)"],
                      {"pred_bin": "{:.0f}", "pred_mean": "{:.0f}", "actual_mean": "{:.0f}",
                       "bias": "{:+.1f}", "rmse": "{:.1f}"}))
    L.append("")
    L.append(f"![Calibration]({'../../assets/figures/' + os.path.basename(fig_paths[1])})")
    L.append("")
    L.append("## 4. Top-3 error-contributing segments")
    L.append("")
    top3 = tables["runway"].head(3)
    for i, (_, r) in enumerate(top3.iterrows(), 1):
        L.append(f"**S{i}. {r['airport']} / runway {r['runway']}** — {int(r['n']):,} rows, "
                 f"RMSE {r['rmse']:.1f}s, bias {r['bias']:+.1f}s, "
                 f"**{r['se_share_pct']:.2f}% of total OOF squared error**.")
    L.append("")
    L.append("## 5. Micro-iteration candidates (ranked)")
    L.append("")
    L.append("In-sample post-processing headroom per airport (upper bounds — honest "
             "leave-one-month-out verdicts are produced by `src/run_micro_iterations.py`):")
    L.append("")
    L.append(md_table(pp.sort_values("gain_affine", ascending=False),
                      ["airport", "bias", "rmse_global_w", "w_opt", "rmse_opt_w", "gain_w",
                       "slope", "rmse_affine", "gain_affine", "cap_q", "gain_cap"],
                      ["Airport", "Bias (s)", "RMSE global-w", "w* opt", "RMSE opt-w",
                       "Gain w", "Slope", "RMSE affine", "Gain affine", "Cap q", "Gain cap"],
                      {"bias": "{:+.1f}", "rmse_global_w": "{:.1f}", "w_opt": "{:.3f}",
                       "rmse_opt_w": "{:.1f}", "gain_w": "{:+.2f}", "slope": "{:.3f}",
                       "rmse_affine": "{:.1f}", "gain_affine": "{:+.2f}", "cap_q": "{:.4f}",
                       "gain_cap": "{:+.2f}"}))
    L.append("")
    L.append("Per-airport component agreement:")
    L.append("")
    L.append(md_table(agree,
                      ["airport", "rmse_tuned", "rmse_per_airport", "rmse_blend", "resid_corr"],
                      ["Airport", "RMSE tuned", "RMSE per-airport", "RMSE blend", "Residual corr"],
                      {"rmse_tuned": "{:.1f}", "rmse_per_airport": "{:.1f}",
                       "rmse_blend": "{:.1f}", "resid_corr": "{:.3f}"}))
    L.append("")
    L.append("## 6. Tail drill-down (who the extreme errors are)")
    L.append("")
    L.append(f"The top 0.03% rows ({drill['tail_n']:,}) carry the bulk of squared error. "
             f"**{drill['tail_operator_na_pct']:.1f}%** of those rows have a missing operator ID.")
    L.append("")
    L.append("**Extreme-tail rows by airport.**")
    L.append("")
    L.append(md_table(drill["tail_by_airport"], ["airport", "rows"], ["Airport", "Tail rows"]))
    L.append("")
    L.append("**Extreme-tail rows by airport × runway (top 10).**")
    L.append("")
    L.append(md_table(drill["tail_by_airport_runway"], ["airport", "runway", "rows"],
                      ["Airport", "Runway", "Tail rows"]))
    L.append("")
    L.append(f"**Missing-operator subgroup** (`AIRCRAFT_OPERATOR_flt` null): {drill['na_rows']:,} rows "
             f"({100.0*drill['na_rows']/n:.2f}% of OOF) carry **{drill['na_se_share']:.1f}% of all squared error**.")
    L.append("")
    L.append(md_table(drill["na_by_airport"], ["airport", "rows", "rmse", "se_share_pct"],
                      ["Airport", "Rows", "RMSE (s)", "Global SE share (%)"],
                      {"rmse": "{:.1f}", "se_share_pct": "{:.2f}"}))
    L.append("")
    L.append("**Top-segment drill-down** (is the segment's error the special flights?):")
    L.append("")
    L.append(md_table(seg_drill,
                      ["airport", "runway", "n", "rmse", "bias", "se_share_pct",
                       "na_se_share_pct", "rmse_excl_na", "tail_rows_here"],
                      ["Airport", "Runway", "Rows", "RMSE (s)", "Bias (s)", "SE share (%)",
                       "Segment SE from missing-op (%)", "RMSE excl. missing-op (s)", "Global-tail rows here"],
                      {"rmse": "{:.1f}", "bias": "{:+.1f}", "se_share_pct": "{:.2f}",
                       "na_se_share_pct": "{:.1f}", "rmse_excl_na": "{:.1f}"}))
    L.append("")
    L.append("## 7. Figures")
    L.append("")
    for p in fig_paths:
        L.append(f"![{os.path.basename(p)}](../../assets/figures/{os.path.basename(p)})")
        L.append("")
    L.append("_All figures generated by `src/run_error_analysis.py`; all tables are "
             "script-computed._")
    L.append("")

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(L))
    log.info("Wrote report: %s", REPORT_PATH)


def main() -> None:
    t0 = datetime.now()
    pdf, blend = load_and_join()
    pdf = add_derived(pdf)
    total_se = float(pdf["se"].sum())

    tables = {
        "airport": decompose(pdf, ["airport"], total_se),
        "month": decompose(pdf, ["month"], total_se),
        "runway": decompose(pdf, ["airport", "runway"], total_se, min_n=MIN_SEG_N),
        "hour_local": decompose(pdf, ["local_hour_int"], total_se),
        "hour_utc": decompose(pdf, ["utc_hour"], total_se),
        "aircraft": decompose(pdf, ["aircraft_type"], total_se, min_n=MIN_SEG_N),
        "operator": decompose(pdf, ["operator"], total_se, min_n=MIN_SEG_N),
        "stand": decompose(pdf, ["stand_zone"], total_se, min_n=MIN_SEG_N),
        "apt_hour": decompose(pdf, ["airport", "local_hour_int"], total_se),
    }
    cal = calibration(pdf)
    tail = tail_analysis(pdf, total_se)
    pp = postproc_headroom(pdf, total_se)
    agree = component_agreement(pdf)
    drill = tail_drilldown(pdf, total_se)
    top3 = list(zip(tables["runway"].head(3)["airport"], tables["runway"].head(3)["runway"]))
    seg_drill = segment_drilldown(pdf, top3, total_se)
    fig_paths = save_figures(tables["airport"], tables["month"], tables["hour_local"], cal, tail, pdf)
    write_report(pdf, blend, tables, cal, tail, pp, agree, drill, seg_drill, fig_paths)

    log.info("Top-3 airport×runway segments:")
    for _, r in tables["runway"].head(3).iterrows():
        log.info("  %s/%s  n=%s  rmse=%.1f  se_share=%.2f%%",
                 r["airport"], r["runway"], f"{int(r['n']):,}", r["rmse"], r["se_share_pct"])
    log.info("Done in %.1f min", (datetime.now() - t0).total_seconds() / 60)


if __name__ == "__main__":
    main()
