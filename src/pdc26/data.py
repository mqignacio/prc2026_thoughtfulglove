"""Data loading for the PRC-OpenSky 2026 baseline pipeline.

Loads monthly training parquet files and ranking parquet, filters to PHASE=DEP,
deduplicates by MVT_ID, and returns Polars DataFrames ready for feature engineering.
"""

from __future__ import annotations

import glob
import logging
import os

import polars as pl

log = logging.getLogger(__name__)

# Project root is two levels up from this file (src/pdc26/data.py -> project root)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_RAW = os.path.join(_PROJECT_ROOT, "data", "raw")


def load_training(filter_months: list[str] | None = None) -> pl.DataFrame:
    """Load and concatenate all monthly training parquet files, filtered to DEP.

    Args:
        filter_months: Optional list of month prefixes (e.g. ["2025-01", "2025-06"])
            to load only a subset for quick iteration. If None, loads all 12 months.

    Returns:
        Polars DataFrame with all DEP rows, deduplicated by MVT_ID_mvt,
        sorted by MVT_TIME_UTC_mvt.
    """
    pattern = os.path.join(DATA_RAW, "training_*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No training parquet files found matching {pattern}")

    # Optionally filter to specific months
    if filter_months:
        files = [f for f in files if any(m in os.path.basename(f) for m in filter_months)]
        log.info("Loading %d filtered training files: %s", len(files),
                 [os.path.basename(f) for f in files])
    else:
        log.info("Loading %d training files", len(files))

    frames: list[pl.DataFrame] = []
    for f in files:
        lf = (
            pl.scan_parquet(f)
            .filter(pl.col("PHASE_mvt") == "DEP")
        )
        df = lf.collect()
        month = os.path.basename(f).split("_")[1][:7]
        log.info("  %s: %s DEP rows", month, f"{df.height:,}")
        frames.append(df)

    combined = pl.concat(frames)

    # Deduplicate by MVT_ID_mvt (EDA confirmed 0 duplicates, but defensive)
    before = combined.height
    combined = combined.unique(subset=["MVT_ID_mvt"], keep="first")
    after = combined.height
    if before != after:
        log.warning("Removed %s duplicate MVT_ID rows (%s -> %s)", f"{before - after:,}",
                    f"{before:,}", f"{after:,}")

    # Sort by time for deterministic ordering
    combined = combined.sort("MVT_TIME_UTC_mvt")

    log.info("Total training DEP: %s rows", f"{combined.height:,}")
    return combined


def load_ranking() -> pl.DataFrame:
    """Load ranking.parquet, filtered to PHASE=DEP.

    Returns:
        Polars DataFrame with ranking DEP rows, sorted by MVT_TIME_UTC_mvt.
    """
    path = os.path.join(DATA_RAW, "ranking.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ranking parquet not found: {path}")

    lf = (
        pl.scan_parquet(path)
        .filter(pl.col("PHASE_mvt") == "DEP")
        .sort("MVT_TIME_UTC_mvt")
    )
    df = lf.collect()
    log.info("Ranking DEP: %s rows", f"{df.height:,}")
    return df


def load_training_movements(filter_months: list[str] | None = None) -> pl.DataFrame:
    """Load training parquet files with BOTH DEP and ARR phases for congestion features.

    Unlike load_training() which filters to DEP only, this returns all movements
    (DEP + ARR) needed for computing airport-level congestion counts. Rows are
    deduplicated by MVT_ID_mvt and sorted by MVT_TIME_UTC_mvt.

    Args:
        filter_months: Optional month prefixes to load a subset.

    Returns:
        Polars DataFrame with all phase rows, sorted by MVT_TIME_UTC_mvt.
    """
    pattern = os.path.join(DATA_RAW, "training_*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No training parquet files found matching {pattern}")

    if filter_months:
        files = [f for f in files if any(m in os.path.basename(f) for m in filter_months)]
        log.info("Loading %d filtered training files (all phases): %s", len(files),
                 [os.path.basename(f) for f in files])
    else:
        log.info("Loading %d training files (all phases)", len(files))

    frames: list[pl.DataFrame] = []
    for f in files:
        # No PHASE filter — keep both DEP and ARR
        df = pl.read_parquet(f)
        month = os.path.basename(f).split("_")[1][:7]
        log.info("  %s: %s rows (DEP+ARR)", month, f"{df.height:,}")
        frames.append(df)

    combined = pl.concat(frames)

    before = combined.height
    combined = combined.unique(subset=["MVT_ID_mvt"], keep="first")
    after = combined.height
    if before != after:
        log.warning("Removed %s duplicate MVT_ID rows (%s -> %s)", f"{before - after:,}",
                    f"{before:,}", f"{after:,}")

    combined = combined.sort("MVT_TIME_UTC_mvt")
    log.info("Total training movements: %s rows", f"{combined.height:,}")
    return combined


def load_ranking_movements() -> pl.DataFrame:
    """Load ranking.parquet with BOTH DEP and ARR phases for congestion features.

    Returns:
        Polars DataFrame with all ranking movements, sorted by MVT_TIME_UTC_mvt.
    """
    path = os.path.join(DATA_RAW, "ranking.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ranking parquet not found: {path}")

    df = pl.read_parquet(path).sort("MVT_TIME_UTC_mvt")
    log.info("Ranking movements: %s rows (DEP+ARR)", f"{df.height:,}")
    return df
