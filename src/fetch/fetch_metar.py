"""Fetch METAR observations from the Iowa State Mesonet archive.

Downloads hourly/special METAR observations for the 10 PRC-2026 airports
from 2025-01-01 through 2026-08-01, saves raw CSVs, and writes a cleaned
parquet table to data/processed/metar_features.parquet.

Usage:
    uv run python src/fetch/fetch_metar.py [--station EDDF] [--force]

API: GET https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py
    ?station=<ICAO>&data=all&year1=&month1=&day1=&year2=&month2=&day2=
    &tz=UTC&format=comma

Field documentation:
    https://mesonet.agron.iastate.edu/request/download.phtml
"""

from __future__ import annotations

import csv
import io
import logging
import os
import sys
import time
from datetime import date, timedelta

import pandas as pd
import requests

log = logging.getLogger(__name__)

# Project root relative to this file's location (src/fetch/fetch_metar.py)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(_PROJECT_ROOT, "data", "raw", "metar")
PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")
METAR_PARQUET = os.path.join(PROCESSED_DIR, "metar_features.parquet")

AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LTFM", "LSZH"]

START_DATE = date(2025, 1, 1)
END_DATE = date(2026, 8, 1)  # Exclusive end

API_BASE = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# Max retries per month-station download
MAX_RETRIES = 3
# Sleep between retries (seconds)
RETRY_DELAY = 10
# Sleep between stations to be polite
STATION_DELAY = 2.0

# Variables we request (data=all returns these)
EXPECTED_HEADER = [
    "station", "valid", "tmpf", "dwpf", "relh", "drct", "sknt", "p01i",
    "alti", "mslp", "vsby", "gust", "skyc1", "skyc2", "skyc3", "skyc4",
    "skyl1", "skyl2", "skyl3", "skyl4", "wxcodes", "ice_accretion_1hr",
    "ice_accretion_3hr", "ice_accretion_6hr", "peak_wind_gust",
    "peak_wind_drct", "peak_wind_time", "feel", "metar", "snowdepth",
]


# ── helpers ──────────────────────────────────────────────────────────────

def _month_range(y: int, m: int) -> tuple[date, date]:
    """Return (first_day, first_day_of_next_month)."""
    start = date(y, m, 1)
    if m == 12:
        end = date(y + 1, 1, 1)
    else:
        end = date(y, m + 1, 1)
    return start, end


def _iter_months(start: date, end: date):
    """Yield (year, month) tuples from start (inclusive) to end (exclusive)."""
    cursor = start
    while cursor < end:
        yield cursor.year, cursor.month
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)


def _download_month(icao: str, y: int, m: int) -> str | None:
    """Download one month of METAR data for *icao* as raw CSV text.

    Returns None if all retries fail.
    """
    s, e = _month_range(y, m)
    params = {
        "station": icao,
        "data": "all",
        "year1": str(y),
        "month1": str(m),
        "day1": "1",
        "year2": str(e.year),
        "month2": str(e.month),
        "day2": "1",
        "tz": "UTC",
        "format": "comma",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(API_BASE, params=params, timeout=(15, 180))
            resp.raise_for_status()
            text = resp.text
            # Confirm we got non-debug data lines
            data_lines = [ln for ln in text.split("\n") if ln and not ln.startswith("#")]
            if len(data_lines) <= 1:  # header only
                log.warning("  %s %04d-%02d: empty response (attempt %d)", icao, y, m, attempt)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                    continue
                return None
            return text
        except requests.RequestException as exc:
            log.warning("  %s %04d-%02d: %s (attempt %d/%d)", icao, y, m, exc, attempt, MAX_RETRIES)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    return None


def _parse_value(val: str) -> float | None:
    """Convert 'M' or empty to None, else float."""
    if val is None:
        return None
    v = val.strip()
    if v in ("", "M", "null"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


# ── flight category ──────────────────────────────────────────────────────

def _ceiling_ft(skyc: list[str | None], skyl: list[float | None]) -> float | None:
    """Derive ceiling height (feet, lowest BKN/OVC layer).

    Returns None if no broken or overcast layer exists.
    """
    cover_to_ceil = {"BKN", "OVC", "VV"}
    for cover, height in zip(skyc, skyl):
        if cover and height is not None and cover.strip() in cover_to_ceil:
            return height
    return None


def _flight_category(vsby_miles: float | None, ceiling_ft: float | None) -> str | None:
    """Classify into LIFR / IFR / MVFR / VFR.

    Uses aviation-standard thresholds (worst of visibility and ceiling).
    """
    if vsby_miles is None and ceiling_ft is None:
        return None

    # Default to VFR — only downgrade when thresholds are met
    cat = "VFR"

    if ceiling_ft is not None:
        if ceiling_ft < 500:
            cat = "LIFR"
        elif ceiling_ft < 1000 and cat != "LIFR":
            cat = "IFR"
        elif ceiling_ft <= 3000 and cat not in ("LIFR", "IFR"):
            cat = "MVFR"

    if vsby_miles is not None:
        if vsby_miles < 1.0:
            cat = "LIFR"
        elif vsby_miles < 3.0 and cat != "LIFR":
            cat = "IFR"
        elif vsby_miles <= 5.0 and cat not in ("LIFR", "IFR"):
            cat = "MVFR"

    return cat


# ── download all ─────────────────────────────────────────────────────────

def download_all(force: bool = False) -> list[dict]:
    """Download raw CSVs for all airports and months, return list of row dicts.

    Skips months where the raw file already exists unless *force* is True.
    Saves each download to ``data/raw/metar/{icao}/{icao}_{yyyymm}.csv``.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    total_months = sum(1 for _ in _iter_months(START_DATE, END_DATE))
    log.info("Downloading METAR for %d airports × %d months = %d requests",
             len(AIRPORTS), total_months, len(AIRPORTS) * total_months)

    rows: list[dict] = []
    done = 0
    failed = 0
    skipped = 0

    for icao in AIRPORTS:
        station_dir = os.path.join(RAW_DIR, icao)
        os.makedirs(station_dir, exist_ok=True)
        station_rows: list[dict] = []
        station_ok = True

        for y, m in _iter_months(START_DATE, END_DATE):
            fname = f"{icao}_{y}{m:02d}.csv"
            fpath = os.path.join(station_dir, fname)

            if os.path.exists(fpath) and not force:
                skipped += 1
                # Read from cached file to accumulate rows
                with open(fpath, "r") as fh:
                    text = fh.read()
                parsed = _parse_csv(text)
                station_rows.extend(parsed)
                continue

            raw = _download_month(icao, y, m)
            if raw is None:
                log.error("  FAILED %s %04d-%02d after %d retries", icao, y, m, MAX_RETRIES)
                failed += 1
                station_ok = False
                continue

            # Save raw CSV
            with open(fpath, "w") as fh:
                fh.write(raw)

            parsed = _parse_csv(raw)
            station_rows.extend(parsed)
            done += 1

            log.info("  %s %04d-%02d: %d obs %s", icao, y, m, len(parsed),
                     "(new)" if not os.path.exists(fpath) else "(overwrite)")

            # Polite pause between months
            time.sleep(0.3)

        log.info("  %s: %d total observations%s", icao, len(station_rows),
                 " — some months FAILED" if not station_ok else "")
        rows.extend(station_rows)

        # Polite pause between stations
        time.sleep(STATION_DELAY)

    log.info("METAR download complete: %d new, %d skipped, %d failed", done, skipped, failed)
    return rows


# ── parse CSV text into dicts ──────────────────────────────────────────

def _parse_csv(text: str) -> list[dict]:
    """Parse a raw CSV response into a list of dicts with typed values.

    Skips debug lines (starting with '#') and the header row.
    """
    lines = [ln for ln in text.split("\n") if ln and not ln.startswith("#")]
    if len(lines) < 2:
        return []

    reader = csv.reader(io.StringIO("\n".join(lines)))
    header = [h.strip() for h in next(reader)]

    expected = {h: i for i, h in enumerate(EXPECTED_HEADER)}
    if not all(col in header for col in EXPECTED_HEADER):
        log.warning("Unexpected header (got %d cols, expected %d): %s",
                    len(header), len(EXPECTED_HEADER), header)
        return []

    rows: list[dict] = []
    for row in reader:
        if not row or len(row) != len(header):
            continue
        valid = row[expected["valid"]].strip()
        if not valid:
            continue

        tmpf = _parse_value(row[expected["tmpf"]])
        dwpf = _parse_value(row[expected["dwpf"]])
        sknt = _parse_value(row[expected["sknt"]])
        drct = _parse_value(row[expected["drct"]])
        gust = _parse_value(row[expected["gust"]])
        vsby = _parse_value(row[expected["vsby"]])
        p01i = _parse_value(row[expected["p01i"]])
        relh = _parse_value(row[expected["relh"]])

        # Sky cover layers
        skyc = [
            row[expected["skyc1"]].strip() if expected["skyc1"] < len(row) and row[expected["skyc1"]].strip() not in ("", "M") else None,
            row[expected["skyc2"]].strip() if expected["skyc2"] < len(row) and row[expected["skyc2"]].strip() not in ("", "M") else None,
            row[expected["skyc3"]].strip() if expected["skyc3"] < len(row) and row[expected["skyc3"]].strip() not in ("", "M") else None,
            row[expected["skyc4"]].strip() if expected["skyc4"] < len(row) and row[expected["skyc4"]].strip() not in ("", "M") else None,
        ]
        skyl = [
            _parse_value(row[expected["skyl1"]]),
            _parse_value(row[expected["skyl2"]]),
            _parse_value(row[expected["skyl3"]]),
            _parse_value(row[expected["skyl4"]]),
        ]

        ceiling = _ceiling_ft(skyc, skyl)
        flight_cat = _flight_category(vsby, ceiling)

        raw_metar = row[expected["metar"]].strip() if expected["metar"] < len(row) else ""
        wxcodes = row[expected["wxcodes"]].strip() if expected["wxcodes"] < len(row) else ""
        snowdepth = _parse_value(row[expected["snowdepth"]]) if expected["snowdepth"] < len(row) else None
        feel = _parse_value(row[expected["feel"]])

        d = {
            "station": row[expected["station"]].strip(),
            "valid": pd.Timestamp(valid, tz="UTC"),
            # Temperature (F → C)
            "tmpc": (tmpf - 32.0) * 5.0 / 9.0 if tmpf is not None else None,
            "dwpc": (dwpf - 32.0) * 5.0 / 9.0 if dwpf is not None else None,
            "relh": relh,
            # Wind
            "wind_dir": drct,
            "wind_kt": sknt,
            "wind_gust_kt": gust,
            # Visibility (miles)
            "vis_miles": vsby,
            # Pressure
            "alti_in": _parse_value(row[expected["alti"]]),
            "mslp_mb": _parse_value(row[expected["mslp"]]),
            # Precipitation (inches)
            "precip_1hr_in": p01i,
            # Sky
            "ceiling_ft": ceiling,
            "flight_category": flight_cat,
            # Weather codes
            "wxcodes": wxcodes,
            # Snow depth
            "snowdepth_in": snowdepth,
            # Apparent temp
            "feel_f": feel,
            # Raw METAR string
            "metar_raw": raw_metar,
        }
        rows.append(d)
    return rows


# ── build parquet ─────────────────────────────────────────────────────────

def build_parquet(rows: list[dict]) -> None:
    """Convert parsed rows to a DataFrame, sort, deduplicate, write parquet."""
    if not rows:
        log.warning("No rows to build parquet from.")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["station", "valid"]).reset_index(drop=True)

    # Drop exact duplicates (same station, same timestamp)
    before = len(df)
    df = df.drop_duplicates(subset=["station", "valid"])
    after = len(df)
    if before != after:
        log.info("Removed %d duplicate (station, valid) rows", before - after)

    log.info("Writing %s rows to %s", f"{len(df):,}", METAR_PARQUET)
    df.to_parquet(METAR_PARQUET, index=False)

    # Quick summary
    print(f"\nMETAR features summary:")
    print(f"  Stations: {df['station'].nunique()} — {sorted(df['station'].unique())}")
    print(f"  Date range: {df['valid'].min()} to {df['valid'].max()}")
    print(f"  Total rows: {len(df):,}")
    print(f"  Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"  Flight-category distribution:")
    print(f"    {df['flight_category'].value_counts().to_string()}")


# ── CLI entry point ──────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    force = "--force" in sys.argv

    log.info("=" * 60)
    log.info("METAR fetch: %s → %s, %d airports", START_DATE, END_DATE, len(AIRPORTS))
    if force:
        log.info("FORCE mode: will re-download existing files")
    log.info("=" * 60)

    t0 = time.time()
    rows = download_all(force=force)
    t1 = time.time()
    log.info("Download phase: %.1f s for %d rows", t1 - t0, len(rows))

    build_parquet(rows)
    log.info("Total time: %.1f s", time.time() - t0)
    log.info("Done — METAR features written to %s", METAR_PARQUET)


if __name__ == "__main__":
    main()