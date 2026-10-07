"""Build holiday feature table for 8 countries covering the 10 PRC-2026 airports.

Uses the ``holidays`` Python package to generate public-holiday flags per
(airport, date). Writes ``data/processed/holiday_features.parquet``.

Airport → country mapping (locked):
    EDDF/EDDM → DE  ·  EGLL → GB (verified: ``holidays`` 0.104 uses "GB")
    EHAM → NL  ·  LEBL/LEMD → ES  ·  LFPG → FR
    LIRF → IT  ·  LTFM → TR  ·  LSZH → CH

Usage:
    uv run python src/fetch/fetch_holidays.py
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date

import holidays as holidays_lib
import pandas as pd

log = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")
HOLIDAY_PARQUET = os.path.join(PROCESSED_DIR, "holiday_features.parquet")

# Airport → ISO country code mapping
AIRPORT_COUNTRY: dict[str, str] = {
    "EDDF": "DE",
    "EDDM": "DE",
    "EGLL": "GB",
    "EHAM": "NL",
    "LEBL": "ES",
    "LEMD": "ES",
    "LFPG": "FR",
    "LIRF": "IT",
    "LTFM": "TR",
    "LSZH": "CH",
}

START_DATE = date(2025, 1, 1)
END_DATE = date(2026, 8, 1)  # Exclusive end

# ISO country codes used
COUNTRIES = sorted(set(AIRPORT_COUNTRY.values()))


def build_holiday_table() -> pd.DataFrame:
    """Build a DataFrame with one row per (airport, date) and holiday flags.

    Returns:
        DataFrame with columns: airport, date, country, is_holiday,
        holiday_name, is_day_before, is_day_after.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Verify that 'GB' works (the task asks to pin this)
    try:
        test = holidays_lib.country_holidays("GB", years=2025)
        log.info("holidays package: GB country code verified (version=%s)",
                 holidays_lib.__version__)
    except KeyError:
        log.error("holidays package does not support 'GB'. Try 'UK' or check installed version.")
        sys.exit(1)

    records: list[dict] = []
    date_range = pd.date_range(START_DATE, END_DATE - pd.Timedelta(days=1), freq="D")

    for country in COUNTRIES:
        try:
            cal = holidays_lib.country_holidays(country, years=[2025, 2026])
        except KeyError:
            log.warning("  Country %s not supported — skipping", country)
            continue

        log.info("  Building holidays for %s (%s sub-divisions)", country, len(cal.subdivisions or []))

        for dt in date_range:
            is_hol = dt in cal
            name = cal.get(dt) if is_hol else None

            rec = {
                "date": dt.date(),
                "country": country,
                "is_public_holiday": is_hol,
                "holiday_name": name if is_hol else None,
            }
            records.append(rec)

    if not records:
        log.error("No holiday records generated — check country codes.")
        return pd.DataFrame()

    # Build country-level df
    country_df = pd.DataFrame(records)
    country_df["date"] = pd.to_datetime(country_df["date"])

    # Add day-before / day-after flags
    country_df["is_day_before_holiday"] = (
        country_df.groupby("country")["is_public_holiday"]
        .shift(-1, fill_value=False)
    )
    country_df["is_day_after_holiday"] = (
        country_df.groupby("country")["is_public_holiday"]
        .shift(1, fill_value=False)
    )

    # Expand to per-airport rows
    # Build a map: country → list of airports
    country_to_airports: dict[str, list[str]] = {}
    for apt, cc in AIRPORT_COUNTRY.items():
        country_to_airports.setdefault(cc, []).append(apt)

    airport_rows: list[dict] = []
    for country, airports in country_to_airports.items():
        sub = country_df[country_df["country"] == country].copy()
        for apt in airports:
            apt_sub = sub.copy()
            apt_sub["airport"] = apt
            airport_rows.append(apt_sub)

    full = pd.concat(airport_rows, ignore_index=True)
    full = full.sort_values(["airport", "date"]).reset_index(drop=True)

    # Column order
    cols = [
        "airport", "date", "country",
        "is_public_holiday", "holiday_name",
        "is_day_before_holiday", "is_day_after_holiday",
    ]
    full = full[cols]

    return full


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("=" * 60)
    log.info("Holiday feature table: %d countries, %s → %s",
             len(COUNTRIES), START_DATE, END_DATE)
    log.info("Airports: %s", AIRPORT_COUNTRY)
    log.info("=" * 60)

    df = build_holiday_table()

    log.info("Writing %s rows to %s", f"{len(df):,}", HOLIDAY_PARQUET)
    df.to_parquet(HOLIDAY_PARQUET, index=False)

    # Quick summary
    print(f"\nHoliday features summary:")
    print(f"  Airports: {df['airport'].nunique()} — {sorted(df['airport'].unique())}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Total rows: {len(df):,}")
    print(f"  Holidays per country:")
    cnt = df[df["is_public_holiday"]].groupby("country")["date"].nunique()
    for cc, n in cnt.items():
        print(f"    {cc}: {n} holiday dates")

    # Verify country code
    log.info("Pin note: using ISO country code 'GB' for UK airports (EGLL). "
             "holidays %s supports 'GB'.", holidays_lib.__version__)


if __name__ == "__main__":
    main()