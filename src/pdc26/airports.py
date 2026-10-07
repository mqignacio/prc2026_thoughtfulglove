"""Airport reference data: timezones and coordinates for PRC-OpenSky 2026.

The 10 reporting airports are hardcoded (public knowledge) per task constraint.
Destination-airport coordinates for the great-circle distance feature are looked
up from the public `airportsdata` package (OpenFlights-derived, MIT-licensed),
falling back to the hardcoded table for any unmatched ICAO code.

Coordinates are rounded and match airportsdata's reported values.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

# IANA/zoneinfo timezone per reporting airport (DST-aware local time).
AIRPORT_TZ: dict[str, str] = {
    "EDDF": "Europe/Berlin",
    "EDDM": "Europe/Berlin",
    "EGLL": "Europe/London",
    "EHAM": "Europe/Amsterdam",
    "LEBL": "Europe/Madrid",
    "LEMD": "Europe/Madrid",
    "LFPG": "Europe/Paris",
    "LIRF": "Europe/Rome",
    "LTFM": "Europe/Istanbul",
    "LSZH": "Europe/Zurich",
}

# Hardcoded lat/lon (public knowledge) for the 10 reporting airports.
AIRPORT_COORDS: dict[str, dict[str, float]] = {
    "EDDF": {"lat": 50.0264, "lon": 8.5431},   # Frankfurt am Main
    "EDDM": {"lat": 48.3538, "lon": 11.7861},  # Munich
    "EGLL": {"lat": 51.4706, "lon": -0.4619},  # London Heathrow
    "EHAM": {"lat": 52.3086, "lon": 4.7639},   # Amsterdam Schiphol
    "LEBL": {"lat": 41.2974, "lon": 2.0833},   # Barcelona El Prat
    "LEMD": {"lat": 40.4722, "lon": -3.5608},  # Madrid Barajas
    "LFPG": {"lat": 49.0128, "lon": 2.5500},   # Paris Charles de Gaulle
    "LIRF": {"lat": 41.8045, "lon": 12.2508},  # Rome Fiumicino
    "LTFM": {"lat": 41.2611, "lon": 28.7419},  # Istanbul
    "LSZH": {"lat": 47.4647, "lon": 8.5492},   # Zurich
}

# All airports we know coordinates for (10 hardcoded reporting airports).
REPORTING_AIRPORTS: list[str] = sorted(AIRPORT_COORDS)


def load_airportsdata_coords() -> dict[str, dict[str, float]]:
    """Load ICAO -> {lat, lon, tz} from the `airportsdata` package.

    Returns an empty dict if the package is unavailable (graceful degradation).
    """
    try:
        import airportsdata  # type: ignore
    except ImportError:
        log.warning("airportsdata not installed; destination coords limited to hardcoded 10")
        return {}
    return airportsdata.load("ICAO")  # keyed by ICAO code


def load_coords() -> dict[str, dict[str, float]]:
    """Build ICAO -> {'lat', 'lon'} lookup for every destination of interest.

    Starts from the hardcoded 10-airport table, then merges the `airportsdata`
    package for full destination coverage. When both exist, the hardcoded value
    wins (it is the task-spec source of truth for the reporting airports).

    Returns:
        Dict mapping ICAO string -> {'lat': float, 'lon': float}.
    """
    coords: dict[str, dict[str, float]] = {
        icao: {"lat": c["lat"], "lon": c["lon"]}
        for icao, c in AIRPORT_COORDS.items()
    }

    extra = load_airportsdata_coords()
    for icao, rec in extra.items():
        # Only add airports not already in the hardcoded table; keep only
        # entries with valid lat/lon (some codes lack coordinates).
        if icao in coords:
            continue
        lat = rec.get("lat")
        lon = rec.get("lon")
        if lat is None or lon is None:
            continue
        try:
            coords[icao] = {"lat": float(lat), "lon": float(lon)}
        except (TypeError, ValueError):
            continue

    log.info("Coordinate lookup ready: %d airports (%d hardcoded + %d from airportsdata)",
             len(coords), len(AIRPORT_COORDS), len(coords) - len(AIRPORT_COORDS))
    return coords


def coordinates_series(airport_codes: pd.Series, coords: dict) -> pd.DataFrame:
    """Map a Series of ICAO codes to a DataFrame with lat/lon columns.

    Unknown codes yield NaN lat/lon (null-safe for the distance feature).
    """
    lat = airport_codes.map(lambda a: coords.get(a, {}).get("lat", float("nan")))
    lon = airport_codes.map(lambda a: coords.get(a, {}).get("lon", float("nan")))
    return pd.DataFrame({"lat": lat.astype("float64"), "lon": lon.astype("float64")})
