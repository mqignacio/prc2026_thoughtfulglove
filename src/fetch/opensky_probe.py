"""OpenSky Trino external-auth probe (Task 3.5, Goal 1).

Establishes a headless-friendly Trino session against trino.opensky-network.org:

1. Sends an unauthenticated statement -> captures the WWW-Authenticate header,
   which carries ``x_redirect_server`` (URL a human opens in any browser) and
   ``x_token_server`` (URL this process long-polls for the token).
2. Prints the redirect URL to stdout and to ``/tmp/opensky_auth_url`` so the
   agent can relay it to the human user (the Mac Mini is headless).
3. Long-polls the token server until the browser login completes, then runs the
   probe queries with a Bearer token.
4. Captures per-query ``stats`` (processedRows / processedBytes / elapsed) so the
   full-pull size projection is measured, not estimated.

Usage:
    uv run --with requests python src/fetch/opensky_probe.py

Nothing here is committed as a credential; the token is written to
``/tmp/opensky_token`` (mode 600) for reuse within the session.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import requests

HOST = "https://trino.opensky-network.org"
USER = "mqignacio"  # OpenSky username, lowercase (documented in resources/)
CATALOG = "minio"
SCHEMA = "osky"
TOKEN_PATH = Path("/tmp/opensky_token")
URL_PATH = Path("/tmp/opensky_auth_url")
POLL_TIMEOUT_SEC = 900

# Reporting-airport coordinates (src/pdc26/airports.py), +/- 0.25 deg bbox.
AIRPORTS = {
    "EDDF": (50.0264, 8.5431), "EDDM": (48.3538, 11.7861), "EGLL": (51.4706, -0.4619),
    "EHAM": (52.3086, 4.7639), "LEBL": (41.2974, 2.0833), "LEMD": (40.4722, -3.5608),
    "LFPG": (49.0128, 2.5500), "LIRF": (41.8045, 12.2508), "LTFM": (41.2611, 28.7419),
    "LSZH": (47.4647, 8.5492),
}
HALF = 0.25
# One busy training hour: 2025-07-15 08:00 UTC.
PROBE_HOUR = 1752566400


def log(msg: str) -> None:
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def headers(token: str | None = None) -> dict[str, str]:
    h = {"X-Trino-User": USER, "X-Trino-Catalog": CATALOG, "X-Trino-Schema": SCHEMA}
    if token:
        h["Authorization"] = "Bearer " + token
    return h


def initiate_flow() -> tuple[str, str]:
    r = requests.post(HOST + "/v1/statement", data="SELECT 1", headers=headers(), timeout=30)
    if r.status_code != 401:
        raise RuntimeError("expected 401 handshake, got %s: %s" % (r.status_code, r.text[:200]))
    auth = r.headers.get("WWW-Authenticate", "")
    redir = re.search(r'x_redirect_server="([^"]+)"', auth)
    tok = re.search(r'x_token_server="([^"]+)"', auth)
    if not redir or not tok:
        raise RuntimeError("no x_redirect_server/x_token_server in: %s" % auth)
    return redir.group(1), tok.group(1)


def wait_for_token(token_server: str) -> str:
    t0 = time.time()
    url = token_server
    while time.time() - t0 < POLL_TIMEOUT_SEC:
        r = requests.get(url, timeout=90)
        d = r.json()
        if d.get("token"):
            token = d["token"]
            TOKEN_PATH.write_text(token)
            TOKEN_PATH.chmod(0o600)
            log("TOKEN acquired after %.1fs (len=%d)" % (time.time() - t0, len(token)))
            return token
        url = d.get("nextUri", url)
        log("waiting for browser login... (%.0fs elapsed)" % (time.time() - t0))
    raise TimeoutError("browser login not completed within %ds" % POLL_TIMEOUT_SEC)


def run_query(sql: str, token: str, label: str) -> dict:
    started = time.time()
    r = requests.post(HOST + "/v1/statement", data=sql, headers=headers(token), timeout=1800)
    if r.status_code >= 400:
        raise RuntimeError("%s -> HTTP %s: %s" % (label, r.status_code, r.text[:400]))
    payload = r.json()
    rows: list = []
    next_uri = payload.get("nextUri")
    stats = payload.get("stats", {})
    while next_uri:
        if payload.get("data"):
            rows.extend(payload["data"])
        r = requests.get(next_uri, headers=headers(token), timeout=1800)
        payload = r.json()
        next_uri = payload.get("nextUri")
        stats = payload.get("stats", stats)
    if payload.get("data"):
        rows.extend(payload["data"])
    elapsed = time.time() - started
    log("%s: rows=%d elapsed=%.1fs processedRows=%s processedBytes=%s"
        % (label, len(rows), elapsed, stats.get("processedRows"),
           stats.get("processedBytes")))
    return {"label": label, "sql": sql, "rows": rows, "elapsed_sec": elapsed,
            "stats": stats, "http_status": r.status_code}


def main() -> int:
    redir, token_server = initiate_flow()
    URL_PATH.write_text(redir)
    log("REDIRECT URL (open in browser, log in as %s):\n%s" % (USER, redir))
    token = wait_for_token(token_server)

    results = []
    results.append(run_query("SHOW TABLES", token, "SHOW TABLES"))

    lat, lon = AIRPORTS["EGLL"]
    q1 = ("SELECT COUNT(*) AS n, COUNT(DISTINCT icao24) AS ac, "
          "SUM(CASE WHEN onground THEN 1 ELSE 0 END) AS n_ground, "
          "COUNT(DISTINCT CASE WHEN onground THEN icao24 END) AS ac_ground "
          "FROM state_vectors_data4 "
          "WHERE hour = %d AND lat BETWEEN %.4f AND %.4f AND lon BETWEEN %.4f AND %.4f "
          "AND time - lastcontact <= 15" % (PROBE_HOUR, lat - HALF, lat + HALF, lon - HALF, lon + HALF))
    results.append(run_query(q1, token, "Q1_EGLL_1h_bbox"))

    cases = "\n".join(
        "      WHEN lat BETWEEN %.4f AND %.4f AND lon BETWEEN %.4f AND %.4f THEN '%s'"
        % (la - HALF, la + HALF, lo - HALF, lo + HALF, code)
        for code, (la, lo) in sorted(AIRPORTS.items()))
    lats = [v[0] for v in AIRPORTS.values()]
    lons = [v[1] for v in AIRPORTS.values()]
    q2 = ("SELECT airport, COUNT(*) AS n, COUNT(DISTINCT icao24) AS ac, "
          "SUM(CASE WHEN onground THEN 1 ELSE 0 END) AS n_ground "
          "FROM (SELECT icao24, onground, CASE\n%s\n    END AS airport "
          "FROM state_vectors_data4 "
          "WHERE hour = %d AND lat BETWEEN %.4f AND %.4f AND lon BETWEEN %.4f AND %.4f "
          "AND time - lastcontact <= 15) "
          "WHERE airport IS NOT NULL GROUP BY airport ORDER BY airport"
          % (cases, PROBE_HOUR, min(lats) - HALF, max(lats) + HALF, min(lons) - HALF, max(lons) + HALF))
    results.append(run_query(q2, token, "Q2_10airports_1h_enclosing_bbox"))

    summary = {
        "probe_hour_unix": PROBE_HOUR,
        "queries": [{"label": x["label"], "rows": len(x["rows"]), "elapsed_sec": round(x["elapsed_sec"], 1),
                     "processedRows": x["stats"].get("processedRows"),
                     "processedBytes": x["stats"].get("processedBytes"),
                     "first_rows": x["rows"][:12]} for x in results],
    }
    Path("/tmp/opensky_probe_summary.json").write_text(json.dumps(summary, indent=2))
    log("SUMMARY:\n" + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
