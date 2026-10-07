# OpenSky Trino Feasibility Probe (Task 3.5, Goal 1)

**Date:** probe 2026-10-03; final status 2026-10-07 (PST) · **Machine:** Mac Mini M4 Pro (`100.82.213.82`) · **Account:** OpenSky `mqignacio`
**Task file:** `tasks/12-opensky_enrichment.md`

## Verdict: **BLOCKED** (probe) → **CANCELLED for the challenge** (final) — account authenticates, but has no Trino Data Access grant, and OpenSky will not release the data during the challenge

OAuth2 external authentication **succeeded** (the browser login flow completed and the Trino server issued a bearer token), but **every** query is rejected at the authorization layer:

```
errorCode: 4, errorName: PERMISSION_DENIED, errorType: USER_ERROR
message: "Access Denied: Cannot execute query"
```

The denial is **blanket** — it applies to catalog-free statements (`SELECT 1`) and metadata statements (`SHOW CATALOGS`, `SHOW SCHEMAS`, `SHOW TABLES`, `DESCRIBE`) as well as to the intended partition-filtered aggregate query. This is the signature of a missing **Trino Data Access** grant (My OpenSky → Request Data Access, application-reviewed), not of a malformed query, a missing partition filter, or a credential mismatch.

Per `tasks/12-opensky_enrichment.md`: *"Confirm the OpenSky account has Trino Data Access approved. If not → STOP → mark Task 3.5 BLOCKED."* That branch is taken. **No full-pull size/scan projection is measurable** — authorization is denied before query planning, so all query stats are zero. Goal 2 (pull + features + ablation) was not started, as instructed.

## Final status (2026-10-07): CANCELLED for the challenge

Following the probe, the human user submitted the Data Access request; OpenSky replied — verbatim — **"You can have data after the challenge ends - it is not available for the challenge."** Trino historical data is therefore **not available during the competition window**. Task 3.5 is **CANCELLED for the challenge** — a definitive decision, not a pending application. The dataset becomes available post-challenge and is worth revisiting as material for the JOAS paper (Phase 5). The BLOCKED verdict below remains the technical finding; the outcome is now final.

## Evidence

### Credentials relay (MacBook → Mini), one-time
- Received 85 bytes; SHA256 `19adcdc2f070355dacd39cfa171c55489f8e82e7cf7e0796c115e687cc6693f4`.
- Installed as `resources/credentials.json`, mode `-rw-------` (600).
- `git check-ignore -v resources/credentials.json` → `.gitignore:176` (ignored). `git ls-files resources/credentials.json` → empty (untracked).
- Payload is the REST OAuth client (`clientId`, `clientSecret` key names only; values never printed). **The Trino path does not use this file** — it uses the OpenSky **account** browser login (`--external-authentication` equivalent). This confirms the task-file caveat.

### Trino handshake (headless-friendly)
`POST https://trino.opensky-network.org/v1/statement` with `X-Trino-User: mqignacio` (no token) returned:

```
HTTP 401
WWW-Authenticate: Basic realm="Trino", Bearer
  x_redirect_server="https://trino.opensky-network.org/oauth2/token/initiate/<redacted>",
  x_token_server="https://trino.opensky-network.org/oauth2/token/<uuid>"
```

The Mini polled `x_token_server` (each request long-polls ~10 s and returns `{"nextUri": ...}` until login completes). After the human user opened the redirect URL in the MacBook browser and logged in, the token was acquired **47 s** after the flow started (`len=3381`). The bearer token is **not** recorded here; it is cached only at `/tmp/opensky_token` (mode 600) for the session.

### Query results (all with the bearer token, catalog `minio`, schema `osky`)

| Query | State | Error | rows | processedRows | processedBytes |
|---|---|---|---|---|---|
| `SELECT 1 AS x` | FAILED | Access Denied: Cannot execute query | 0 | 0 | 0 |
| `SHOW CATALOGS` | FAILED | Access Denied: Cannot execute query | 0 | 0 | 0 |
| `SHOW SCHEMAS FROM minio` | FAILED | Access Denied: Cannot execute query | 0 | 0 | 0 |
| `SHOW TABLES` | FAILED | Access Denied: Cannot execute query | 0 | 0 | 0 |
| `DESCRIBE state_vectors_data4` | FAILED | Access Denied: Cannot execute query | 0 | 0 | 0 |
| Q1 — EGLL ±0.25° bbox, `hour = 1752566400` (2025-07-15 08:00 UTC), `time - lastcontact <= 15` | FAILED | Access Denied: Cannot execute query | 0 | 0 | 0 |

The HTTP status of each statement submission was **200** (the request is accepted and queued); the failure is returned in the follow-up `nextUri` payload as a Trino `error` object with `state: FAILED`.

### What could not be measured
- Connection latency to first byte of a *successful* query, rows returned, and per-query scan bytes — all gated behind the execute permission.
- Full-pull sizing. For context only, the actually-needed windows span **14 months** (2025-01..2025-12 + 2026-01 + 2026-07) = **427 days = 10,248 hour-partitions** = **102,480 airport-hour cells** at 10 airports. None of this can be costed until access is granted.

## Reproduce

```bash
cd /Users/emkyu-agent/opensky_prc_2026
uv run --with requests python src/fetch/opensky_probe.py   # prints x_redirect_server, polls token, runs probes
```

## Credentials hygiene
- `resources/credentials.json`: gitignored (`.gitignore:176`), untracked, mode 600, valid JSON.
- Bearer token: session-only at `/tmp/opensky_token` (mode 600); never written to the repo.
- Tracked-file credential scan (keywords + long hex runs, file:line + boolean only): clean — one prose mention in `src/reports/data_verification.md` and SHA256 data-hash columns in `src/reports/error_analysis.md`; no secrets.

## Caveat (unverified)
This run assumes the browser login used the **same** account (`mqignacio`) as `X-Trino-User`. If a different OpenSky account was used, the denial could reflect that mismatch rather than a missing grant. The human user logged in during the flow and was instructed to use `mqignacio`; the blanket denial across `SELECT 1` is consistent with the far more likely cause — no Data Access grant.

## Recommendation
1. **Apply for Trino Data Access now** (human-only, website form): My OpenSky → Request Data Access, with an aviation-research justification (the PRC-2026 taxi-out study + the JOAS paper). Review is application-based and may not complete before the **Oct 9 feature/model freeze**, so treat access as *pending*, not imminent.
2. **Proceed to Task 4.2** (final submission) as the critical path; do not block on OpenSky. The prior is already low-yield: Task 3.1 congestion features captured actual-traffic density for Δ ≈ −0.10%, and 55.4% of blend squared error comes from metadata-missing special flights that ADS-B enrichment cannot address.
3. If access is granted before the freeze, re-run this probe (Goal 1) to obtain the latency/volume numbers, then decide on Goal 2. If after, record it as post-deadline material for the JOAS paper.
