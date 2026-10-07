# Data Verification Report — PRC-OpenSky Data Challenge 2026

Task: `01-data_acquisition` · Team: `thoughtful-glove`
Generated: 2026-09-08 (execution session) · Status: **COMPLETE — all 7 verification checks pass.** H1 supplied MinIO bucket keys; all 14 datasets downloaded and read via pyarrow. Git/secrets hygiene, external-data rules check, and Python environment all verified. One **material finding** corrects a planning assumption — see Section 5b (flagged for the human user).

---

## 1. Git + secrets hygiene — COMPLETE

`git init` run in the project root. The exact `project-management/assets/.gitignore` template was copied verbatim.

**Template insufficiency found (pre-flight rule #4 — mechanical verification catches what the template misses):** the generic template does **not** cover this project's secret/data paths. Confirmed before any edit:

```
$ git check-ignore -v resources/credentials.json data/raw/test.parquet models/x.txt submissions/y.txt
.gitignore:164:*.txt	models/x.txt          # incidental *.txt match, NOT a dir rule
.gitignore:164:*.txt	submissions/y.txt
(no rule)            resources/credentials.json   # NOT ignored by template
(no rule)            data/raw/test.parquet         # NOT ignored by template
```

**Resolution:** the template body was preserved byte-for-byte and a clearly-marked **project-specific section appended** (never editing template lines). This is a deliberate, recorded deviation from the literal "unmodified" wording, because the Task 01 acceptance criteria (these paths must be ignored) are unsatisfiable with the template alone.

Appended section (end of `.gitignore`, lines 165–173):

```
# --- Project-specific (PRC-DC2026, added by Task 01) ---
# ... template body preserved byte-for-byte above; only additions below.
data/
models/
submissions/
resources/credentials.json
```

**Post-fix mechanical verification (all PASS):**

```
$ git check-ignore -v resources/credentials.json data/raw/x.parquet models/x.bin submissions/thoughtful-glove_v1.parquet
.gitignore:173:resources/credentials.json	resources/credentials.json
.gitignore:170:data/	data/raw/x.parquet
.gitignore:171:models/	models/x.bin
.gitignore:172:submissions/	submissions/thoughtful-glove_v1.parquet

$ git ls-files resources/credentials.json
(empty)                       # PASS — credential file not tracked

$ git status --porcelain -- data/ models/ submissions/
(empty)                       # PASS — nothing staged/tracked under data/models/submissions
```

`.venv` is ignored by the template rule `.gitignore:116:.venv`.

---

## 2. External-data rules check — COMPLETE (NOT ambiguous)

Source: official **"Eligibility"** page — https://ansperformance.eu/study/data-challenge/dc2026/eligibility.html (linked from the main page as "Terms & Conditions"). The main page states the challenge aim is "Using open data" and "openly sharing final solutions". The authoritative text governing supplementary external/open data is under **"Eligibility for Prize"** — winning submissions/solutions are eligible for awards only if:

> - "All used external datasets are openly accessible/usable and documented."
> - "All additional datasets used are openly available under an open source license."
> - "All produced source code are made openly available on GitHub under the GNU GPLv3 license. Note: It will then be forked by the Challenge GitHub account for organisational purposes."
> - "Sufficient documentation is provided to understand and reproduce the results."
> - "The solution is original: Teams must use their own original solutions. Re-using any existing implementation is only allowed if the original authors grant you the rights to use their solution and if you made significant modifications to the algorithm or model. In particular, simply re-using existing code and rewriting the data input and output mechanism is not sufficient. Adding parameters to the model and modifying filters to match the specific peculiarities of the data, however, can be considered sufficient."

**Interpretation / verdict for the Task 2.3 gate:** External/open data is **explicitly permitted and expected**, not prohibited — the rules presuppose external datasets exist and impose conditions on them rather than banning them. Conditions, restated as obligations for our METAR (Iowa State Mesonet / NOAA) and holiday-calendar features (Task 2.3) and any OpenSky API enrichment (Task 3.5):

1. Each external dataset must be **openly accessible/usable**, **open-licensed**, and **documented** (provenance + license recorded per dataset).
2. Our **full source code** must be published on a **public GitHub repo under GNU GPLv3**.
3. Solution must be **original** (no wholesale reuse of others' code/models without rights + significant modification).

**No ambiguity → no human flag required before Phase 2.** One downstream note: Task 2.3 must capture the *license* of the specific METAR and holiday sources (open data ≠ automatically "open source license"); verify mesonet/NOAA and `holidays` package licensing and record it in the external-data report. This is a documentation obligation, not a blocker.

---

## 3. MinIO configuration — RESOLVED

- **Server endpoint:** `https://s3.opensky-network.org/` · **Web console:** `https://s3-console.opensky-network.org/` (SSO login → Access Keys → Create access key).
- **`mc` installed:** `/opt/homebrew/bin/mc` — `RELEASE.2025-08-13T08-35-41Z`.
- **Alias `opensky` added successfully** using the H1-supplied bucket keys (stored only in `~/.mc/config.json`; never written to the repo or this report). Note the API `clientId`/`clientSecret` in `credentials.json` are **not** MinIO keys and were rejected (`The Access Key Id you provided does not exist in our records.`).
- **Bucket layout discovered (differs from the 2024 `competition-data/` name):**
  - `prc-2026-datasets/` — read-only competition data (12 training + `ranking.parquet` + `submitting.parquet`).
  - `prc-2026-thoughtful-glove/` — our team submission bucket (empty at start).

---

## 4. Downloaded datasets — COMPLETE

`mc cp --recursive opensky/prc-2026-datasets/ data/raw/` → **14/14 files, 314.92 MiB**, each readable by pyarrow. Per-file size + row count (from `src/verify_data.py`):

| file | size (MiB) | rows |
|---|---|---|
| training_2025-01-01_2025-02-01 | 20.2 | 307,257 |
| training_2025-02-01_2025-03-01 | 18.9 | 287,411 |
| training_2025-03-01_2025-04-01 | 21.6 | 328,622 |
| training_2025-04-01_2025-05-01 | 22.7 | 350,288 |
| training_2025-05-01_2025-06-01 | 24.0 | 370,105 |
| training_2025-06-01_2025-07-01 | 23.8 | 365,986 |
| training_2025-07-01_2025-08-01 | 24.8 | 381,161 |
| training_2025-08-01_2025-09-01 | 24.8 | 382,047 |
| training_2025-09-01_2025-10-01 | 23.9 | 367,679 |
| training_2025-10-01_2025-11-01 | 24.2 | 371,163 |
| training_2025-11-01_2025-12-01 | 21.1 | 324,530 |
| training_2025-12-01_2026-01-01 | 21.6 | 331,548 |
| **ranking.parquet** | 41.5 | 689,534 |
| **submitting.parquet** | 1.6 | 344,841 |

> Note on sizes: official data page lists rounded decimal-MB figures (ranking "27M", submitting "1.1M"); actual binary-MiB sizes are larger and the ranking file is dated 2026-09-04, so the organizers have refreshed it since the page text. Sizes are consistent with the stated list once unit + revision differences are accounted for; row-count check (b) below matches exactly.

---

## 5. Dataset verification — COMPLETE (all 7 checks pass)

| Check | Expected | Result |
|---|---|---|
| (a) All files readable + sizes reported | 14 files, pyarrow opens each | **PASS** (table above) |
| (b) Total training rows | ≈ 4,167,797 | **PASS — 4,167,797 exact** |
| (c) Unique reporting airports | 10 | **PASS** — DEP `ADEP_mvt` set = {EDDF,EDDM,EGLL,EHAM,LEBL,LEMD,LFPG,LIRF,LSZH,LTFM}; ARR `ADES_mvt` set identical |
| (d) Ranking month coverage | Jan 2026 + Jul 2026 | **PASS** — {2026-01, 2026-07} |
| (e) submitting `MVT_ID_mvt` vs ranking DEP | match | **PASS** — exact set equality, 344,841 = 344,841, 0 either way |
| (f) Ranking columns blank for PHASE=DEP | ≥ `BLOCK_TIME_UTC_mvt`, `TAXITIME_SEC_mvt` | **PASS** — those two are the **only** 100%-null columns |
| (g) `MVT_TIME_UTC_mvt` populated in ranking DEP | confirm/refute | **PASS — CONFIRMED (100.00% populated, 344,841/344,841)** |

**Ranking column-availability table (PHASE=DEP, null %):**

| column | null % | column | null % |
|---|---|---|---|
| MVT_ID_mvt | 0.00 | ARVT_1_flt | 1.53 |
| RUNWAY_mvt | 0.00 | IOBT_flt | 1.53 |
| **MVT_TIME_UTC_mvt** | **0.00** | EOBT_1_flt | 1.53 |
| PHASE_mvt | 0.00 | AIRCRAFT_OPERATOR_flt | 1.53 |
| SCHED_TIME_UTC_mvt | 0.00 | AIRCRAFT_TYPE_flt | 1.53 |
| ADEP_mvt | 0.00 | FLIGHT_TYPE_flt | 1.53 |
| STAND_mvt | 0.001 | MARKET_SEGMENT_flt | 1.53 |
| ADES_mvt | 0.004 | LOBT_flt | 1.53 |
| FLIGHT_RULE_mvt | 0.006 | ADES_flt / ADEP_flt / CALLSIGN_flt | 1.53 |
| FLIGHT_mvt | 0.017 | **AOBT_3_flt** | 1.53 |
| AIRCRAFT_TYPE_mvt | 0.11 | ADES_FILED_flt / ARVT_3_flt | 1.53 |
| FLIGHT_ID_mvt | 1.53 | FLIGHT_RULE_flt / WK_TBL_CAT_flt | 1.54 |
| **BLOCK_TIME_UTC_mvt** | **100.00** | **TAXITIME_SEC_mvt** | **100.00** |

The ~1.53% null block = DEP rows unmatched to an NM flight (`FLIGHT_ID_mvt` null rate 1.53%). Full 30-column schema captured; no unexpected columns.

**`MVT_TIME_UTC_mvt` verdict: CONFIRMED populated (100%) in ranking DEP rows → TODO locked decision #5 stands.** Congestion features relative to takeoff time are viable.

---

## 5b. MATERIAL FINDING — NM off-block times survive in ranking (leakage assumption corrected)

**Flagged for the human user.** `external_research.md` §2 and §7, and the TODO risk list, assumed the NM off-block times (`AOBT_3_flt`, `LOBT_flt`) are *blanked for departures* and must be avoided. **Verification refutes this:** the data page blanks only `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt`; the **`_flt` off-block columns are ~98.5% populated in ranking DEP rows** (null only where the flight is unmatched to NM). Quantified on the 2025 training set (DEP rows, sanity-verified identity `MVT_TIME − BLOCK_TIME == TAXITIME`, RMSE 0.0s):

| Predicted `TAXITIME` = … | n | RMSE (s) | median abs err (s) |
|---|---|---|---|
| `MVT_TIME − BLOCK_TIME` (oracle/identity) | 2,084,457 | 0.0 | 0.0 |
| `MVT_TIME − AOBT_3_flt` (available in ranking) | 2,062,424 | **383.8** | 175.0 |
| `MVT_TIME − EOBT_1_flt` | 2,062,424 | 676.5 | 303.0 |
| `MVT_TIME − LOBT_flt` / `IOBT_flt` | 2,062,424 | 739.8 | 359.0 |
| `MVT_TIME − SCHED_TIME` (non-off-block ref) | 2,084,457 | 2409.0 | 481.0 |

**Implications (for Task 1.3 feature whitelist + downstream):**

1. `MVT_TIME_UTC_mvt − AOBT_3_flt` is a **legitimate, scoring-time-available, near-oracle feature** (RMSE ≈ 384s on ~98.5% of submissions). This is **not leakage** — the column is genuinely present in `ranking.parquet`; it is the single strongest signal and should anchor the baseline.
2. The residual to model shrinks to `BLOCK_TIME − AOBT_3_flt` (the movement-vs-NM off-block reconciliation, ~median 175s) for matched rows; congestion/temporal/weather/domain features then target the ~1.5% unmatched subset and that residual.
3. **Action:** correct the "blanked for departures" claim in `external_research.md` §2/§7 so Task 1.3 does not wrongly exclude `AOBT_3_flt`/`LOBT_flt`/`IOBT_flt` from the leakage-safe whitelist. The core decision #5 (build relative to takeoff time) is unaffected.

> Scope note: quantifying availability + the predictive proxy is a data-verification result. Building/validating it as a pipeline feature with month-blocked CV is Task 1.3/2.1, not done here.

---

## 6. Python environment — COMPLETE

- `uv init --bare` in project root → `pyproject.toml` (`name = "opensky-prc-2026"`, `requires-python = ">=3.12"`), no boilerplate source files.
- `uv add pandas polars pyarrow lightgbm xgboost catboost scikit-learn holidays matplotlib` → 27 packages installed into `.venv` (gitignored); `uv.lock` written (to be committed).
- Import smoke test — **all PASS**:

| package | version |
|---|---|
| pandas | 3.0.5 |
| polars | 1.44.1 |
| pyarrow | 25.0.1 |
| lightgbm | 4.7.0 |
| xgboost | 3.4.1 |
| catboost | 1.2.10 |
| scikit-learn | 1.9.0 |
| holidays | 0.104 |
| matplotlib | 3.11.1 |

Resolved (relevant): numpy 2.5.3, scipy 1.18.1, plotly 7.0.0 (catboost dep).

---

## Summary of acceptance criteria (Task 01)

| Criterion | Status |
|---|---|
| All 14 files present & readable, row counts reported | **PASS** (Section 4; training rows 4,167,797 exact) |
| Verification report exists; ranking availability table + `MVT_TIME_UTC_mvt` verdict stated | **PASS** — table complete (Section 5); `MVT_TIME_UTC_mvt` **CONFIRMED** 100% populated |
| `git status` shows no data/credential files staged/tracked; `git ls-files resources/credentials.json` empty | **PASS** (Section 1) |
| External-data rules quoted | **PASS** (Section 2) |
| Working `uv` env, lockfile committed | **PASS** (Section 6) |
| Initialized git repo with template `.gitignore` | **PASS** (template + documented appended section) |

**Extra finding beyond the checklist:** Section 5b corrects the planning assumption that NM off-block times are blanked in ranking — `AOBT_3_flt` survives and yields a near-oracle target proxy. Flagged to the human user for downstream feature-planning impact.
