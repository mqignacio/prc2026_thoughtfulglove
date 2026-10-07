# Feature Inventory — PRC-OpenSky Data Challenge 2026

Task: `03-feature_inventory` · Team: `thoughtful-glove`
Generated: 2026-09-08 · Status: **FINAL (decision session)**
Evidence: `src/reports/eda_findings.md` §4 (ranking availability audit), `src/reports/data_verification.md` §5b, `resources/external_research.md` §2/§5.

**Rule:** Later coding sessions must not add non-whitelisted features without updating this file and re-running the ranking-availability check.

---

## 1. Whitelist — Base Columns (28 of 30)

All columns below are confirmed populated in `ranking.parquet` PHASE=DEP (344,841 rows). Source: EDA §4.

### 1a. Movement columns (`_mvt`)

| # | Column | Type | Populated % (rank DEP) | Planned Transforms |
|---|--------|------|-----------------------:|--------------------|
| 1 | MVT_ID_mvt | float64 (ID) | 100.0000 | Row identifier only; never a feature |
| 2 | PHASE_mvt | string | 100.0000 | Filter to DEP for training; ARR used for congestion counts |
| 3 | RUNWAY_mvt | string | 100.0000 | → categorical; → runway heading (numeric); → runway direction group per airport |
| 4 | MVT_TIME_UTC_mvt | timestamp[us, UTC] | 100.0000 | → hour-of-day (sin/cos); → day-of-week; → local-hour (DST-aware per airport); → month; primary time anchor for congestion windows |
| 5 | SCHED_TIME_UTC_mvt | timestamp[us, UTC] | 100.0000 | → schedule deviation = MVT − SCHED; → scheduled hour-of-day; secondary congestion anchor |
| 6 | ADEP_mvt | string | 100.0000 | Airport identifier (10 levels); grouping key for per-airport models |
| 7 | STAND_mvt | string | 99.9988 | → categorical (high cardinality 133–238 per airport); → stand zone/group extracted from string pattern; target-encoded within airport |
| 8 | ADES_mvt | string | 99.9962 | → categorical (1,105 unique); route-distance proxy via haversine(ADEP, ADES); destination airport encoding |
| 9 | FLIGHT_RULE_mvt | string | 99.9942 | → binary IFR vs VFR (2 levels) |
| 10 | FLIGHT_mvt | string | 99.9832 | Flight number (20,329 unique); low direct value; may encode airline prefix for operator imputation on unmatched rows |
| 11 | AIRCRAFT_TYPE_mvt | string | 99.8875 | → categorical (199 levels); → wingspan class / size class mapping; → wake category fallback when _flt null |

### 1b. NM flight-plan columns (`_flt`, ~98.47% populated)

All `_flt` columns share the same null pattern: null when `FLIGHT_ID_mvt` is null (~1.53%, 5,283 rows). LightGBM handles NaN natively.

| # | Column | Type | Populated % (rank DEP) | Planned Transforms |
|---|--------|------|-----------------------:|--------------------|
| 12 | FLIGHT_ID_mvt | float64 (FK) | 98.4680 | → binary `nm_matched` indicator (drives all _flt availability) |
| 13 | AOBT_3_flt | timestamp[us, UTC] | 98.4660 | **Near-oracle anchor**: `MVT_TIME − AOBT_3` ≈ target (RMSE 384s); → taxi phase durations relative to off-block |
| 14 | LOBT_flt | timestamp[us, UTC] | 98.4660 | → `MVT − LOBT`; → `LOBT − EOBT_1` (pre-pushback delay); flow-control signal |
| 15 | IOBT_flt | timestamp[us, UTC] | 98.4660 | → `MVT − IOBT`; computed slot vs AOBT gap |
| 16 | EOBT_1_flt | timestamp[us, UTC] | 98.4660 | → `MVT − EOBT_1` (RMSE 677s); → `EOBT_1 − SCHED` (schedule change); departure planning reference |
| 17 | ARVT_1_flt | timestamp[us, UTC] | 98.4660 | Planned arrival; low direct taxi-out relevance; available for cross-check |
| 18 | ARVT_3_flt | timestamp[us, UTC] | 98.4660 | Same as above (later version) |
| 19 | AIRCRAFT_OPERATOR_flt | string | 98.4660 | → categorical (579 unique, anonymized hash); CV-safe target encoding; operator hub effects |
| 20 | AIRCRAFT_TYPE_flt | string | 98.4660 | Supplementary to `_mvt` type; use `_mvt` as primary (EDA §3: prefer movement table) |
| 21 | FLIGHT_TYPE_flt | string | 98.4660 | → categorical (5 levels: S/non-S charter etc.) |
| 22 | CALLSIGN_flt | string | 98.4660 | Airline prefix extractable (first 2–3 chars) → operator fallback for unmatched rows |
| 23 | MARKET_SEGMENT_flt | string | 98.4660 | → categorical (8 levels: Mainline/Lowcost/Regional/Cargo/Business/etc.) |
| 24 | ADES_FILED_flt | string | 98.4660 | → diversion flag = `(ADES_mvt ≠ ADES_FILED_flt)` (1.39% diverted) |
| 25 | ADES_flt | string | 98.4660 | NM-system destination; supplementary to `_mvt` |
| 26 | ADEP_flt | string | 98.4660 | Perfect agreement with `ADEP_mvt` (EDA §3); redundant but confirms join integrity |
| 27 | FLIGHT_RULE_flt | string | 98.4619 | Supplementary to `_mvt` rule |
| 28 | WK_TBL_CAT_flt | string | 98.4619 | → ordinal L(1) < M(2) < H(3) < J(4); strong monotonic effect (EDA §5: 700→1416s) |

---

## 2. Blacklist — Prohibited Columns (2 of 30)

| Column | Type | Populated % (rank DEP) | Reason |
|--------|------|-----------------------:|--------|
| TAXITIME_SEC_mvt | int32 | **0.0000** | **Target variable.** Blanked by organizers for scoring; using it would be trivial leakage. |
| BLOCK_TIME_UTC_mvt | timestamp[us, UTC] | **0.0000** | **Same-flight leakage.** The off-block time directly defines the target (`TAXI = MVT − BLOCK`). Blanked in ranking DEP. Not derivable from any other available column exactly. |

**Note on near-oracle derived features:** `MVT_TIME_UTC_mvt − AOBT_3_flt` is NOT blacklisted despite being highly predictive of the target. It uses two columns genuinely present at scoring time and is not identical to the target (RMSE 384s residual remains). This is legitimate feature engineering, not leakage.

---

## 3. Candidate Derived Features by Family

### 3a. Temporal

| Feature | Definition | Rationale |
|---------|-----------|-----------|
| hour_sin, hour_cos | sin/cos encoding of UTC hour from `MVT_TIME_UTC_mvt` | EDA §5: U-shaped pattern (trough 04–06, peak 09 & 23) |
| dow | Day-of-week (0–6) from `MVT_TIME_UTC_mvt` | Weekend vs weekday bank structure differences |
| local_hour_sin, local_hour_cos | DST-aware local hour per airport (UTC+0/+1/+2/+3) | EDA §5 + external §5: airports span 4 TZ offsets; local operations rhythm matters |
| month | Calendar month from `MVT_TIME_UTC_mvt` | Seasonal demand (Jan vs Jul scoring window) |
| sched_deviation_min | `(MVT_TIME − SCHED_TIME)` in minutes | Schedule slip captures operational disruption context |
| is_peak_departure | Boolean: hour in {07–09, 15–17} per airport local time | External §5: peak-bank congestion intervals |

### 3b. Airport-Geometry — **IMPLEMENTED (Task 09 ablation)**

**Ablation result:** D2 (+geometry) Δ=−2.83s vs D0 (baseline+METAR+congestion). Geometry is the strongest single domain family. Coordinates are hardcoded for the 10 reporting airports in `src/pdc26/airports.py` and extended to all 1,630 destinations via the `airportsdata` package (ICAO-keyed, 94.2% coverage). See `src/reports/domain_ablation.md`.

| Feature | Definition | Rationale |
|---------|-----------|-----------|
| airport_id | `ADEP_mvt` categorical (10 levels) | EDA §1: 2× spread across airports (741s LSZH to 1364s EGLL) |
| runway_id | `RUNWAY_mvt` categorical (50 levels) | EDA §5: within-airport variation up to 5× (EHAM rw22=278s vs rw24=666s) |
| runway_heading | Numeric heading decoded from runway designator (e.g., "27R" → 270°) | External §5: headwind/tailwind component affects taxi speed |
| stand_zone | Extracted zone/terminal group from `STAND_mvt` string pattern (146 levels) | Geometry proxy: distance from stand to active runway |
| stand_runway_interaction | Categorical concat(stand_zone, runway) (1,577 levels) | Interaction of stand group with active runway |
| dest_distance_km | Haversine(ADEP_mvt, ADES_mvt) via `airportsdata` | Route-length proxy: long-haul uses larger aircraft, different stands |

**Implementation:** `build_domain_features(groups=["geometry"])` in `src/pdc26/features.py`. `runway_heading`, `stand_zone`, `stand_runway_interaction` as integer codes (fixed vocab); `dest_distance_km` numeric. Native `RUNWAY_mvt` / `STAND_mvt` categoricals already present in baseline.

### 3c. Flight Characteristics — **IMPLEMENTED (Task 09 ablation)**

**Ablation result:** D3 (+flight_chars) Δ=−1.37s vs D0. `operator_enc` is a smoothed target encoding computed strictly within-fold (fit on train rows, mapped to validation rows); `assert_no_target_encoding_leakage` runs per fold. See `src/reports/domain_ablation.md`.

| Feature | Definition | Rationale |
|---------|-----------|-----------|
| wake_cat_ordinal | L=1, M=2, H=3, J=4 from `WK_TBL_CAT_flt` (already in baseline) | EDA §5: monotonic 700→1416s; strong independent signal |
| market_segment_code | Categorical `MARKET_SEGMENT_flt` (8 levels) | EDA §5: Mainline 1033s > Lowcost 969s > Regional 770s |
| flight_type_code | Categorical `FLIGHT_TYPE_flt` (5 levels) | Scheduled/charter/non-scheduled operational context |
| operator_enc | Target-encoded `AIRCRAFT_OPERATOR_flt` (677 levels, CV-safe, within-fold) | Operator procedures, gate assignment patterns |
| flight_rule | Binary IFR/VFR from `FLIGHT_RULE_mvt` (already in baseline) | VFR ops differ in routing/taxi patterns |
| diversion_flag | Boolean: `ADES_mvt ≠ ADES_FILED_flt` (1.39%) | Diversion context may correlate with weather/congestion |
| eobt_sched_delta_sec | `EOBT_1_flt − SCHED_TIME_UTC_mvt` in seconds | Airline planning slip before pushback |
| nm_matched | Boolean: `FLIGHT_ID_mvt` is not null (98.47%) | Single indicator for the ~1.5% unmatched block |

### 3d. Near-Oracle Anchors (from `_flt` off-block times)

| Feature | Definition | Rationale |
|---------|-----------|-----------|
| anchor_aobt_sec | `MVT_TIME_UTC_mvt − AOBT_3_flt` in seconds | Verification §5b: RMSE 384s vs true target; strongest single predictor |
| anchor_eobt_sec | `MVT_TIME_UTC_mvt − EOBT_1_flt` in seconds | RMSE 677s; backup when AOBT unavailable |
| anchor_iobt_sec | `MVT_TIME_UTC_mvt − IOBT_flt` in seconds | RMSE 740s; computed-slot timing |
| anchor_lobt_sec | `MVT_TIME_UTC_mvt − LOBT_flt` in seconds | RMSE 740s; last off-block target |
| pre_pushback_delay | `LOBT_flt − EOBT_1_flt` in seconds | Flow-control / slot-delay signal (time between planning and actual pushback) |
| aoht_vs_eobt | `AOBT_3_flt − EOBT_1_flt` in seconds | Off-block execution vs plan: ground delay program indicator |

### 3e. Congestion — **IMPLEMENTED (Task 08 ablation)**

Allowed time columns for congestion windows: `MVT_TIME_UTC_mvt` (primary), `SCHED_TIME_UTC_mvt`. All confirmed populated in both DEP and ARR phases in ranking.

**Ablation result:** C1 (all congestion) Δ=−0.34s vs C0 (baseline+METAR). Marginal overall improvement; per-airport gains at LTFM (−3.61s), LEBL (−3.45s), EDDM (−1.39s). Congestion features are whitelisted but contribute limited marginal signal beyond near-oracle anchors. See `src/reports/congestion_ablation.md`.

| Feature | Definition | Time basis | Rationale |
|---------|-----------|------------|-----------|
| dep_count_{5,15,30,60}min | Count of DEP movements at same airport within ±N min of focal takeoff | `MVT_TIME_UTC_mvt` (DEP rows) | External §5 ref-3: departure queue length is top predictor |
| arr_count_{5,15,30,60}min | Count of ARR movements at same airport within ±N min of focal takeoff | `MVT_TIME_UTC_mvt` (ARR rows) | Arrival pressure signal; EDA §5: r up to +0.14 (LEBL) |
| hourly_dep_actual | Actual DEP count at same airport in same calendar hour | `MVT_TIME_UTC_mvt` grouped | Airport-hour throughput load |
| hourly_arr_actual | Actual ARR count at same airport in same calendar hour | `MVT_TIME_UTC_mvt` grouped | Arrival-side load |
| hourly_dep_sched | Scheduled DEP count at same airport in same calendar hour | `SCHED_TIME_UTC_mvt` grouped | Planned throughput baseline |
| hourly_dep_gap | `hourly_dep_actual − hourly_dep_sched` | Derived | Demand-capacity imbalance signal |
| relative_busyness | Total movements (DEP+ARR) in airport-hour ÷ monthly median for that airport-month-hour_of_day | `MVT_TIME_UTC_mvt` grouped | Label-free relative busyness index; normalizes for airport operating rhythm |
| queue_proxy_30min | DEP takeoffs in [t−30, t] minus scheduled DEPs in [t−30, t] | `MVT_TIME` vs `SCHED_TIME` | Demand-vs-throughput imbalance proxy |
| runway_dep_count_15min | DEP count on same `RUNWAY_mvt` within ±15 min | `MVT_TIME_UTC_mvt` + `RUNWAY_mvt` | Runway-specific departure queue saturation |
| runway_arr_count_15min | ARR count on same `RUNWAY_mvt` within ±15 min | `MVT_TIME_UTC_mvt` + `RUNWAY_mvt` | Runway-specific arrival pressure |

**No-leakage guarantee:** All congestion features use only movement times (`MVT_TIME_UTC_mvt`, `SCHED_TIME_UTC_mvt`), phase labels, airport/runway IDs. Never reads `TAXITIME_SEC_mvt` or `BLOCK_TIME_UTC_mvt`. Counts are computed from movement times only, so no target leakage even when the focal row's own movement falls within its window.

**Implementation:** `build_congestion_features()` in `src/pdc26/features.py`. Uses sorted arrays + `np.searchsorted` for O(N log N) efficiency. Feature build time: 95.1s for 2.08M DEP rows against 4.17M movements.

**ARR completeness for congestion:** Ranking ARR rows (344,693) have `MVT_TIME_UTC_mvt` at 100.0%, `ADES_mvt` at 100.0%, `RUNWAY_mvt` at 100.0%. ARR phase is fully viable for congestion features in the scoring window.

### 3f. External Weather (METAR) — **WHITELISTED (Task 07 ablation)**

Source: Iowa State Mesonet ASOS archive. Join: nearest METAR at or before takeoff within 60 min (`merge_asof`, backward). Coverage: 100.0% of movements matched. Ablation result: A2 (METAR only) Δ=−1.55s vs baseline; strongest per-airport gains at EHAM (−4.36s), LTFM (−3.79s), EDDM (−3.40s).

| Feature | Definition | Rationale |
|---------|-----------|----------|
| metar_tmpc | Temperature (°C) at nearest observation | Cold-weather ops (de-icing, reduced visibility) increase taxi time |
| metar_dwpc | Dewpoint (°C) | Fog/condensation proxy when combined with temperature |
| metar_relh | Relative humidity (%) | Moisture-related operational impacts |
| metar_wind_dir | Wind direction (degrees) | Headwind/tailwind component affects taxi speed |
| metar_wind_kt | Wind speed (knots) | Strong winds may require different taxi routing |
| metar_wind_gust_kt | Wind gust speed (knots) | Gust events affect ground operations |
| metar_vis_miles | Visibility (statute miles) | Low visibility procedures increase taxi time |
| metar_alti_in | Altimeter setting (inHg) | Pressure system indicator |
| metar_ceiling_ft | Ceiling height (feet) | Low ceiling triggers IFR procedures |
| metar_flight_cat_ord | Flight category ordinal (VFR=0, MVFR=1, IFR=2, LIFR=3) | Composite weather severity index |
| minutes_since_metar | Minutes between METAR observation and takeoff | Recency of weather observation; stale obs less informative |

### 3g. Holiday Flags — **NOT WHITELISTED (Task 07 ablation: neutral)**

Ablation result: A1 (holidays only) Δ=+0.39s vs baseline (no improvement). National public holiday flags do not provide measurable signal for taxi-out prediction at this granularity. Not added to the active feature set. May be revisited if regional/sub-national holidays are added later.

---

## 4. Open Questions

| # | Question | Status | Resolution |
|---|----------|--------|------------|
| Q1 | Are arrival-phase rows in ranking complete enough for congestion features? | **RESOLVED** | Yes — 344,693 ARR rows; `MVT_TIME`, `ADES`, `RUNWAY` all ≥99.996% populated. Monthly distribution mirrors DEP (Jan 152K, Jul 192K). |
| Q2 | Should `AOBT_3_flt` be treated as leakage? | **RESOLVED** | No. Column is genuinely present in `ranking.parquet` (98.47%). Using it produces a near-oracle estimate but with 384s RMSE residual — real modeling work remains. Documented in verification §5b. |
| Q3 | Fallback branch: is `MVT_TIME_UTC_mvt` unreliable in ranking? | **RESOLVED — NOT TRIGGERED** | Confirmed 100.00% populated (verification §5, check g). Decision #5 stands. No need for schedule-time-only feature set. |
| Q4 | How to handle the ~1.5% unmatched flights (all `_flt` null)? | **RESOLVED** | Single binary `nm_matched` indicator. LightGBM native NaN handling for `_flt`-derived features. For these rows, anchors (§3d) are null; model falls back on temporal + geometry + `_mvt` features. |
| Q5 | Should we build separate models per airport or one global model? | **DEFERRED to Task 3.3** | EDA finding #1 supports both; decision by CV comparison. Feature inventory is model-agnostic. |
| Q6 | Is `CALLSIGN_flt` useful beyond operator identification? | **DEFERRED** | First 2–3 chars encode airline; redundant with `AIRCRAFT_OPERATOR_flt` for matched rows. Potential use: impute operator for the ~1.5% unmatched rows where `FLIGHT_mvt` prefix may help. Evaluate in Task 3.2. |

---

## 5. Fallback Branch Assessment (TODO Decision #5)

**Condition:** If `MVT_TIME_UTC_mvt` were blanked/unreliable in ranking DEP rows, exclude takeoff-time-derived features and define schedule-time-only set.

**Verdict: FALLBACK NOT TRIGGERED.**

Evidence: `data_verification.md` §5 check (g): `MVT_TIME_UTC_mvt` is **100.00% populated** (344,841/344,841) in ranking DEP. The primary strategy (congestion relative to takeoff time, near-oracle anchors from off-block times) is fully viable.

If this finding were ever invalidated (e.g., organizer data refresh), the schedule-time-only fallback would restrict features to: `SCHED_TIME_UTC_mvt` (hour/dow), `EOBT_1_flt`, `IOBT_flt`, `LOBT_flt` (as planning references), and all static categoricals. Congestion windows would shift to `SCHED_TIME_UTC_mvt` basis. Human user notification required before replanning.

---

## 6. Summary Statistics

| Metric | Value |
|--------|------:|
| Total schema columns | 30 |
| Whitelisted base columns | 28 |
| Blacklisted columns | 2 |
| Candidate derived features | 37 |
| Congestion features (implemented, Task 08) | 16 |
| Domain features (implemented, Task 09) | 14 |
| Near-oracle anchor features | 6 |
| External weather features (whitelisted, Task 07) | 11 |
| Holiday features (tested, not whitelisted) | 3 |
| Open questions resolved | 4 |
| Open questions deferred | 2 |

---

## 7. Usage Guidelines for Downstream Tasks

- **Task 2.1 (baseline):** Use `anchor_aobt_sec` + temporal + airport/runway/wake as minimum viable feature set. Expect RMSE well under 384s with these alone.
- **Task 3.1 (congestion):** §3e features implemented in `build_congestion_features()`. Ablation Δ=−0.34s vs baseline+METAR. Use `load_training_movements()` / `load_ranking_movements()` for DEP+ARR data.
- **Task 3.2 (domain):** §3b geometry + §3c flight characteristics + local-time implemented in `build_domain_features()` (§3a local-hour now DST-aware). Ablation Δ=−3.58s all-families vs baseline+METAR+congestion (D4 = 338.43s); geometry strongest (−2.83s). See `src/reports/domain_ablation.md`.
- **Any new feature:** Must cite which whitelisted base column(s) it derives from and verify those columns' populated rate hasn't changed.
- **External features (Task 07):** METAR weather features (§3f) are whitelisted based on ablation evidence (Δ=−1.55s). Use `join_external_features()` from `pdc26.features`. Holiday flags (§3g) tested but not included (neutral delta).
