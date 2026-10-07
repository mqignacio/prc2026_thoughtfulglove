# EDA Findings — PRC-OpenSky Data Challenge 2026

Task: `02-eda` · Team: `thoughtful-glove`
Generated: 2026-09-08 · Status: **COMPLETE**
Notebook: `src/notebooks/02_eda.ipynb` (27 cells, runs top-to-bottom without error)

---

## 1. Target Analysis — `TAXITIME_SEC_mvt` (DEP)

### Per-airport statistics (training DEP, 2,085,047 rows)

| Airport | n | Mean (s) | Median (s) | p95 (s) | Max (s) | ≤0 | <60s | >60min | >120min |
|---------|------:|--------:|-------:|------:|--------:|----:|-----:|-------:|--------:|
| EGLL | 239,546 | 1,364.2 | 1,319 | 1,989 | 11,452 | 0 | 7 | 728 | 37 |
| LIRF | 160,704 | 1,194.6 | 1,025 | 2,090 | 131,167 | 0 | 19 | 1,967 | 480 |
| LTFM | 272,965 | 1,053.0 | 963 | 1,749 | 10,309 | 2 | 3 | 794 | 21 |
| LFPG | 239,552 | 1,019.5 | 954 | 1,681 | 84,240 | 65 | 136 | 396 | 37 |
| LEMD | 212,242 | 1,014.7 | 985 | 1,559 | 3,598 | 0 | 3 | 0 | 0 |
| LEBL | 179,705 | 956.5 | 906 | 1,501 | 3,604 | 5 | 65 | 1 | 0 |
| EDDF | 230,141 | 863.5 | 837 | 1,399 | 6,907 | 0 | 4 | 73 | 0 |
| EDDM | 167,334 | 811.9 | 774 | 1,320 | 6,899 | 0 | 2 | 59 | 0 |
| EHAM | 247,951 | 784.6 | 742 | 1,309 | 11,032 | 3 | 192 | 82 | 3 |
| LSZH | 134,907 | 740.5 | 711 | 1,240 | 87,341 | 313 | 674 | 26 | 6 |

**Figure:** `assets/figures/02_01_target_hist_log.png` — log-scale histograms per airport showing right-skewed distributions that become approximately normal after log₁₀ transform.

### Key findings affecting modeling

- **EGLL is the outlier airport** for taxi-out duration (mean 22.7 min vs fleet-wide ~15 min). LIRF and LTFM are next tier (~17–18 min). LSZH/EHAM are fastest (~12–13 min). This confirms the need for **per-airport features or models**.
- **LIRF has the heaviest tail**: 1,967 rows >60 min (1.22% of its departures), 480 >2 hr, max 131,167s (~36 hr). These extreme values at LIRF likely reflect de-icing holds, technical stops, or data errors. RMSE will be heavily influenced by these.
- **Negative values**: 369 rows (0.018%) with taxi-out < 0, concentrated at LSZH (313) and LFPG (65). All are small magnitude (-1 to -12 seconds), consistent with timestamp rounding artifacts between MVT_TIME and BLOCK_TIME. **Action: clip to minimum 1 second or exclude.**
- **Zero/near-zero**: 19 exact zeros + 441 values in 1–29s range (0.021%). Negligible count but some may be genuine very-short taxi (cargo ops at night); others are data errors. **Action: flag but do not auto-exclude; let CV decide.**
- **Extreme values (>12 hr)**: only 34 rows across all airports. These are almost certainly data errors (AOG, maintenance relocation logged as movement). **Action: cap at a reasonable ceiling (e.g., 4 hr = 14,400s) via CV-tuned clipping.**

---

## 2. Missingness

### Null rates by column × dataset × phase (all values in %)

| Column | Rank DEP | Rank ARR | Train DEP | Train ARR |
|--------|----------:|----------:|-----------:|-----------:|
| MVT_ID_mvt | 0.00 | 0.00 | 0.00 | 0.00 |
| PHASE_mvt | 0.00 | 0.00 | 0.00 | 0.00 |
| RUNWAY_mvt | 0.00 | 0.00 | 0.00 | 0.00 |
| MVT_TIME_UTC_mvt | 0.00 | 0.00 | 0.00 | 0.00 |
| SCHED_TIME_UTC_mvt | 0.00 | 0.00 | 0.00 | 0.00 |
| ADEP_mvt | 0.00 | 0.004 | 0.00 | 0.004 |
| STAND_mvt | 0.0012 | 0.0015 | 0.0013 | 0.0017 |
| ADES_mvt | 0.0038 | 0.00 | 0.0059 | 0.00 |
| FLIGHT_RULE_mvt | 0.0058 | 0.0186 | 0.0097 | 0.0233 |
| FLIGHT_mvt | 0.0168 | 0.0113 | 0.0090 | 0.0107 |
| AIRCRAFT_TYPE_mvt | 0.1125 | 0.0061 | 0.0723 | 0.0068 |
| FLIGHT_ID_mvt | 1.5320 | 0.4088 | 1.0755 | 0.4084 |
| All `_flt` columns* | 1.5340 | 0.7404 | 1.0777 | 0.7418 |
| WK_TBL_CAT_flt | 1.5381 | 0.7439 | 1.0812 | 0.7450 |
| FLIGHT_RULE_flt | 1.5381 | 0.7439 | 1.0812 | 0.7450 |
| **BLOCK_TIME_UTC_mvt** | **100.00** | 0.00 | 0.00 | 0.00 |
| **TAXITIME_SEC_mvt** | **100.00** | 0.00 | 0.00 | 0.00 |

\* All `_flt` columns share the same null pattern because they are populated only when `FLIGHT_ID_mvt` matches an NM flight record.

**Figure:** `assets/figures/02_02_missingness_heatmap.png` — heatmap of null rates across all 30 columns × 4 dataset-phase combinations.

### Key findings affecting modeling

- **Only 2 columns are fully missing** in ranking DEP: `BLOCK_TIME_UTC_mvt` and the target `TAXITIME_SEC_mvt`. Every other column is ≥98.46% populated.
- The ~1.53% null block in `_flt` columns corresponds exactly to unmatched flights (`FLIGHT_ID_mvt` null rate = 1.532%). **Imputation strategy needed for ~5,283 ranking DEP rows** where no flight data exists.
- **ARR-phase rows have lower null rates** than DEP (~0.74% vs ~1.53% for _flt). This means arrival-based congestion features will have slightly better coverage.
- Core movement columns are always populated — these are the safest feature sources.

---

## 3. Join Quality

### FLIGHT_ID_mvt match rate

| Dataset | Matched | Unmatched | Match % |
|---------|--------:|----------:|--------:|
| Train DEP | 2,062,623 | 22,424 | 98.92% |
| Ranking DEP | 339,558 | 5,283 | 98.47% |

### Field mismatches between `_mvt` and `_flt` (where both non-null, train DEP)

| Field | Compared | Mismatched | Mismatch % |
|-------|----------:|----------:|----------:|
| ADEP | 2,062,577 | 0 | 0.0000% |
| ADES | 2,062,497 | 29,160 | 1.4138% |
| AIRCRAFT_TYPE | 2,062,577 | 7,013 | 0.3400% |
| FLIGHT_RULE | 2,062,340 | 1,349 | 0.0654% |

### Duplicate MVT_ID check

| Dataset | Duplicate MVT_IDs |
|---------|------------------:|
| Train DEP | 0 |
| Ranking DEP | 0 |

### Key findings affecting modeling

- **ADEP agreement is perfect** (0% mismatch) — the departure airport in the movement table always matches the flight table. Safe to use either source.
- **ADES has 1.41% disagreement** — filed destination ≠ actual destination (diversions, route changes). Use `_mvt` ADES as ground truth; `_flt` ADES_FILED as diversion signal.
- **Aircraft type mismatch (0.34%)**: minor but real. Movement-table type (from radar/schedule) vs flight-plan type can differ for codeshares or equipment swaps. Prefer `_mvt` for consistency.
- **No duplicate MVT_IDs** — each movement row is unique. No deduplication needed.
- **~1.5% of ranking DEP rows have no flight data at all** — features derived from `_flt` columns will be null for these. The model must handle this gracefully (LightGBM native NaN handling, or impute with "unmatched" indicator).

---

## 4. Ranking Availability Audit

Complete populated-rate table for all 30 columns, ranking PHASE=DEP (344,841 rows):

| Column | Populated | Total | Populated % |
|--------|----------:|------:|------------:|
| MVT_ID_mvt | 344,841 | 344,841 | 100.0000 |
| PHASE_mvt | 344,841 | 344,841 | 100.0000 |
| RUNWAY_mvt | 344,841 | 344,841 | 100.0000 |
| MVT_TIME_UTC_mvt | 344,841 | 344,841 | 100.0000 |
| SCHED_TIME_UTC_mvt | 344,841 | 344,841 | 100.0000 |
| ADEP_mvt | 344,841 | 344,841 | 100.0000 |
| STAND_mvt | 344,837 | 344,841 | 99.9988 |
| ADES_mvt | 344,828 | 344,841 | 99.9962 |
| FLIGHT_RULE_mvt | 344,821 | 344,841 | 99.9942 |
| FLIGHT_mvt | 344,783 | 344,841 | 99.9832 |
| AIRCRAFT_TYPE_mvt | 344,453 | 344,841 | 99.8875 |
| FLIGHT_ID_mvt | 339,558 | 344,841 | 98.4680 |
| AOBT_3_flt | 339,551 | 344,841 | 98.4660 |
| IOBT_flt | 339,551 | 344,841 | 98.4660 |
| ARVT_1_flt | 339,551 | 344,841 | 98.4660 |
| EOBT_1_flt | 339,551 | 344,841 | 98.4660 |
| AIRCRAFT_OPERATOR_flt | 339,551 | 344,841 | 98.4660 |
| AIRCRAFT_TYPE_flt | 339,551 | 344,841 | 98.4660 |
| FLIGHT_TYPE_flt | 339,551 | 344,841 | 98.4660 |
| CALLSIGN_flt | 339,551 | 344,841 | 98.4660 |
| MARKET_SEGMENT_flt | 339,551 | 344,841 | 98.4660 |
| ADES_FILED_flt | 339,551 | 344,841 | 98.4660 |
| ADES_flt | 339,551 | 344,841 | 98.4660 |
| ADEP_flt | 339,551 | 344,841 | 98.4660 |
| LOBT_flt | 339,551 | 344,841 | 98.4660 |
| ARVT_3_flt | 339,551 | 344,841 | 98.4660 |
| FLIGHT_RULE_flt | 339,537 | 344,841 | 98.4619 |
| WK_TBL_CAT_flt | 339,537 | 344,841 | 98.4619 |
| **TAXITIME_SEC_mvt** | **0** | 344,841 | **0.0000** |
| **BLOCK_TIME_UTC_mvt** | **0** | 344,841 | **0.0000** |

### Key findings affecting modeling

- **Leakage-safe whitelist**: all 28 columns except `TAXITIME_SEC_mvt` and `BLOCK_TIME_UTC_mvt` are available at scoring time. Any feature built from these 28 columns is legitimate.
- **`AOBT_3_flt` confirmed at 98.47%** — validates Task 01 §5b finding. The near-oracle proxy `MVT_TIME_UTC_mvt − AOBT_3_flt` is viable for ~339,551 of 344,841 ranking DEP rows.
- **All movement columns ≥99.89% populated** — runway, stand, aircraft type, times are all reliable primary feature sources.
- **The 1.53% gap** (5,283 rows) is uniform across all `_flt` columns — they all go null together when `FLIGHT_ID_mvt` is null. This is a single binary indicator ("matched to NM") rather than per-column sparsity.

---

## 5. Basic Relationships

### Taxi-out vs hour-of-day (UTC, training DEP)

| Hour | Mean (s) | Median (s) | Count | | Hour | Mean (s) | Median (s) | Count |
|-----:|--------:|--------:|--------:|-|-----:|--------:|--------:|--------:|
| 00 | 1,116.1 | 895 | 5,026 | | 12 | 1,040.2 | 969 | 130,182 |
| 01 | 1,032.9 | 903 | 4,506 | | 13 | 1,014.3 | 957 | 123,955 |
| 02 | 1,089.0 | 1,025 | 7,424 | | 14 | 1,003.0 | 923 | 130,623 |
| 03 | 928.1 | 845 | 18,584 | | 15 | 956.7 | 889 | 124,920 |
| 04 | 894.1 | 842 | 49,775 | | 16 | 985.2 | 904 | 112,059 |
| 05 | 911.8 | 852 | 95,282 | | 17 | 973.5 | 899 | 98,892 |
| 06 | 904.3 | 840 | 98,340 | | 18 | 961.2 | 899 | 103,249 |
| 07 | 936.6 | 870 | 104,845 | | 19 | 963.7 | 900 | 114,871 |
| 08 | 992.0 | 924 | 125,479 | | 20 | 983.7 | 912 | 111,613 |
| 09 | 1,052.5 | 971 | 137,602 | | 21 | 1,076.2 | 1,008 | 57,577 |
| 10 | 1,020.7 | 954 | 141,119 | | 22 | 1,079.3 | 968 | 28,995 |
| 11 | 1,028.9 | 964 | 141,394 | | 23 | 1,197.3 | 1,072 | 18,735 |

**Figure:** `assets/figures/02_03_hour_of_day.png` — bar charts of mean taxi-out and departure volume by UTC hour.

### Taxi-out vs runway (top/bottom, ≥500 ops)

| Airport | Runway | Mean taxi (s) | n |
|---------|--------|-------------:|------:|
| LTFM | 16R | 1,691.7 | 1,516 |
| LTFM | 34L | 1,604.2 | 8,796 |
| LIRF | 16R | 1,525.4 | 11,219 |
| LIRF | 34L | 1,430.1 | 4,139 |
| EGLL | 27R | 1,399.9 | 77,027 |
| ... | | | |
| EHAM | 22 | 278.0 | 4,697 |
| EHAM | 04 | 352.6 | 2,833 |
| EHAM | 24 | 666.4 | 85,236 |
| LSZH | 28 | 690.4 | 84,800 |
| EHAM | 36C | 728.7 | 30,900 |

### Taxi-out by wake category

| Wake Cat | Mean (s) | Median (s) | n |
|----------|--------:|--------:|--------:|
| J (Super) | 1,416.1 | 1,327 | 14,712 |
| H (Heavy) | 1,172.5 | 1,090 | 420,886 |
| M (Medium) | 936.5 | 878 | 1,616,701 |
| L (Light) | 700.1 | 603 | 10,204 |

### Taxi-out by market segment

| Segment | Mean (s) | Median (s) | n |
|---------|--------:|--------:|--------:|
| Mainline | 1,032.7 | 957 | 1,325,748 |
| Cargo | 1,018.3 | 953 | 60,021 |
| Non-Scheduled | 995.6 | 905 | 14,577 |
| Lowcost | 969.2 | 911 | 399,482 |
| Other | 889.9 | 832 | 6,673 |
| Military | 835.0 | 723 | 713 |
| Business | 779.3 | 697 | 44,498 |
| Regional | 770.4 | 724 | 210,865 |

### Congestion: arrival volume and co-departures vs taxi-out

Sampled 50,000 DEP rows across first 5 airports. Pearson correlation between taxi-out and:

| Airport | ARR ±30min | Co-DEP ±15min |
|---------|----------:|-------------:|
| EDDF | -0.069 | +0.120 |
| EDDM | -0.035 | +0.000 |
| EGLL | -0.009 | +0.200 |
| EHAM | +0.057 | -0.034 |
| LEBL | +0.139 | +0.175 |

**Figure:** `assets/figures/02_04_congestion_arr_vs_taxi.png` — scatter plots of arrival count ±30min vs taxi-out minutes, per airport.

### Key findings affecting modeling

- **Hour-of-day shows a U-shaped pattern**: peaks at 23:00 UTC (mean 1,197s) and 09:00 (1,053s); troughs at 04:00–06:00 (894–912s). Late-night hours have low volume but high mean (cargo operations with longer taxi routes). Peak traffic hours (09:00–12:00) also show elevated taxi times due to congestion. **Action: include sin/cos encoded hour-of-day + local-hour features.**
- **Runway matters significantly within airports**: LTFM 16R is 1,692s vs 34L at 1,604s; EHAM 22 is 278s vs 24 at 666s. Same-airport runway assignment changes taxi distance dramatically. **Action: runway should be a categorical feature; consider runway-direction (heading) encoding.**
- **Wake category has strong monotonic effect**: J > H > M > L with ~700s spread. Larger aircraft taxi slower due to speed restrictions and routing. **Action: include wake category; it partially overlaps with aircraft type but adds independent signal.**
- **Market segment ordering is logical**: Mainline/Cargo (hub operations, large gates) > Lowcost (secondary terminals) > Regional/Business (smaller aircraft, closer stands). **Action: include as categorical.**
- **Congestion signal is weak-to-moderate in raw form**: ARR count ±30min correlates r = -0.07 to +0.14; co-departure count ±15min up to r = +0.20 (EGLL). The linear relationship is modest, but **non-linear interaction effects (queue buildup, runway saturation) are expected** and tree models capture these naturally. **Action: build rolling-window congestion counts as features, but don't expect them to dominate — airport/runway/time-of-day carry more signal individually.**
- **EDDF shows negative ARR correlation** (-0.069): counter-intuitive, possibly because arrivals and departures use different runways at EDDF, reducing interference. **Action: consider runway-configuration-aware congestion features in Task 03+.**

---

## Summary of Modeling-Relevant Findings

| # | Finding | Impact on modeling |
|---|---------|--------------------|
| 1 | Taxi-out varies 2× across airports (741s LSZH to 1364s EGLL) | Per-airport features or separate models essential |
| 2 | Right-skewed distribution, approximately log-normal | Consider log-target transform or robust loss |
| 3 | 4,126 rows >60 min; 34 rows >12 hr (data errors) | Clipping/winsorizing strategy needed (CV-tuned) |
| 4 | 369 negative values (timestamp rounding) | Clip to ≥1s or exclude; negligible impact |
| 5 | Only 2 columns unavailable in ranking DEP | 28-column leakage-safe whitelist confirmed |
| 6 | AOBT_3_flt survives in ranking (98.47%) | Near-oracle anchor: MVT_TIME − AOBT_3 ≈ target |
| 7 | 1.53% unmatched rows have all _flt columns null | Single "NM matched" binary indicator handles this |
| 8 | ADES mismatch 1.41%; AIRCRAFT_TYPE mismatch 0.34% | Prefer _mvt fields; use _flt as supplementary |
| 9 | No duplicate MVT_IDs | No deduplication step needed |
| 10 | Hour-of-day U-shape; peak at 23:00 and 09:00 | Include cyclical time encoding + local time |
| 11 | Within-airport runway variation up to 5× (EHAM 22 vs 24) | Runway as categorical + heading encoding |
| 12 | Wake category monotonic: J(1416)>H(1173)>M(937)>L(700) | Strong ordinal feature |
| 13 | Congestion correlations weak-moderate (r ≤ 0.20) | Build but don't over-rely; GBT captures interactions |
| 14 | EDDF negative ARR correlation suggests runway config matters | Config-aware congestion features for future tasks |
