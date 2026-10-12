# Final-Phase Data Verification (Task 17)

**Status:** complete · **Verified (machine date):** 2026-10-09 10:55 +0800 (Asia/Manila; `date` reports the tz label `PST` but the offset is `+0800` — the label is a known macOS/Asia-Manila artifact, see TODO audit A3) · **Machine:** Mac Mini (`macmini-hub`) · **Verifier:** `src/verify_final_data.py` (re-runnable; all numbers below are its printed output).

Scope: verification only — no model runs, no predictions. Consumed by Tasks 18, 19, 21, and the JOAS paper update (Task 23).

## Summary — three GO/NO-GO verdicts

| # | Question | Result | Verdict |
|---|---|---|---|
| a | **Anchor survives** (`AOBT_3_flt` family populated at ranking rates) | final pooled **98.6468%** vs ranking **98.4660%** (+0.18 pts); worst month 98.3938% | **GO** |
| b | **Hard-row structure present** (null-operator rate ≥ ~1%) | final DEP operator-null **1.3532%** (9,077/670,790) vs 2025 reference 1.08% | **GO** |
| c | **External features join ≥ 99%** of final DEP rows | worst-month METAR **99.9659%**; holiday **100.0000%** | **GO** |

**OVERALL: ALL CHECKS GO.** The pipeline design (anchor-first feature family, hard-row experiment, external-feature coverage) remains valid for the final phase.

Supporting checks: blank discipline **PASS**; dither within ±12 s **PASS**.

## 0. File inventory — size, SHA256, bucket match

Both files were downloaded with `uv run --with minio python src/minio_transfer.py get opensky/prc-2026-datasets/<file> data/raw/<file>` (credentials read from `~/.mc/config.json`, never printed).

| File | Size | Bytes | Bucket size | Match | SHA256 |
|---|---|---|---|---|---|
| `final_ranking.parquet` | 81.876 MB | 85,853,399 | 85,853,399 | ✅ | `4b0692a792846079516a2749e034b0abda55df72c213eabdcda52a4d1f6bc69f` |
| `final_submitting.parquet` | 2.865 MB | 3,003,722 | 3,003,722 | ✅ | `b3445d80e29bf599a5583ca9675f63ddf57b51fdf1ca3f498facb38ea84f3924` |

Sizes match the 2026-10-08 11:04 UTC upload listing exactly.

## 1. Structure

- `final_ranking`: **1,341,197 rows**, **30 columns**.
- Months present: `['2026-01', '2026-02', '2026-06', '2026-07']` — **exactly the expected set**.
- `MVT_TIME_UTC_mvt` null rate over all rows: **0.0000%**.

PHASE counts per month:

| Month | ARR | DEP |
|---|---:|---:|
| 2026-01 | 152,871 | 152,719 |
| 2026-02 | 144,274 | 144,158 |
| 2026-06 | 181,440 | 181,791 |
| 2026-07 | 191,822 | 192,122 |
| **Total** | **670,407** | **670,790** |

Per-reporting-airport rows (DEP counted by `ADEP_mvt`, ARR by `ADES_mvt`):

| Airport | DEP | ARR | TOTAL |
|---|---:|---:|---:|
| LTFM | 89,793 | 90,187 | 179,980 |
| LFPG | 77,520 | 77,324 | 154,844 |
| EGLL | 77,221 | 77,214 | 154,435 |
| EHAM | 75,865 | 75,828 | 151,693 |
| LEMD | 72,480 | 72,394 | 144,874 |
| EDDF | 70,671 | 70,603 | 141,274 |
| LEBL | 59,314 | 59,155 | 118,469 |
| LIRF | 52,146 | 52,033 | 104,179 |
| EDDM | 50,872 | 50,819 | 101,691 |
| LSZH | 44,908 | 44,850 | 89,758 |

Distinct DEP `ADEP_mvt` = **10** (expected 10).

## 2. Schema — `final_ranking` vs `ranking`

- Column **name set equal**: ✅ ; **name order equal**: ✅ ; **dtype mismatches: 0**.
- `missing in final: []` ; `extra in final: []`.

Final schema (30 columns): `MVT_ID_mvt` (float64); `FLIGHT_ID_mvt` (float64); `FLIGHT_mvt`, `FLIGHT_RULE_mvt`, `ADEP_mvt`, `ADES_mvt`, `PHASE_mvt`, `AIRCRAFT_TYPE_mvt`, `RUNWAY_mvt`, `STAND_mvt` (str); `MVT_TIME_UTC_mvt`, `BLOCK_TIME_UTC_mvt`, `SCHED_TIME_UTC_mvt` (datetime64[us, UTC]); `TAXITIME_SEC_mvt` (float64); `LOBT_flt`, `IOBT_flt`, `EOBT_1_flt`, `ARVT_1_flt`, `AOBT_3_flt`, `ARVT_3_flt` (datetime64[us, UTC]); `CALLSIGN_flt`, `ADEP_flt`, `ADES_flt`, `ADES_FILED_flt`, `MARKET_SEGMENT_flt`, `FLIGHT_RULE_flt`, `FLIGHT_TYPE_flt`, `AIRCRAFT_TYPE_flt`, `WK_TBL_CAT_flt`, `AIRCRAFT_OPERATOR_flt` (str).

**No schema change vs `ranking.parquet`** — the serving feature code needs no schema adaptation.

## 3. `final_submitting` — rows, nulls, ID order

- Rows: **670,790** = **exactly the `final_ranking` DEP row count** (match ✅).
- Columns: `['MVT_ID_mvt', 'TAXITIME_SEC_mvt']`.
- `MVT_ID_mvt` nulls: **0** (expected 0). `TAXITIME_SEC_mvt` null rate: **100.00%** — this is the submission template placeholder (the column to be filled), identical to the original `submitting.parquet`.
- **Row order exact vs `final_ranking` DEP: True** (element-by-element `MVT_ID_mvt` sequence). ID set equality: True.
- Jan/Jul overlap with `submitting.parquet`: final Jan/Jul DEP ids = 344,841; `submitting.parquet` ids = 344,841; **set equality True**; 0 in one but not the other. `submitting.parquet` ⊆ `final_submitting` exactly.

## 4. Anchor availability [CRITICAL]

Populated rate of the NM off-block columns on **final DEP rows**, per month and pooled, vs `ranking.parquet` DEP rows (recomputed in-script, not quoted):

| Column | 2026-01 | 2026-02 | 2026-06 | 2026-07 | **POOLED** |
|---|---:|---:|---:|---:|---:|
| `AOBT_3_flt` | 98.3938 | 99.0843 | 98.6429 | 98.5233 | **98.6468** |
| `LOBT_flt` | 98.3938 | 99.0843 | 98.6429 | 98.5233 | **98.6468** |
| `IOBT_flt` | 98.3938 | 99.0843 | 98.6429 | 98.5233 | **98.6468** |
| `EOBT_1_flt` | 98.3938 | 99.0843 | 98.6429 | 98.5233 | **98.6468** |
| `ARVT_1_flt` | 98.3938 | 99.0843 | 98.6429 | 98.5233 | **98.6468** |

`ranking.parquet` DEP reference (recomputed): all five columns **98.4660%** populated (1.5340% null).

**Verdict (a): GO.** Final pooled `AOBT_3_flt` populated 98.6468% is +0.18 pts above the ranking rate; the worst month (2026-01, 98.3938%) is still well above the 90% floor. The near-oracle anchor family survives in the final files — pipeline design unchanged, no escalation needed.

## 5. Hard-row structure

- Pooled null `AIRCRAFT_OPERATOR_flt` on final DEP rows: **1.3532%** (9,077/670,790); 2025 OOF reference 1.08%.
- Fully-null `_flt` block (all 16 `_flt` columns null): **1.3532%** (9,077) — identical to the operator-null set.
- Per month: 2026-01 1.6062%, 2026-02 0.9157%, 2026-06 1.3571%, 2026-07 1.4767%.
- Per airport: EHAM 2.4965%, LSZH 2.3314%, LFPG 1.6834%, LTFM 1.4801%, LEBL 1.2796%, LIRF 1.1871%, EDDF 1.0839%, EDDM 0.8826%, EGLL 0.6099%, LEMD 0.6043%.

**Verdict (b): GO.** The hard-row structure is present at ~1.35% pooled, above the ~1% threshold, so the Task 19 hard-row hypothesis is testable on the final files. Note the airport mix differs from 2025: the nulls are now led by EHAM and LSZH, whereas the 2025 tail was LIRF-dominated — this is a distribution shift worth carrying into Task 19's gate design, not a blocker.

## 6. Blank discipline

| Column | DEP null% | ARR null% |
|---|---:|---:|
| `BLOCK_TIME_UTC_mvt` | 100.0000 | 0.0000 |
| `TAXITIME_SEC_mvt` | 100.0000 | 0.0000 |
| `MVT_TIME_UTC_mvt` | 0.0000 | 0.0000 |

**PASS** — targets (`BLOCK_TIME`/`TAXITIME`) are blanked for DEP and populated for ARR; the anchor time is 100% populated.

## 7. Dither quantification

Jan/Jul DEP rows joined to `ranking.parquet` on `MVT_ID_mvt` (id set equality True; 344,841 joined rows).

- `MVT_TIME_UTC_mvt` delta min/max: **−12.0 s / +12.0 s**.
- |Δ| ≤ 12 s: **100.00%**; Δ = 0: **8.17%**; 0 < |Δ| ≤ 12 s: **91.83%**; |Δ| > 12 s: **0.0000%**.
- Distribution: ≤−7 s 36,265; −6..−1 s 121,956; 0 s 28,190; 1..6 s 121,882; ≥7 s 36,548.

**Interpretation.** ~91.83% of Jan/Jul movements moved, matching the organizer's "~92% of movements" statement; the dither magnitude is bounded at ±12 s. `ranking.parquet` acts as the undithered baseline and the final export re-drew a fresh ±12 s shift per movement. In quadrature this adds ≈6.9 s RMS to the ~384 s anchor — negligible (<0.02% in quadrature). Train/serve consistency holds because the 2025 training files carry the same dither.

## 8. External feature coverage

METAR (per-airport `merge_asof`, backward, 60-min tolerance — same semantics as `src/pdc26/features.py::join_external_features`):

- Observations loaded: 277,489; pooled join rate **99.9884%** (670,712/670,790).
- Per month: 2026-01 100.0000%, 2026-02 100.0000%, 2026-06 99.9659%, 2026-07 99.9917%.

Holiday features (`data/processed/holiday_features.parquet`, regenerated this session — it was missing on this machine):

- Rows: 5,770; airports 10; date range 2025-01-01 → 2026-07-31.
- Pooled join rate **100.0000%**; every month 100.0000%.
- Coverage of the four final months confirmed: 2026-01 (310 rows), 2026-02 (280), 2026-06 (300), 2026-07 (310) — all 10 airports each.

**Verdict (c): GO** — external features join ≥ 99% of final DEP rows in every month.

## 9. Script output (verbatim)

The complete stdout of `uv run python src/verify_final_data.py` is reproduced in the sections above (0)–(9). Re-running the script on the same files reproduces these numbers; no check is eyeball-only.

## Notes for downstream tasks

- **Task 18 (serving pipeline):** schema is identical to `ranking.parquet` (0 dtype mismatches), so the frozen v5 feature code applies unchanged. Rebuild features from `final_ranking.parquet` (never copy v5 predictions); the dither is the only time change.
- **Task 19 (hard-row experiment):** hard rows are present at 1.3532% pooled (per-month 0.92–1.61%); the null set equals the fully-null `_flt` block exactly. Budget the experiment against this final-file rate, and note the EHAM/LSZH shift.
- **Task 21/23 (repo/paper):** this report is the evidence for the anchor-availability claim in the JOAS update and the review-ready repo.
