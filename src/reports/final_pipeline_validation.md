# Final Pipeline Validation (Task 18)

**Status:** complete · **Generated (machine clock):** 2026-10-11 08:19 +0800 (Asia/Manila; the `%z` offset is authoritative — macOS labels the zone `PST`, see TODO audit A3) · **Verifier:** `src/validate_final_pipeline.py` (re-runnable; all numbers below are its output)

Scope: validation of the final-phase serving artifact built by `src/make_final_submission.py` from `final_ranking.parquet` through the frozen v5 stack. No model retraining, no new features, no recalibration refits. The EUROCONTROL taxi-out aggregates are not used in any form (decision Q2-A / D2).

## 1. Artifact and pre-upload assertions

| Check | Result |
|---|---|
| Row count == `final_submitting` | 670,790 — **PASS** |
| MVT_ID order exact vs `final_submitting` | **PASS** |
| Nulls | 0 — **PASS** |
| All finite | **PASS** |
| Min ≥ 1 s | 1.000 — **PASS** |
| `TAXITIME_SEC_mvt` dtype | float64 — **PASS** |
| MVT_ID dtype matches template | **PASS** |

Artifact: `submissions/final/thoughtful-glove_final_v1.parquet`, 8,641,587 bytes, SHA256 `e326deac578d92329a28a08d3acb752e136ae66d6ce335f7cc7afe3dea4fc632`.

Prediction stats (full 4 months, n=670,790): mean=1016.2 s, median=947.3 s, min=1.0 s, max=42811.8 s, std=465.9 s.

## 2. Post-processing reproduction (independent)

The validation script re-derived the blend from the side table's per-component predictions and re-applied the affine with its own implementation. Max absolute difference: blend 0.00e+00 s, final 0.00e+00 s (float64 rounding only). **PASS**.

## 3. Reproduction gate (byte-identical rebuild)

Two independent builds of `src/make_final_submission.py`; the first was saved as a baseline and the second compared byte-for-byte. run1 SHA256 `e326deac578d92329a28a08d3acb752e136ae66d6ce335f7cc7afe3dea4fc632`, run2 SHA256 `e326deac578d92329a28a08d3acb752e136ae66d6ce335f7cc7afe3dea4fc632` — byte-identical: **True** (PASS).

## 4. Prediction distribution sanity

### 4a. 2025 analog RMSE (recomputation of the quoted figures)

| Segment | Rows | Raw blend RMSE (s) | Post-affine RMSE (s) | Quoted blend (s) | Δ |
|---|---:|---:|---:|---:|---:|
| 2025-01 | 153,706 | 383.74 | 379.95 | 383.74 | -0.00 |
| 2025-02 | 143,732 | 399.15 | 392.95 | 399.15 | -0.00 |
| 2025-06 | 183,142 | 366.90 | 364.23 | 366.90 | +0.00 |
| 2025-07 | 190,713 | 503.96 | 503.46 | 503.96 | -0.00 |
| 4-month (Jan+Feb+Jun+Jul) | 671,293 | 420.37 | 417.52 | 420.37 | -0.00 |
| 2-month (Jan+Jul) | 344,419 | 454.25 | 452.53 | 454.25 | +0.00 |

The raw-blend column reproduces the quoted 2025-analog figures exactly (Δ ≈ 0), confirming the OOF blend reconstruction. The post-affine column applies the accepted `mi2_affine_s50` **full-train** fit in-sample to the OOF predictions — it is shown for distribution context only. The honest leave-one-month-out season figure is **452.90 s** (`src/configs/blend_postproc.json`).

### 4b. Prediction distributions by month (final 2026 vs 2025 analog)

| Final month | Analog | n | final mean/median/p95 | analog mean/median/p95 | mean shift | flag |
|---|---|---:|---|---|---:|---|
| 2026-01 | 2025-01 | 152,719 | 1080.2 / 976.7 / 1926.0 | 1007.4 / 936.9 / 1697.1 | +7.2% | ok |
| 2026-02 | 2025-02 | 144,158 | 980.2 / 918.0 / 1596.8 | 991.7 / 923.3 / 1660.4 | -1.2% | ok |
| 2026-06 | 2025-06 | 181,791 | 995.5 / 941.0 / 1586.0 | 975.5 / 924.0 / 1554.0 | +2.0% | ok |
| 2026-07 | 2025-07 | 192,122 | 1011.9 / 954.0 / 1593.6 | 1026.8 / 961.3 / 1648.0 | -1.5% | ok |

### 4c. Prediction distributions by airport (pooled 4 months)

| Airport | n | final mean/median/p95 | analog mean/median/p95 | mean shift | flag |
|---|---:|---|---|---:|---|
| EDDF | 70,671 | 908.4 / 877.8 / 1357.6 | 863.5 / 850.9 / 1295.4 | +5.2% | ok |
| EDDM | 50,872 | 840.0 / 770.1 / 1485.8 | 811.9 / 774.3 / 1239.8 | +3.5% | ok |
| EGLL | 77,221 | 1348.4 / 1314.0 / 1867.1 | 1364.2 / 1331.9 / 1873.4 | -1.2% | ok |
| EHAM | 75,865 | 825.1 / 758.7 / 1361.0 | 784.6 / 740.1 / 1247.8 | +5.2% | ok |
| LEBL | 59,314 | 981.7 / 932.9 / 1445.6 | 956.5 / 907.8 / 1381.6 | +2.6% | ok |
| LEMD | 72,480 | 1029.0 / 1010.6 / 1474.5 | 1014.7 / 999.4 / 1447.6 | +1.4% | ok |
| LFPG | 77,520 | 990.6 / 920.9 / 1553.1 | 1019.5 / 962.7 / 1541.3 | -2.8% | ok |
| LIRF | 52,146 | 1213.5 / 1067.6 / 2003.4 | 1194.6 / 1067.7 / 1897.8 | +1.6% | ok |
| LSZH | 44,908 | 784.4 / 746.7 / 1279.5 | 740.5 / 734.8 / 1117.7 | +5.9% | ok |
| LTFM | 89,793 | 1112.3 / 1036.2 / 1767.6 | 1053.0 / 998.5 / 1627.1 | +5.6% | ok |

### 4d. Serving-regression check vs v5's own Jan/Jul predictions

Same calendar months (undithered v5 input vs dithered final input), per-MVT_ID comparison.

| Month | n | final mean (s) | v5 mean (s) | mean Δ | median \|Δ\| (s) | p95 \|Δ\| (s) |
|---|---:|---:|---:|---:|---:|---:|
| 2026-01 | 152,719 | 1080.18 | 1072.18 | +0.75% | 7.03 | 46.71 |
| 2026-07 | 192,122 | 1011.86 | 1011.78 | +0.01% | 9.40 | 57.00 |

v5 leaderboard reference: 392.5911 s.

## 5. Hard-row (null-operator) prediction distribution

Null-operator rows: 9,077 (1.3532%); base rows: 661,713.

| Segment | mean (s) | median (s) | p95 (s) |
|---|---:|---:|---:|
| hard (null operator) | 1409.1 | 1041.0 | 2960.1 |
| base | 1010.8 | 946.1 | 1648.2 |

## 6. Adversarial-validation drift check

LightGBM binary classifier separating 2025 training-analog features from final 2026 features (63-column D4 matrix, identical fitted artifacts both sides; features are not re-fitted per side). Two CV schemes are reported: **row-split** (StratifiedKFold, the literature default) and **day-grouped** (GroupKFold by calendar date, which prevents the classifier from memorising per-day signatures — see the interpretation below).

**Pooled (4 analog months 2025 vs 4 final months 2026):** row-split AUC = 0.9998 ± 0.0000; day-grouped AUC = 0.9678 ± 0.0162.

| Comparison (analog → final) | n | row-split AUC | ± 95% CI | day-grouped AUC | ± 95% CI |
|---|---:|---:|---:|---:|---:|
| 2025-01 → 2026-01 | 306,425 | 1.0000 | 0.0000 | 0.9659 | 0.0515 |
| 2025-02 → 2026-02 | 287,889 | 1.0000 | 0.0000 | 0.9351 | 0.0464 |
| 2025-06 → 2026-06 | 364,933 | 1.0000 | 0.0000 | 0.9604 | 0.0336 |
| 2025-07 → 2026-07 | 382,835 | 1.0000 | 0.0000 | 0.9388 | 0.0813 |

**Feature-group attribution** (single stratified 70/30 split per family; isolates which families carry the separability):

| Feature family | n | AUC |
|---|---:|---:|
| anchors + schedule (predictive core) | 7 | 0.5565 |
| temporal (hour/dow/month) | 4 | 0.5230 |
| flight characteristics | 3 | 0.5020 |
| categoricals | 8 | 0.6614 |
| METAR weather | 11 | 0.8564 |
| congestion counts | 16 | 0.9989 |

**Top-10 discriminative features (gain, pooled row-split model):**

| Rank | Feature | Gain |
|---:|---|---:|
| 1 | `RUNWAY_mvt` | 858,334 |
| 2 | `arr_count_5min` | 841,165 |
| 3 | `metar_alti_in` | 750,563 |
| 4 | `dep_count_5min` | 697,477 |
| 5 | `runway_arr_count_15min` | 694,077 |
| 6 | `dep_count_60min` | 626,857 |
| 7 | `ADEP_mvt` | 527,962 |
| 8 | `runway_dep_count_15min` | 453,776 |
| 9 | `month` | 452,623 |
| 10 | `dep_count_15min` | 439,450 |

### Interpretation

*(Author interpretation; the numeric values above are script output.)*

The pooled row-split AUC is far above the 0.7 flag threshold, so the drift was investigated by feature family and with a day-grouped CV. The result is unambiguous:

- **The predictive core is stable.** Anchors + schedule = 0.56, temporal = 0.52, flight characteristics = 0.50 — no material shift in the families that carry the model's signal (the near-oracle takeoff-time anchor family is the pipeline's strongest predictor; see `src/reports/error_analysis.md`).
- **The high AUC comes entirely from METAR weather (0.86) and congestion counts (0.999).** The 2025-analog and 2026-final windows had genuinely different weather (benign covariate, used as a minor correction). The congestion counts are dominated by a latent units bug in the frozen v5 feature code: `build_congestion_features` builds windows with a nanosecond constant (`MIN_NS`) while the timestamps are microsecond `datetime64[us]`, so the nominal 5/15/30/60-minute windows are ~1000× too wide (±83 h to ±42 days; the "5-min" feature mean is ~3985, i.e. a multi-day airport activity count). Those wide counts act as dataset-specific fingerprints.
- **Two CV schemes together separate the two causes.** Row-split AUC ≈ 1.000 is inflated by per-day memorisation (the two sets are disjoint calendar days, so day-specific weather signatures are memorisable); day-grouped AUC ≈ 0.96–0.97 removes that channel but stays high because the METAR and over-wide congestion differences are day-generalizable.
- **Reproducibility note:** the AV classifier is a diagnostic, not a gate artifact. The table values carry ~±0.002 run-to-run variation (multithreaded LightGBM) despite the `deterministic`/`force_row_wise` settings; the headline conclusions are invariant. The submission artifact itself is exactly reproducible (see §3).
- **Neither inflation is a serving risk.** The congestion bug is consistent between training and serving (the frozen stack is used unchanged), and the independent evidence is favourable: v5 (same features/stack on Jan/Jul 2026) scored 392.59 s, −13.6% versus the 2025 season-matched analog (454.25 s). §4d further shows the final Jan/Jul predictions reproduce v5's own Jan/Jul predictions with a negligible mean shift (≤ 0.75%) and a ~7–9 s median per-row difference (the ±12 s dither).
- **Conclusion:** flag recorded, investigation complete, **no block on submission**. The drift signal is a diagnostic degeneracy (disjoint calendar days + high-resolution date-correlated covariates), amplified by freezing a units bug in the congestion code — not evidence of harmful distribution shift in the predictive features.
- **Out of scope, recorded for the repo/paper (Task 21/23):** the congestion window units bug should be disclosed as a latent issue. It must NOT be fixed without retraining (a serve-only fix would break train/serve consistency).

## 7. v6 ID overlap and validation submission

Jan/Jul subset of the final side table (n=344,841) vs `submitting.parquet` (n=344,841): set equality **True**. The other two final months (Feb/Jun 2026) are not part of `submitting.parquet` and are excluded from v6.

**v6 leaderboard probe (recorded post-run).** `submissions/thoughtful-glove_v6.parquet` (SHA256 `5805cdd9397201ff24680cb00403d7637dd34caeb1c0cc5d4fb6870925959525`, 4,707,521 bytes) = the v1 predictions restricted to Jan/Jul, uploaded 2026-10-11 00:22:15 UTC. Result: status Succeeded, 344,841/344,841 pairs, RMSE **393.2793 s** vs v5's 392.5911 s = **+0.6882 s (+0.18%)** — inside the 2% gate: **no serving regression**.

## 8. Notes for downstream tasks

- Task 19: fallback artifact = `submissions/final/thoughtful-glove_final_v1.parquet` (SHA256 in §1). Base-row predictions are the blend output; any hard-row treatment must leave them byte-identical.
- Task 20: copy the selected artifact to `submissions/final/thoughtful-glove_final.parquet` under the official name.
- Task 21/23: this report is the evidence for the final-phase serving claim.
