# Baseline Results — LightGBM Month-Blocked CV

Task: `04-baseline_pipeline` · Team: `thoughtful-glove`
Status: **UPDATED (Task 2.2b — winsorize-errors target treatment + unclipped predictions)**
Generated: 2026-09-09
Runtime: 242s (4.0 min)

---

## Configuration

- **Model:** LightGBM regression
- **Features:** 21 (temporal + anchors + categoricals, no congestion)
- **Training rows:** 2,085,047 DEP movements (Jan–Dec 2025)
- **Target:** TAXITIME_SEC_mvt (winsorize-errors treatment: negatives < 1s → 1s, > 86400s → airport-specific P99; **raw unclipped** used for the headline CV metric)
- **Categorical vocabulary:** persisted to `models/baseline_lgbm_vocab.json` (shared train/score mapping)
- **Seed:** 42
- **Params:** lr=0.05, leaves=63, subsample=0.8, colsample=0.8
- **Max rounds:** 500 (early stopping @ 50 rounds)
- **Prediction clip:** floor at 1s, no upper bound (relaxed from [0, 7200]s)
- **Task 2.2b finding:** E-winsorize-errors target treatment **344.94s** vs baseline 365.07s (−5.51%)

---

## Headline CV Metrics (selection signal)

| Metric | RMSE (s) | MAE (s) |
|--------|---------:|--------:|
| **Raw** unclipped target (competition metric) | **344.07** | **154.01** |
| Clipped target [1, 14400s] (informational) | 285.84 | 153.59 |
| Constant-baseline (raw) — predict overall training mean for every row | 546.36 | — |

> **Raw-target RMSE is the headline model-selection metric.** The clipped metric is retained for continuity.
> The constant baseline makes the raw RMSE interpretable: the model's RMSE of 344.07s is 37.0% vs the trivial predictor (546.36s).

---

## CV Results — Per-Month

| Fold Month | N Val | Best Iter | RMSE (clipped) | MAE (clipped) | RMSE (raw) | MAE (raw) |
|------------|------:|----------:|----------------:|--------------:|-----------:|----------:|
| 2025-01 | 153,706 | 485 | 373.72 | 151.93 | 391.62 | 152.09 |
| 2025-02 | 143,732 | 500 | 299.14 | 153.45 | 414.09 | 154.09 |
| 2025-03 | 164,449 | 138 | 236.74 | 146.37 | 263.66 | 146.53 |
| 2025-04 | 175,288 | 235 | 245.70 | 143.76 | 245.70 | 143.76 |
| 2025-05 | 185,202 | 497 | 239.11 | 147.02 | 312.77 | 147.48 |
| 2025-06 | 183,142 | 376 | 283.94 | 155.19 | 373.50 | 155.90 |
| 2025-07 | 190,713 | 500 | 352.67 | 174.30 | 507.26 | 175.90 |
| 2025-08 | 191,182 | 491 | 306.71 | 158.12 | 306.72 | 158.12 |
| 2025-09 | 183,950 | 250 | 308.83 | 158.86 | 343.01 | 159.12 |
| 2025-10 | 185,674 | 500 | 249.10 | 151.10 | 249.10 | 151.10 |
| 2025-11 | 162,332 | 493 | 235.53 | 147.39 | 329.12 | 148.02 |
| 2025-12 | 165,677 | 193 | 258.77 | 152.35 | 302.23 | 152.64 |

**Season-matched (Jan+Jul) raw RMSE: 459.26s** — the closest CV analog to the leaderboard scoring set. Compare with the overall raw RMSE (344.07s) to see the season-mix effect (gap = 115.19s).

---

## Per-Airport RMSE — Worst Fold (2025-01)

Worst fold by RMSE: **2025-01**

| Airport | N | Mean Actual (s) | RMSE (s) | MAE (s) |
|---------|--:|----------------:|---------:|--------:|
| LFPG | 17,900 | 1079.8 | 804.94 | 185.17 |
| LIRF | 11,317 | 1089.2 | 603.52 | 215.08 |
| LTFM | 21,368 | 1041.1 | 249.05 | 170.73 |
| EGLL | 19,420 | 1330.1 | 248.84 | 173.83 |
| LSZH | 9,735 | 775.1 | 228.52 | 134.84 |
| EHAM | 18,460 | 811.2 | 224.06 | 128.32 |
| EDDM | 11,266 | 870.2 | 194.89 | 129.59 |
| EDDF | 15,496 | 884.8 | 188.70 | 130.32 |
| LEMD | 16,510 | 985.8 | 172.58 | 120.17 |
| LEBL | 12,234 | 887.4 | 159.85 | 117.26 |

---

## Artifacts

| File | Description |
|------|-------------|
| `models/baseline_lgbm.txt` | Trained LightGBM booster (full 2025 data) |
| `models/baseline_lgbm_vocab.json` | Categorical vocabulary (shared train/score mapping) |
| `data/processed/oof_predictions.parquet` | Out-of-fold predictions (per MVT_ID, incl. `actual_raw`) |
| `data/processed/ranking_predictions.parquet` | Ranking DEP predictions from the full model |

## Key Observations

- Best fold (raw): 2025-04 (RMSE 245.70s)
- Worst fold (raw): 2025-07 (RMSE 507.26s)
- Inter-fold (raw) RMSE spread: 261.56s — indicates high temporal sensitivity
- Near-oracle anchor RMSE floor: ~384s (MVT_TIME − AOBT_3 alone). Overall raw RMSE of 344.07s beats this floor.
- Raw RMSE vs season-matched (Jan+Jul) 459.26s: above the season comparator by 115.19s.
- Worst airport in worst fold: LFPG (RMSE 804.94s) vs best: LEBL (RMSE 159.85s)
