# Model Comparison Report — Task 3.3

Task: `10-model_tuning` · Team: `thoughtful-glove`
Generated: 2026-09-10 · Seed: 42

---

## Configuration

- **Feature set:** D4 (baseline + METAR + congestion + domain, 62 features)
- **Target:** E-winsorize-errors (Task 05b winner)
- **CV:** 12-fold month-blocked (leave-one-month-out, Jan–Dec 2025)
- **Metric:** raw unclipped RMSE (competition metric)
- **Families:** LightGBM / XGBoost / CatBoost, family-equivalent default params

## Overall Raw RMSE

| Family | RMSE (s) | MAE (s) |
|--------|---------:|--------:|
| lgbm | 338.56 | 150.97 |
| xgb | 355.00 | 157.85 |
| cat | 362.24 | 169.30 |

## Per-Month Raw RMSE

| Fold Month | lgbm | xgb | cat |
|------------|-----:|----:|----:|
| 2025-01 | 394.10 | 410.83 | 415.73 |
| 2025-02 | 395.33 | 424.99 | 433.01 |
| 2025-03 | 252.15 | 271.63 | 284.80 |
| 2025-04 | 243.65 | 253.16 | 267.22 |
| 2025-05 | 307.12 | 315.24 | 327.40 |
| 2025-06 | 373.80 | 390.81 | 395.50 |
| 2025-07 | 504.83 | 528.24 | 522.16 |
| 2025-08 | 302.49 | 329.70 | 335.03 |
| 2025-09 | 335.55 | 349.32 | 355.58 |
| 2025-10 | 245.96 | 250.48 | 266.13 |
| 2025-11 | 323.55 | 333.76 | 344.21 |
| 2025-12 | 287.56 | 300.38 | 316.83 |

## Per-Airport Raw RMSE

| Airport | N | lgbm | xgb | cat | Mean Actual |
|---------|--:|-----:|----:|----:|------------:|
| LIRF | 160,704 | 907.06 | 963.55 | 960.25 | 1194.6 |
| LFPG | 239,552 | 322.99 | 329.45 | 337.29 | 1019.5 |
| LSZH | 134,907 | 301.54 | 308.32 | 318.17 | 740.5 |
| LTFM | 272,965 | 276.95 | 289.37 | 309.14 | 1053.0 |
| EGLL | 239,546 | 241.79 | 248.14 | 259.32 | 1364.2 |
| LEBL | 179,705 | 207.58 | 213.90 | 226.26 | 956.5 |
| LEMD | 212,242 | 185.20 | 191.48 | 204.91 | 1014.7 |
| EDDF | 230,141 | 180.05 | 186.59 | 195.55 | 863.5 |
| EDDM | 167,334 | 174.08 | 179.19 | 189.82 | 811.9 |
| EHAM | 247,951 | 168.21 | 176.82 | 194.40 | 784.6 |

---

## Conclusion

Best family under identical CV: **lgbm** (338.56s).

## Global vs Per-Airport (LightGBM, full 12-fold)

| Model | Raw RMSE (s) |
|-------|-------------:|
| Global (D4, airport features) | 338.56 |
| 10 per-airport models (combined OOF) | 335.98 |

_Per-airport combined RMSE is computed by stitching each airport's OOF predictions and scoring against raw target._
