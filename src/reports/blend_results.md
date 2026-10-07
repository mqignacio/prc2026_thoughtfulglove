# Blend Results — Task 3.4

Task: `11-blend_submission` · Team: `thoughtful-glove`
Generated: 2026-10-07 · Seed: 42

---

## Configuration

- **Feature set:** D4 (baseline + METAR + congestion + domain, 62 features)
- **Target:** E-winsorize-errors
- **CV:** 12-fold month-blocked (leave-one-month-out, Jan–Dec 2025)
- **Metric:** raw unclipped RMSE (competition metric)
- **Weights fit on OOF predictions only — never leaderboard feedback.**

## Components (single-model OOF raw RMSE)

| Component | Family | Mode | OOF RMSE (s) | Median best_iter |
|-----------|--------|------|-------------:|-----------------:|
| tuned_lgbm | lgbm | global | 337.16 | 466 |
| per_airport_lgbm | lgbm | per_airport | 335.01 | EDDF:372, EDDM:340, EGLL:312, EHAM:337, LEBL:200, LEMD:188, LFPG:262, LIRF:170, LSZH:190, LTFM:414 |
| xgb | xgb | global | 352.22 | 499 |
| cat | cat | global | 362.02 | 299 |

## Residual Correlation (OOF errors)

| | tuned_lgbm | per_airport_lgbm | xgb | cat |
|---|---:|---:|---:|---:|
| tuned_lgbm | 1.000 | 0.965 | 0.947 | 0.923 |
| per_airport_lgbm | 0.965 | 1.000 | 0.935 | 0.917 |
| xgb | 0.947 | 0.935 | 1.000 | 0.969 |
| cat | 0.923 | 0.917 | 0.969 | 1.000 |

## Blend Search (simplex-constrained, all subsets)

| Rank | Subset | OOF RMSE (s) | Weights |
|-----:|--------|-------------:|---------|
| 1 | tuned_lgbm + per_airport_lgbm | 333.0208 | tuned_lgbm=0.409, per_airport_lgbm=0.591 |
| 2 | tuned_lgbm + per_airport_lgbm + xgb | 333.0208 | tuned_lgbm=0.409, per_airport_lgbm=0.591, xgb=0.000 |
| 3 | tuned_lgbm + per_airport_lgbm + cat | 333.0208 | tuned_lgbm=0.409, per_airport_lgbm=0.591, cat=0.000 |
| 4 | tuned_lgbm + per_airport_lgbm + xgb + cat | 333.0208 | tuned_lgbm=0.409, per_airport_lgbm=0.591, xgb=0.000, cat=0.000 |
| 5 | per_airport_lgbm + xgb | 334.6715 | per_airport_lgbm=0.879, xgb=0.121 |
| 6 | per_airport_lgbm + xgb + cat | 334.6715 | per_airport_lgbm=0.879, xgb=0.121, cat=0.000 |
| 7 | per_airport_lgbm + cat | 334.9319 | per_airport_lgbm=0.949, cat=0.051 |
| 8 | per_airport_lgbm | 335.0146 | per_airport_lgbm=1.000 |
| 9 | tuned_lgbm + xgb | 336.9924 | tuned_lgbm=0.907, xgb=0.093 |
| 10 | tuned_lgbm + xgb + cat | 336.9924 | tuned_lgbm=0.907, xgb=0.093, cat=0.000 |

- **Simple average (all 4):** 339.0508 s
- **Best single component:** per_airport_lgbm (335.0146 s)
- **Selected blend:** tuned_lgbm + per_airport_lgbm (333.0208 s)
- **Blend beats best single:** True (gain 1.9938 s)

## Decision

Weighted blend of **2 components** selected (OOF 333.02 s ≤ best single 335.01 s).

Weights: `tuned_lgbm` = 0.4094, `per_airport_lgbm` = 0.5906
