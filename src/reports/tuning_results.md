# Hyperparameter Tuning Report — Task 3.3

Task: `10-model_tuning` · Team: `thoughtful-glove`
Generated: 2026-10-02 · Seed: 42

---

## Configuration

- **Feature set:** D4 (baseline + METAR + congestion + domain, 62 features)
- **Target:** E-winsorize-errors (Task 05b winner)
- **Metric:** raw unclipped RMSE (competition metric)
- **Search:** Optuna (TPE), 4 representative months
- **Validation:** full 12-fold month-blocked CV for finalists

## Optuna Search Results

- Subsample RMSE (4 rep. months) = 328.93s

| Hyperparameter | Value |
|----------------|------:|
| num_leaves | 221 |
| learning_rate | 0.0623 |
| min_child_samples | 12 |
| feature_fraction | 0.919 |
| bagging_fraction | 0.823 |

## Finalist vs Baseline (full 12-fold)

| Config | Full-CV Raw RMSE (s) |
|--------|---------------------:|
| Baseline LGBM (default params) | 338.56 |
| Tuned LGBM | 337.54 |
| Δ | -1.02 |

## Outlier Re-Check (retuned scope)

| Target treatment | Full-CV Raw RMSE (s) |
|------------------|---------------------:|
| E-winsorize-errors (current) | 337.54 |
| B-raw (negatives → 0 only) | 338.86 |

Verdict: E-winsorize-errors still holds after tuning (winsorize ≤ raw).
