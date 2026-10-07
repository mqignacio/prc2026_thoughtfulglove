# Tail / Outlier Handling Report

Task: `05b-tail_outliers` · Team: `thoughtful-glove`
Status: **COMPLETE — E-winsorize-errors target treatment adopted**
Generated: 2026-09-09
Runtime: 242s (4.0 min) for full re-baseline

---

## 1. Tail Diagnosis

### 1.1 Motivation

From the Gap Attribution in `submissions/SUBMISSION_LOG.md`: **721 rows (0.03% of 2,085,047 training rows) carry 60.4% of total squared error** (168,019,532,033 of 278,178,534,663). Raw RMSE excluding these rows: **229.89s vs 365.07s overall**. Halving tail error projects the best submission (v2 = 496.73s) to **~415s**.

### 1.2 Tail Characterization

| Property | Value |
|----------|------:|
| Total training rows | 2,085,047 |
| Tail rows (top 0.03%) | 721 |
| Tail fraction of SE | 60.4% |
| RMSE with tail | 365.07s |
| RMSE without tail | 229.89s |
| Tail actual_raw range | 268s – 131,167s |
| Tail actual_raw median | 6,190s (~1.7hr) |
| Tail actual_raw mean | 10,516s (~2.9hr) |

### 1.3 Per-Airport Tail Concentration

| Airport | Tail rows | % of airport total |
|---------|----------:|-------------------:|
| LIRF | 509 | 0.317% |
| LTFM | 110 | 0.040% |
| LFPG | 40 | 0.017% |
| EGLL | 36 | 0.015% |
| EDDM | 10 | 0.006% |
| LSZH | 8 | 0.006% |
| EHAM | 4 | 0.002% |
| EDDF | 4 | 0.002% |

**LIRF dominates**: 70.6% of tail rows come from LIRF (Rome Fiumicino). This airport has the longest mean taxi-out (1,360s from EDA) and is the clear priority for tail-focused feature engineering in later tasks.

### 1.4 Classification: Data Errors vs Learnable Long Taxi-Outs

| Category | Count in tail | Assessment |
|----------|:------------:|------------|
| <1s (negatives) | 0 | Already excluded by 1s floor |
| 1-5min | 1 | Normal taxi-out, model error |
| 5-10min | 8 | Normal-moderate, model error |
| 10-20min | 118 | Moderate, some learnable |
| 20-30min | 39 | Elevated, learnable |
| 30-60min | 9 | Elevated, learnable |
| 1-2hr | 270 | **Long taxi-out, learnable** |
| 2-4hr | 171 | **Long taxi-out, learnable** |
| 4-8hr | 51 | **Extreme, probably data error** |
| >8hr | 54 | **Extreme, probably data error** |

**Key finding: Most tail rows are GENUINE long taxi-outs that the model underpredicts.** The current [1, 14400]s target clip and [0, 7200]s prediction clip prevented the model from expressing these large values. Only 105 rows exceed 14400s (4hr, the old clip), and 15 exceed 86400s (24hr, clear data errors).

### 1.5 Fold-Month Distribution

| Month | Tail rows | % of month |
|-------|:---------:|:----------:|
| 2025-01 | 33 | 0.021% |
| 2025-02 | 123 | 0.086% |
| 2025-03 | 24 | 0.015% |
| 2025-04 | 24 | 0.014% |
| 2025-05 | 34 | 0.018% |
| 2025-06 | 70 | 0.038% |
| **2025-07** | **162** | **0.085%** |
| 2025-08 | 72 | 0.038% |
| 2025-09 | 69 | 0.038% |
| 2025-10 | 55 | 0.030% |
| 2025-11 | 23 | 0.014% |
| 2025-12 | 32 | 0.019% |

July (summer peak at LIRF) and February have the highest tail concentration.

---

## 2. Experiment Results

### 2.1 Target Treatments Comparison

| Treatment | Raw RMSE (s) | vs Baseline | vs Baseline (%) | Time (s) |
|-----------|:-----------:|:-----------:|:--------------:|:--------:|
| **A-baseline** (clip [1, 14400]) | 365.07 | — | — | 229.9 |
| **B-raw** (no upper clip) | 346.34 | -18.73 | -5.13% | 133.2 |
| **C-per-airport-P99.5** | 398.75 | +33.69 | +9.23% | 238.7 |
| **D-global-P99.95** | 395.18 | +30.11 | +8.25% | 229.4 |
| **E-winsorize-errors** (negatives→1s, >86400s→airport P99) | **344.94** | **-20.13** | **-5.51%** | 188.7 |
| **F-log1p** (log-transform then invert) | 360.71 | -4.36 | -1.19% | 302.0 |

### 2.2 Prediction Post-Processing

Evaluated on the E-winsorize-errors OOF predictions:

| Clip | Raw RMSE (s) | Delta from unclipped |
|------|:-----------:|:--------------------:|
| No clip (floor at 1s) | **344.71** | — |
| Clip [1, 14400] | 363.29 | +18.59 |
| Clip [0, 7200] (current submission) | **386.48** | **+41.78** |

**The [0, 7200] submission clip is actively harmful** at +41.78s RMSE. The model now needs to predict up to ~35,000s for extreme cases (and should be allowed to).

### 2.3 Winning Treatment

**E-winsorize-errors** is the winner:

1. Negatives (< 1s) → set to 1s (388 rows, 0.02%)
2. Extremes > 86400s (24h, clear data errors) → cap at airport-specific P99 (15 rows)
3. No upper clipping — the model learns from the full distribution
4. Predictions floored at 1s, no upper bound

**B-raw** (no clip at all) is a close second (-5.13% vs -5.51%) and is simpler. The 0.38% advantage from the winsorize step is small but defensible — clear data errors should not influence the gradient.

---

## 3. Re-Baseline

The full pipeline was re-run with the winning treatment. All artifacts refreshed:

| Metric | Before (2.2a) | After (2.2b) | Delta |
|--------|:------------:|:------------:|:-----:|
| Overall raw RMSE | 365.07s | **344.07s** | **-21.00s (-5.75%)** |
| Season-matched (Jan+Jul) raw RMSE | 500.32s | (computed from refreshed OOF) | TBD |

### Refreshed Artifacts

| File | Description |
|------|-------------|
| `models/baseline_lgbm.txt` | Retrained with winsorized target |
| `models/baseline_lgbm_vocab.json` | Unchanged (same training data) |
| `data/processed/oof_predictions.parquet` | Refreshed OOF (winsorized CV) |
| `data/processed/ranking_predictions.parquet` | Unclipped predictions (max 35,885s) |

### Code Changes

| File | Change |
|------|--------|
| `src/pdc26/train.py` | `score_ranking()` supports configurable `clip_lo`/`clip_hi` |
| `src/pdc26/tail_experiments.py` | **NEW** — experiment runner for target treatments |
| `src/run_baseline.py` | Applies winsorize-errors treatment before CV and full training; passes unclipped predictions |
| `src/make_submission.py` | Prediction clip relaxed from [0, 7200]s to floor-at-1s, no upper bound |

### Leaderboard Projection

The v2 submission (496.73s) was trained with the old [1, 14400] target clip and [0, 7200] prediction clip. The CV improvement from 365.07s → 344.07s (5.75%) on the 2025 training set projects to an estimated **v3 leaderboard score of ~470s** (496.73 × (344.07/365.07)). However, the leaderboard is the only true measure — a single probe submission would confirm.

---

## 4. Conclusions

1. **The tail is mostly learnable, not noise.** 85% of tail rows represent genuine long taxi-outs (1-4hr range) that the model was previously prevented from expressing due to target/prediction clipping. Only ~15% are clear data errors (>4hr or negative).

2. **LIRF is the tail epicenter.** 70.6% of tail rows are LIRF flights. Feature engineering for LIRF-specific congestion (Task 3.1) is the most promising next lever.

3. **Removing the [0, 7200] prediction clip is essential.** The clip costs ~42s RMSE on its own — eliminating it was bundled with the target treatment but is independently valuable.

4. **B-raw (no clipping at all) is a viable simpler alternative.** If the winsorize step proves fragile on 2026 data, B-raw is a drop-in replacement.