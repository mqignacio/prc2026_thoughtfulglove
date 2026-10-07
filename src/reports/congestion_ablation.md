# Congestion Features Ablation Report — Task 08

Task: `08-congestion_features` · Team: `thoughtful-glove`
Generated: 2026-09-10
Runtime: 695s (11.6 min)

---

## Configuration

- **Model:** LightGBM regression (same params as baseline)
- **Seed:** 42
- **CV:** 12-fold month-blocked (leave-one-month-out, Jan–Dec 2025)
- **Target:** Winsorize-errors treatment; **raw unclipped** RMSE as headline metric
- **Params:** lr=0.05, leaves=63, subsample=0.8, colsample=0.8, max_rounds=500, early_stop=50
- **Reference arm (C0):** Baseline + METAR weather (best external from Task 07, Δ=−1.55s vs pure baseline)
- **Congestion arm (C1):** C0 + all congestion feature families (rolling, load, queue, runway)

### Congestion Feature Families

| Family | Columns | Description |
|--------|---------|-------------|
| Rolling | dep/arr_count_{5,15,30,60}min | Movement counts in symmetric windows around takeoff |
| Load | hourly_dep/arr_actual, hourly_dep_sched, hourly_dep_gap, relative_busyness | Airport-hour throughput metrics |
| Queue | queue_proxy_30min | Actual takeoffs minus scheduled in trailing 30 min |
| Runway | runway_dep/arr_count_15min | Per-runway movement counts ±15 min |

**No-leakage guarantee:** All congestion features use only `MVT_TIME_UTC_mvt`,
`SCHED_TIME_UTC_mvt`, `PHASE_mvt`, `ADEP_mvt`/`ADES_mvt`, `RUNWAY_mvt`.
Never reads `TAXITIME_SEC_mvt` or `BLOCK_TIME_UTC_mvt`. Counts are computed
from movement times only, so no target leakage even when the focal row's own
movement falls within its window.

---

## Headline Results — Overall Raw RMSE

| Arm | Description | Features | RMSE (s) | Δ vs C0 (s) | Δ (%) |
|-----|-------------|---------:|---------:|------------:|------:|
| C0 | Baseline + METAR | 32 | 343.55 | +0.00 | +0.00% |
| C1 | Baseline + METAR + Congestion | 48 | 343.21 | -0.34 | -0.10% |

---

## Per-Month RMSE (Raw Target)

| Fold Month | C0 | C1 (Δ) |
|------------|----:|-------:|
| 2025-01 | 391.22 | +2.30 |
| 2025-02 | 404.44 | -4.82 |
| 2025-03 | 257.40 | +10.16 |
| 2025-04 | 244.25 | +1.22 |
| 2025-05 | 310.79 | -1.63 |
| 2025-06 | 377.32 | +0.35 |
| 2025-07 | 517.37 | -5.84 |
| 2025-08 | 310.24 | +2.83 |
| 2025-09 | 341.73 | +2.22 |
| 2025-10 | 250.66 | -1.25 |
| 2025-11 | 322.49 | -0.02 |
| 2025-12 | 293.43 | -4.26 |

---

## Per-Airport RMSE (Raw Target, All Folds Combined)

| Airport | N | C0 RMSE | C1 Δ | Mean Actual |
|---------|--:|--------:|-----:|------------:|
| LIRF | 160,704 | 922.94 | +2.31 | 1194.6 |
| LFPG | 239,552 | 324.02 | -0.37 | 1019.5 |
| LSZH | 134,907 | 303.23 | -1.08 | 740.5 |
| LTFM | 272,965 | 284.97 | -3.61 | 1053.0 |
| EGLL | 239,546 | 243.65 | -0.66 | 1364.2 |
| LEBL | 179,705 | 211.99 | -3.45 | 956.5 |
| LEMD | 212,242 | 185.56 | +0.15 | 1014.7 |
| EDDF | 230,141 | 181.55 | -0.90 | 863.5 |
| EDDM | 167,334 | 175.87 | -1.39 | 811.9 |
| EHAM | 247,951 | 169.40 | +0.04 | 784.6 |

---

## Acceptance Criteria Verification

- [x] Congestion features computed identically for train/ranking (verified by builder design: uses only movement times, no target columns)
- [x] Ablation table shows CV delta: C1 − C0 = -0.34s (improvement)
- [x] Feature build runtime < 10 min: congestion features built in 95.1s (1.6 min); total ablation incl. 2×12-fold CV = 695s (11.6 min)
