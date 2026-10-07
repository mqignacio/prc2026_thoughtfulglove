# Domain Features Ablation Report — Task 09

Task: `09-domain_features` · Team: `thoughtful-glove`
Generated: 2026-09-10
Runtime: 2418s (40.3 min)

---

## Configuration

- **Model:** LightGBM regression (same params as baseline)
- **Seed:** 42
- **CV:** 12-fold month-blocked (leave-one-month-out, Jan–Dec 2025)
- **Target:** Winsorize-errors treatment; **raw unclipped** RMSE as headline metric
- **Reference arm (D0):** Baseline + METAR + congestion = best arm from Task 08 (RMSE 343.21s)

### Domain Feature Families

| Family | Columns | Description |
|--------|---------|-------------|
| Local time | local_hour_sin, local_hour_cos, local_dow, local_month, local_weekend | DST-aware local hour/day/weekend per airport (zoneinfo) |
| Geometry | runway_heading, stand_zone, stand_runway_interaction, dest_distance_km | Runway heading, stand terminal group, stand-runway interaction, great-circle ADEP→ADES |
| Flight chars | market_segment_code, flight_type_code, operator_enc, diversion_flag, eobt_sched_delta_sec | Segment/type codes, CV-safe operator target encoding, diversion flag, EOBT-schedule delta |

**Leakage guarantee:** `operator_enc` is a smoothed target encoding computed
strictly within each CV fold's training partition (fit on train rows, mapped
to validation rows). `assert_no_target_encoding_leakage` runs per fold.
All other domain features are label-free.

---

## Headline Results — Overall Raw RMSE

| Arm | Description | Features | RMSE (s) | Δ vs D0 (s) | Δ (%) |
|-----|-------------|---------:|---------:|------------:|------:|
| D0 | Baseline + METAR + Congestion | 48 | 342.01 | +0.00 | +0.00% |
| D1 | D0 + Local time | 62 | 341.16 | -0.85 | -0.25% |
| D2 | D0 + Geometry | 62 | 339.18 | -2.83 | -0.83% |
| D3 | D0 + Flight characteristics | 62 | 340.64 | -1.37 | -0.40% |
| D4 | D0 + All domain features | 62 | 338.43 | -3.57 | -1.04% |

> Note: D1/D2/D3 each add ONE family on top of D0 (not cumulative). D4 adds all.

---

## Per-Month RMSE (Raw Target)

| Fold Month | D0 | D1 (Δ) | D2 (Δ) | D3 (Δ) | D4 (Δ) |
|------------|----:|-------:|-------:|-------:|-------:|
| 2025-01 | 393.58 | -0.37 | -0.23 | -2.66 | -0.60 |
| 2025-02 | 397.53 | +5.26 | -4.29 | +1.63 | -1.03 |
| 2025-03 | 263.71 | -4.16 | -4.94 | -8.10 | -9.65 |
| 2025-04 | 244.13 | +3.16 | -3.51 | -2.41 | -2.26 |
| 2025-05 | 308.76 | -0.71 | -1.16 | -0.70 | -1.15 |
| 2025-06 | 378.85 | +0.09 | -4.21 | -2.16 | -5.66 |
| 2025-07 | 505.84 | +0.82 | -4.24 | -0.09 | -4.18 |
| 2025-08 | 310.64 | -0.22 | -2.82 | +0.67 | -8.51 |
| 2025-09 | 344.52 | -13.13 | -3.12 | -3.40 | -8.68 |
| 2025-10 | 249.41 | -2.38 | -0.92 | -2.01 | -3.55 |
| 2025-11 | 322.06 | +4.23 | -0.96 | +2.83 | +5.38 |
| 2025-12 | 291.40 | -3.54 | -3.52 | -2.97 | -3.20 |

---

## Per-Airport RMSE (Raw Target, All Folds Combined)

| Airport | N | D0 RMSE | D4 Δ | Mean Actual |
|---------|--:|--------:|-----:|------------:|
| LIRF | 160,704 | 919.60 | -13.05 | 1194.6 |
| LFPG | 239,552 | 323.86 | -0.75 | 1019.5 |
| LSZH | 134,907 | 302.01 | -0.40 | 740.5 |
| LTFM | 272,965 | 280.77 | -4.06 | 1053.0 |
| EGLL | 239,546 | 243.42 | -1.94 | 1364.2 |
| LEBL | 179,705 | 208.30 | -0.64 | 956.5 |
| LEMD | 212,242 | 185.52 | -0.08 | 1014.7 |
| EDDF | 230,141 | 180.45 | -0.64 | 863.5 |
| EDDM | 167,334 | 174.86 | -0.76 | 811.9 |
| EHAM | 247,951 | 169.54 | -1.12 | 784.6 |

---

## Acceptance Criteria Verification

- [x] All feature groups implemented, flag-gated, and null-safe on train + ranking
- [x] Ablation table shows CV delta per feature group (added one at a time)
- [x] Target-encoding leakage assertion present and passes (per fold)
