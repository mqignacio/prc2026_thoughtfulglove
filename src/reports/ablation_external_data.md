# External Data Ablation Report — Task 07

Task: `07-ablation_feasibility` · Team: `thoughtful-glove`
Generated: 2026-09-10
Runtime: 889s (14.8 min)

---

## Configuration

- **Model:** LightGBM regression (same params as baseline)
- **Seed:** 42
- **CV:** 12-fold month-blocked (leave-one-month-out, Jan–Dec 2025)
- **Target:** Winsorize-errors treatment; **raw unclipped** RMSE as headline metric
- **Params:** lr=0.05, leaves=63, subsample=0.8, colsample=0.8, max_rounds=500, early_stop=50
- **External data sources:**
  - METAR: Iowa State Mesonet ASOS archive (277,489 obs, 10 airports, 100% joinability)
  - Holidays: `holidays` v0.104, national public holidays (5,770 rows, 8 countries)

---

## Headline Results — Overall Raw RMSE

| Arm | Description | Features | RMSE (s) | Δ vs A0 (s) | Δ (%) |
|-----|-------------|---------:|---------:|------------:|------:|
| A0 | Baseline (no external) | 21 | 344.82 | +0.00 | +0.00% |
| A1 | Baseline + Holidays | 24 | 345.20 | +0.39 | +0.11% |
| A2 | Baseline + METAR | 32 | 343.27 | -1.55 | -0.45% |
| A3 | Baseline + Both | 35 | 343.24 | -1.57 | -0.46% |

---

## Per-Month RMSE (Raw Target)

| Fold Month | A0 | A1 (Δ) | A2 (Δ) | A3 (Δ) |
|------------|----:|-------:|-------:|-------:|
| 2025-01 | 392.70 | -0.76 | -2.51 | -0.68 |
| 2025-02 | 412.59 | +2.86 | -5.95 | -7.18 |
| 2025-03 | 265.29 | -1.11 | -1.46 | -2.73 |
| 2025-04 | 244.82 | +0.89 | -2.11 | -1.24 |
| 2025-05 | 311.99 | +0.69 | -0.89 | -2.28 |
| 2025-06 | 373.48 | -1.64 | +2.60 | +1.50 |
| 2025-07 | 513.03 | +0.73 | +1.21 | +2.47 |
| 2025-08 | 307.10 | -0.04 | +2.53 | +2.18 |
| 2025-09 | 342.25 | +0.37 | -0.63 | -0.34 |
| 2025-10 | 250.51 | -0.67 | -1.01 | -1.35 |
| 2025-11 | 329.75 | +4.09 | -7.29 | -6.91 |
| 2025-12 | 300.97 | -0.92 | -7.67 | -7.85 |

---

## Per-Airport RMSE (Raw Target, All Folds Combined)

| Airport | N | A0 RMSE | A1 Δ | A2 Δ | A3 Δ | Mean Actual |
|---------|--:|--------:|-----:|-----:|-----:|------------:|
| LIRF | 160,704 | 921.71 | +2.24 | -1.08 | -1.83 | 1194.6 |
| LFPG | 239,552 | 327.45 | -0.60 | -3.22 | -3.09 | 1019.5 |
| LSZH | 134,907 | 303.29 | -0.05 | -0.10 | +0.11 | 740.5 |
| LTFM | 272,965 | 289.38 | +0.64 | -3.79 | -3.35 | 1053.0 |
| EGLL | 239,546 | 244.22 | -0.24 | -0.07 | -0.04 | 1364.2 |
| LEBL | 179,705 | 212.86 | -0.30 | -0.60 | -0.16 | 956.5 |
| LEMD | 212,242 | 186.83 | -0.54 | -1.11 | -0.74 | 1014.7 |
| EDDF | 230,141 | 182.69 | +0.20 | -0.97 | -0.94 | 863.5 |
| EDDM | 167,334 | 179.63 | -0.56 | -3.40 | -3.36 | 811.9 |
| EHAM | 247,951 | 174.06 | -0.03 | -4.36 | -4.22 | 784.6 |

---

## Coverage Context

From `external_data_coverage.md`:
- **METAR join coverage:** 100.0% (2,429,661/2,429,888 movements matched within 60 min)
- **Median gap:** 15.0 minutes (P95: 28.5 min)
- **Holiday coverage:** 5,770 airport-date rows across 8 countries

> A null METAR delta alongside 100% join coverage means "weather features don't help",
> NOT "coverage was insufficient to measure".

---

## OpenSky API Go/No-Go Decision

### Criterion (a): Rules Permit Supplementary Open Data

**VERIFIED.** The official Eligibility page (https://ansperformance.eu/study/data-challenge/dc2026/eligibility.html) states:

> - "All used external datasets are openly accessible/usable and documented."
> - "All additional datasets used are openly available under an open source license."

External/open data is explicitly permitted and expected. Our METAR (Iowa State Mesonet, open data) and holiday calendar (`holidays` package, MIT license) already satisfy these conditions. OpenSky Network data is similarly open-access (CC BY-SA 4.0 for REST API data).

### Criterion (b): Credit Feasibility

- **Daily allowance:** 4,000 credits/day (Standard tier)
- **Ranking window:** Jan 2026 + Jul 2026 (~60 days total)
- **Enrichment target:** Historical ADS-B state vectors for 10 airports during ranking months
- **Estimated queries:** ~10 airports × 60 days × ~1 query/day (bulk state-vector endpoint) ≈ 600 queries total
- **Credit cost:** Well within daily allowance (600 queries over 60 days = ~10 queries/day vs 4,000/day limit)
- **Verdict:** Credits are sufficient for a useful enrichment.

### Criterion (c): Expected Headroom

METAR weather features show a measurable improvement (A2 Δ=-1.55s, A3 Δ=-1.57s). Per-airport RMSE spread of 747.6s (range 174.1–921.7s) indicates substantial airport-specific variance that weather/operational context could further explain. OpenSky ADS-B data (real-time traffic density, ground movement patterns) can complement METAR with direct operational congestion signals at finer temporal resolution. Credits are sufficient (estimated ~10 queries/day vs 4,000/day allowance).

### Verdict: **GO**

---

## Acceptance Criteria Verification

- [x] A0 reproduces baseline RMSE: 344.82s vs baseline 344.07s (Δ=+0.75s, +0.22% — PASS within ±1%)
- [x] All four ablation arms reported with overall + per-airport numbers
- [x] GO/NO-GO verdict present with all three criteria addressed
