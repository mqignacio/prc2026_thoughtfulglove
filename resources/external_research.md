# External Research Summary — PRC-OpenSky Data Challenge 2026

Status: draft (planning session, 2026-09-08). Sources verified via official pages and web search on 2026-09-08.

## 1. Challenge Facts (official sources)

| Item | Value | Source |
|---|---|---|
| Task | Predict taxi-out time (`TAXITIME_SEC_mvt`) of departing flights at 10 major European airports | [DC2026 main page](https://ansperformance.eu/study/data-challenge/dc2026/) |
| Scoring | RMSE on departures in Jan and Jul 2026 (ranking dataset) | Main page |
| Timeline | 2026-09-01 to 2026-10-11 23:59:59 CET (~4.5 weeks remaining as of 2026-09-08) | Main page |
| Training data | ~4,167,797 movements, full year 2025, monthly parquet files (~277 MB total) | [Data page](https://ansperformance.eu/study/data-challenge/dc2026/data.html) |
| Ranking data | Same columns; `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` blanked for PHASE=DEP; single file `ranking.parquet` (27 MB) | Data page |
| Submission | Fill `submitting.parquet` (columns `MVT_ID_mvt`, `TAXITIME_SEC_mvt`); name `thoughtful-glove_vN.parquet`; upload to MinIO bucket `prc-2026-thoughtful-glove` | Data page + project notes |
| Airports (10) | EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LTFM, LSZH | Data page Table 1 |
| Requirements | Open data, open code (public GitHub repo), JOAS open-access paper encouraged (esp. top teams) | Main page |
| Prizes | 5000 EUR combined for top 3 teams | Main page |

Note: the main page says "10 airports" in one place and "11 airport movements" in another; the authoritative Table 1 lists exactly 10 airports. Treat 10 as correct.

## 2. Critical Modeling Insight: What Is Known at Prediction Time

The ranking dataset blanks `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` for departures, but **`MVT_TIME_UTC_mvt` (actual takeoff time) remains available**. Since `TAXITIME = MVT_TIME - BLOCK_TIME`, the task is effectively: *given the actual takeoff time (and all other context), infer when the aircraft pushed back*. This is a **post-operations analysis** setting, not a pre-departure forecast — consistent with the stated rationale (identify constrained intervals, measure excess fuel/CO2 vs unconstrained status). Implications:

- Takeoff time of day, runway, stand, and **simultaneous airport state (arrivals + other departures around that takeoff time)** are legitimate, highly predictive features.
- Congestion features must be computed **relative to the takeoff time**, not the movement off-block time. **CORRECTION (Task 1.1 verification, 2026-09-08):** only `BLOCK_TIME_UTC_mvt` and the target `TAXITIME_SEC_mvt` are blanked for DEP in `ranking.parquet`; the NM off-block times `AOBT_3_flt`/`LOBT_flt`/`IOBT_flt` are **~98.5% populated** (they survive — the earlier assumption that they were blanked is refuted). Because these columns are present at scoring time, using them is **not** leakage. Empirically `MVT_TIME_UTC_mvt − AOBT_3_flt` reconstructs the target at RMSE ≈ 384s (median 175s) — the strongest single anchor. See `src/reports/data_verification.md` §5b.
- Congestion features can therefore be built from any ranking-available time: `MVT_TIME_UTC_mvt`, `SCHED_TIME_UTC_mvt`, `AOBT_3_flt`/`EOBT_1_flt`/`IOBT_flt`/`LOBT_flt`, and arrival-phase rows. The only same-flight time genuinely missing is the movement `BLOCK_TIME_UTC_mvt`.

## 3. Data Schema Notes (from official column description)

- Movement columns (`_mvt`): MVT_ID, FLIGHT_ID, FLIGHT (number), FLIGHT_RULE (I/V/NA), ADEP, ADES, PHASE (DEP/ARR), MVT_TIME_UTC (takeoff if DEP, landing if ARR), BLOCK_TIME_UTC (off-block if DEP, in-block if ARR), SCHED_TIME_UTC, AIRCRAFT_TYPE, RUNWAY, STAND, TAXITIME_SEC.
- Flight columns (`_flt`, from EUROCONTROL NM): LOBT, CALLSIGN, ADEP/ADES, ADES_FILED (diversion flag if != ADES), MARKET_SEGMENT (Mainline/Regional/Low-Cost/Business Aviation/All-Cargo/Charter/...), IOBT, FLIGHT_RULE, FLIGHT_TYPE, AIRCRAFT_TYPE, WK_TBL_CAT (L/M/H/J wake), AIRCRAFT_OPERATOR (anonymized), EOBT_1, ARVT_1, AOBT_3, ARVT_3.
- Organizers explicitly warn: real messy data, inconsistencies between movement and flight tables, not reconciled. Military/Head-of-State/sensitive movements removed.
- Both DEP and ARR phase rows are provided — ARR rows are usable to build arrival-pressure/congestion features even in the ranking window.

## 4. Prior PRC Data Challenge Editions (2024, 2025)

- 2024: actual takeoff weight prediction. 2025: fuel burn per flight interval (2127 submissions from 51 teams; winner announced Dec 2025).
- Winning approaches in prior editions were **gradient-boosting based with heavy domain feature engineering**, e.g. the JOAS paper "A Physics-Guided Gradient Boosting Framework for Open-Source Aviation Fuel Estimation" — physics-informed features + LightGBM-style models, not deep learning.
- Team repos are public: https://github.com/prc-data-challenge-2024 and https://github.com/prc-data-challenge-2025 — useful for MinIO access patterns, submission tooling, CV design, and open-repo structure requirements. The 2026 target (taxi-out) is different from 2024/2025, so modeling code transfers poorly; infrastructure code transfers well.

## 5. Taxi-Out Time Prediction Literature (key findings)

1. **"Can machines learn how to forecast taxi-out time?" (Sea-Tac comparison study)** — ensemble ML (random forests / boosted trees) clearly outperformed OLS and penalized regression for taxi-out prediction. Supports the ensemble-first strategy.
2. **ICAS 2022 paper (ML taxi-time prediction)** — ML alternatives to regression for departure taxi time; confirms tree ensembles are the standard strong baseline.
3. **"Predicting Taxi-Out Time at Congested Airports" (2022)** — taxi-out is strongly driven by **departure queue length and runway configuration**; queue-based features (number of departures awaiting takeoff in a time window) are among the top predictors.
4. **"Exploring prediction accuracy for optimal taxi times" (Journal of Air Transport Management, 2024)** — pushback delay, ground delay program status, and demand-capacity imbalance significantly associated with taxi times.
5. **Stochastic departure metering with ML taxi-time prediction (Aerospace/MDPI 2026)** — both point and distributional taxi-time models; distributional outputs (quantile regression) can help if error metrics reward calibration, but RMSE scoring means point prediction of the conditional mean is optimal.

### Consensus feature families from literature (adapted to our schema)
- **Airport congestion**: counts of departures/arrivals in rolling windows around the flight's takeoff time (from movement times, which exist in ranking data); departure queue proxies; per-airport-per-hour load; unscheduled/scheduled demand gap.
- **Temporal**: hour-of-day, day-of-week, month, seasonality, holiday flags; local-time vs UTC conversion per airport (airports span UTC+0 to UTC+3 — local hour matters for bank structures).
- **Airport-geometry**: runway ID, stand ID (stand-to-runway distance is airport-layout-driven), airport (10 levels, one-hot or per-airport models).
- **Flight characteristics**: aircraft type, wake category, market segment, operator (anonymized but stable), flight rule, ADES (route/distance), diversion flag.
- **Weather (external)**: METAR wind (headwind on departure runway), visibility/ceiling (low visibility procedures slow taxi), precipitation, snow, temperature (de-icing → big taxi-out increases in winter).

## 6. External Data Sources (user-approved: all three, with feasibility gate + ablation)

| Source | Access | Cost/limits | Notes |
|---|---|---|---|
| METAR/TAF for the 10 airports | Iowa State Mesonet archive (mesonet.agron.iastate.edu, `ASOS/` and `awos/` CSV download) or NOAA | Free, bulk HTTP | Hourly/special observations; join to movements by nearest METAR timestamp within 1h. Very likely allowed (open data, consistent with rules) — verify rules text during feasibility check. |
| Public holidays | `holidays` Python package or open holiday APIs covering DE, UK, NL, ES, FR, IT, TR, CH | Free, local | Per-airport country calendar; school-holiday proxies optional. |
| OpenSky historical API | REST/Trino with `resources/credentials.json` (client `mqignacio-api-client`) | Standard tier: 4,000 credits/day | Could enrich with ADS-B (e.g., actual surface movement timing, runway occupancy). Risk: credit budget over 4M movements, coverage, and rules ambiguity. **Feasibility-gated**: only pursue if METAR+holiday features leave measurable headroom and rules clearly allow. |

## 7. Risks and Pitfalls Identified

- **Leakage (UPDATED after Task 1.1 verification):** only `BLOCK_TIME_UTC_mvt` and the target `TAXITIME_SEC_mvt` are unavailable for DEP in `ranking.parquet`. The NM off-block columns (`AOBT_3_flt`, `LOBT_flt`, `IOBT_flt`) **are present** and are legitimate features (not leakage). Build features ONLY from columns confirmed populated in `ranking.parquet` (see `src/reports/data_verification.md` §5 availability table).
- **Anonymized operators**: consistent within dataset; treat as categorical, mind high cardinality (target encoding with CV-safe scheme).
- **Messy joins**: movement↔flight mismatches; FLIGHT_ID_mvt nullable; duplicated MVT_IDs possible. EDA must quantify.
- **Airport heterogeneity**: taxi-out distributions differ per airport (e.g., LHR short taxi vs IST long). Per-airport models vs global model with airport features — decide by CV.
- **Outliers — PROMOTED to the top lever (Task 2.2 finding, 2026-09-09)**: extreme taxi times (data errors, de-icing, remote de-icing pads). RMSE punishes them, and this is now quantified: **721 rows (0.03% of the 2025 training set) carry 60.3% of total squared error** — RMSE 365.18s overall vs 230.00s excluding them. Raw truth spans **−12 s to 131,167 s** (the latter is a data error). Halving tail error projects the leaderboard from 496.73s to **~415s**, which dwarfs any expected feature gain. Note the leaderboard scores **raw** `TAXITIME_SEC_mvt`, so clipping *predictions* helps only up to the true tail; clipping the *training target* (as the baseline does, [1, 14400] s) biases predictions low on genuinely long taxi-outs. Tune clip/winsorize via **raw-target** CV.
- **Submission cadence (RESOLVED in Task 2.2, 2026-09-09)**: limit is **5 submissions/day** and **1 GB total per bucket**. Quote from the ranking page: "There is currently a limit of 5 submissions per day (and a total max of 1GB per bucket.)" A submission is 4.5 MiB, so cadence — not storage — binds. Result files land in **~22 s**, so scoring feedback is effectively synchronous; no polling loop needed. Do not rely on leaderboard feedback as primary model selection (user chose month-blocked CV as primary).
- **Scoring rule (RESOLVED in Task 2.2, 2026-09-09)**: the leaderboard ranks by the team's **best** submission, not the latest — "Teams will be ranked based on the best Root Mean Square Error (RMSE) achieved across their submissions." Experiment submissions carry no downside risk; a bad score cannot overwrite a good one. Confirmed by the leaderboard UI header "Best submission per team." See `submissions/SUBMISSION_LOG.md`.
- **Timezone**: all times UTC; local-time features need per-airport offset (with DST transitions across Jan/Jul 2026 scoring windows).

## 8. Strategy Synthesis

1. **Ensemble of gradient-boosted trees (LightGBM primary, XGBoost/CatBoost candidates) on tabular features** — aligned with persona's hypothesis (classic ML + domain features, no deep learning needed) and prior-edition winners.
2. **Congestion features built from ranking-available time columns** are the expected single biggest win (literature consensus).
3. **Temporal month-blocked CV** across 2025 as primary model selection; leaderboard as confirmation only.
4. **External data (METAR + holidays) added behind a feasibility gate and validated by ablation**; OpenSky API only if headroom remains and credits allow.
5. **Final weeks: blending + per-airport error analysis + submission iteration.**

## References

- Challenge: https://ansperformance.eu/study/data-challenge/dc2026/ ; /rationale.html ; /data.html
- Ranking: https://prc-data-challenge-2026.netlify.app/ranking.html
- OpenSky API: https://openskynetwork.github.io/opensky-api/
- Prior repos: https://github.com/prc-data-challenge-2024 , https://github.com/prc-data-challenge-2025
- Prior JOAS papers: "A Physics-Guided Gradient Boosting Framework for Open-Source Aviation Fuel Estimation"; "Fuel Burn Prediction from Open Flight Trajectory and Meteorological Data for the PRC Data Challenge 2025" (JOAS, 2026)
- Taxi-out literature: Sea-Tac comparison study (ResearchGate, 2018); ICAS2022-0267; "Predicting Taxi-Out Time at Congested Airports" (2022); "Exploring prediction accuracy for optimal taxi times" (JATM 2024); "Stochastic Departure Metering with ML Taxi-Time Prediction" (Aerospace 2026)
- METAR archive: https://mesonet.agron.iastate.edu/
