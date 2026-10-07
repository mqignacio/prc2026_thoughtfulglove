# External Data Coverage Report — Task 2.3

Generated: 2026-09-10 · Status: **COMPLETE**

---

## 1. METAR Features (`data/processed/metar_features.parquet`)

### Source
- **Archive:** Iowa State Mesonet ASOS/METAR archive (https://mesonet.agron.iastate.edu/)
- **API endpoint:** `GET /cgi-bin/request/asos.py` with `data=all`, `format=comma`, `tz=UTC`
- **License:** Open data, freely available for research and commercial use. Data sourced from Unidata IDD, NCEI ISD, and MADIS. No restrictive license; attribution appreciated but not required.
- **Fetch script:** `src/fetch/fetch_metar.py` (reproducible, month-by-month bulk download with retry logic)

### Coverage Summary

| Metric | Value |
|---|---|
| Total observations | 277,489 |
| Airports covered | 10/10 |
| Date range | 2025-01-01 00:00 UTC → 2026-07-31 23:55 UTC |
| Months downloaded | 190/190 (all succeeded after retries) |
| Columns | 18 (station, valid, tmpc, dwpc, relh, wind_dir, wind_kt, wind_gust_kt, vis_miles, alti_in, mslp_mb, precip_1hr_in, ceiling_ft, flight_category, wxcodes, snowdepth_in, feel_f, metar_raw) |

### Per-Airport Observation Counts

| Airport | Observations | Temp non-null | Visibility non-null |
|---|---|---|---|
| EDDF | 27,692 | 27,692 | 27,692 |
| EDDM | 27,694 | 27,689 | 27,694 |
| EGLL | 27,686 | 27,686 | 27,686 |
| EHAM | 27,615 | 27,615 | 27,615 |
| LEBL | 28,015 | 28,015 | 28,015 |
| LEMD | 28,110 | 28,110 | 28,110 |
| LFPG | 27,705 | 27,705 | 27,704 |
| LIRF | 27,605 | 27,602 | 27,600 |
| LSZH | 27,687 | 27,685 | 27,685 |
| LTFM | 27,680 | 27,680 | 27,680 |

### Gap Analysis
**No gaps > 1 day detected at any airport.** All 10 airports have continuous coverage across the full 2025-01-01 to 2026-07-31 period.

### Flight Category Distribution

| Category | Count | % |
|---|---|---|
| VFR | 218,715 | 78.8% |
| MVFR | 39,060 | 14.1% |
| IFR | 11,232 | 4.0% |
| LIFR | 8,476 | 3.1% |

### Derived Fields
- **Temperature:** Converted from Fahrenheit to Celsius (`tmpc`, `dwpc`)
- **Ceiling:** Lowest BKN/OVC/VV layer height in feet (derived from skyc1–4 + skyl1–4)
- **Flight category:** LIFR/IFR/MVFR/VFR derived from ceiling and visibility thresholds
- **Raw METAR string:** Preserved in `metar_raw` column for downstream parsing if needed

---

## 2. Holiday Features (`data/processed/holiday_features.parquet`)

### Source
- **Package:** `holidays` v0.104 (Python)
- **License:** MIT License (open source, permissive)
- **Country code pin:** Uses ISO `"GB"` for United Kingdom (verified at runtime; `holidays` 0.104 supports `"GB"`)
- **Fetch script:** `src/fetch/fetch_holidays.py`

### Coverage Summary

| Metric | Value |
|---|---|
| Total rows | 5,770 (10 airports × 577 days) |
| Countries | 8 (CH, DE, ES, FR, GB, IT, NL, TR) |
| Date range | 2025-01-01 → 2026-07-31 |
| Columns | 7 (airport, date, country, is_public_holiday, holiday_name, is_day_before_holiday, is_day_after_holiday) |

### Holidays per Country

| Country | Holiday dates | Airports |
|---|---|---|
| CH | 6 | LSZH |
| DE | 15 | EDDF, EDDM |
| ES | 13 | LEBL, LEMD |
| FR | 18 | LFPG |
| GB | 10 | EGLL |
| IT | 20 | LIRF |
| NL | 19 | EHAM |
| TR | 26 | LTFM |

Note: These are national-level public holidays only. Regional/sub-national holidays are not included (could be added later if ablation shows value).

---

## 3. Joinability to Movements

### Methodology
For each DEP movement in training (2025) + ranking (Jan/Jul 2026), found the nearest METAR observation **at or before** the movement timestamp within a 60-minute window using `pd.merge_asof(direction='backward', tolerance='60min')`.

### Results

| Airport | Matched | Total | % |
|---|---|---|---|
| EDDF | 266,458 | 266,458 | 100.0% |
| EDDM | 193,300 | 193,300 | 100.0% |
| EGLL | 279,386 | 279,386 | 100.0% |
| EHAM | 286,133 | 286,133 | 100.0% |
| LEBL | 209,727 | 209,785 | 100.0% |
| LEMD | 249,179 | 249,196 | 100.0% |
| LFPG | 279,402 | 279,424 | 100.0% |
| LIRF | 187,535 | 187,603 | 100.0% |
| LSZH | 158,047 | 158,057 | 100.0% |
| LTFM | 320,494 | 320,546 | 100.0% |
| **OVERALL** | **2,429,661** | **2,429,888** | **100.0%** |

### Time Gap Statistics (METAR observation → movement takeoff)

| Statistic | Seconds | Minutes |
|---|---|---|
| Median | 901 | 15.0 |
| Mean | 903 | 15.0 |
| P95 | 1,710 | 28.5 |

**Verdict:** ≥95% acceptance criterion **exceeded** — effectively 100% of movements have a METAR within 60 min before takeoff. The median gap of 15 minutes reflects the ~30-minute observation cadence at most European airports.

---

## 4. Consumer Note for Task 2.4 (Ablation)

- Use the **raw-target CV metric** (from Task 2.2a) when evaluating METAR feature deltas.
- A null ablation delta with this level of coverage means "weather features don't help", NOT "coverage was insufficient to measure".
- The 227 unmatched movements (0.009%) are negligible and should not affect ablation conclusions.
- Raw METAR strings are preserved in `metar_raw` for potential additional parsing (e.g., runway-specific wind, RVR) if initial numeric features show limited signal.

---

## 5. Data Quality Notes

- Precipitation data (`precip_1hr_in`) is noted as "unavailable for non-US sites" by the IEM documentation. Values present may be unreliable for European airports — treat with caution during ablation.
- Temperature/dewpoint have <0.02% missingness across all airports (negligible).
- Visibility has similarly negligible missingness.
- Snow depth is mostly missing for European airports (expected — few snow events at these latitudes except EDDM/LTFM in winter).
