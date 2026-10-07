# PRC-OpenSky Data Challenge 2026 — Team `thoughtful-glove`

Open solution for the [PRC-OpenSky Data Challenge 2026](https://ansperformance.eu/study/data-challenge/dc2026/):
predict **departure taxi time** (seconds) for flights at 10 major European
airports. This repository contains the full modelling pipeline, the experiment
reports, and the figures behind our final submission and the accompanying JOAS
paper.

**Final result:** leaderboard RMSE **392.59 s**, rank **#170 / 232 teams**
(best of five submissions; official scoring API, 2026-10-07).
Month-blocked cross-validation of the same model: **333.02 s** (blended) and
**331.00 s** with the accepted post-processing calibration.

- Task: predict `TAXITIME_SEC_mvt` = actual taxi time, in seconds, from the
  moment a flight leaves the stand to the moment it reaches the runway.
- Metric: RMSE. Lower is better.
- Team: `thoughtful-glove`.

---

## 1. Problem

Airport surface congestion, runway configuration, weather, and airline
behaviour all affect how long an aircraft spends taxiing out. The challenge
provides historical flight movements and asks for a taxi-time prediction on a
held-out set of departures. The value is operational: better taxi-time
estimates improve block-time planning, slot adherence, and fuel/emissions
estimates.

---

## 2. Data

The organizers distribute the datasets to registered participants; they are
**not redistributed here** (organizer data terms). Obtain them from the
challenge platform with your OpenSky account, then place the parquet files in
`data/raw/`.

| File(s) | Rows (total) | Description |
|---|---:|---|
| `training_2025-*.parquet` (12 monthly files) | 4,167,797 | 2025 flight movements, both arrival and departure phases |
| `ranking.parquet` | 689,534 | Evaluation-period movements (Jan 2026 + Jul 2026) |
| `submitting.parquet` | 344,841 | The departure rows to predict |

Training uses the departure (`PHASE_mvt == "DEP"`) subset: **2,085,047** rows.
The reporting airports are EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF,
LSZH, and LTFM.

**Setup:**

```bash
uv sync
# Place the organizer parquet files in data/raw/ (see below), then:
uv run python src/fetch/fetch_competition_data.py --from-dir <your-download-dir>
#   or, with a configured mc alias for the challenge endpoint:
uv run python src/fetch/fetch_competition_data.py --mc-alias opensky
uv run python src/verify_data.py          # row counts, airport sets, MVT_ID equality
```

`src/verify_data.py` must print **all checks PASS** before training.

**External features** are fetched from public sources (not distributed here):

```bash
uv run python src/fetch/fetch_metar.py       # Iowa State Mesonet METAR archive
uv run python src/fetch/fetch_holidays.py    # public holidays for the 8 airport countries
```

---

## 3. Features

Features were locked in `resources/feature_inventory.md` after a
ranking-availability audit (every feature must exist in `ranking.parquet`).
They fall into four groups:

- **Base / anchors** — time-of-day, day-of-week, month, local hour (DST-aware),
  schedule deviation, runway, stand, aircraft type, flight rule, great-circle
  route distance, plus near-oracle off-block anchors from the NM flight plan
  (`AOBT_3_flt`, `LOBT_flt`, `IOBT_flt`).
- **Congestion** (`D2`) — arrival/departure counts in rolling time windows
  around each movement, computed per airport and runway.
- **Weather** (`D3`) — METAR observations (wind, visibility, ceiling, weather
  codes) joined to each movement.
- **Domain** (`D4`) — holiday flags, runway heading/direction groups,
  per-airport target encodings, and `FLIGHT_PREFIX` (the airline ICAO code
  parsed from the flight number, the only operator signal available for the
  ~1% of rows with missing operator metadata).

The categorical vocabulary is fit on the training set once and applied to the
ranking set (unseen values → `-1`), so training and scoring share one mapping.

---

## 4. Model

- **Target treatment:** `E-winsorize-errors` — negatives floored to 1 s, values
  above 24 h capped at the airport-specific P99. CV is scored on the **raw
  unclipped** target, matching the competition metric.
- **Validation:** 12-fold leave-one-month-out (month-blocked) CV on 2025.
  Model selection uses CV only; the leaderboard is never a feedback signal.
- **Components:** a global LightGBM tuned with Optuna (52 trials), and a stack
  of 10 per-airport LightGBM models, on the D4 feature set.
- **Blend:** simplex-constrained weight search on out-of-fold predictions only.
  The selected blend is `tuned_lgbm` (0.409) + `per_airport_lgbm` (0.591).
  XGBoost and CatBoost receive zero weight.
- **Post-processing:** per-airport affine recalibration with the slope shrunk
  50% toward 1 (`mi2_affine_s50`), accepted by a leave-one-month-out test.

Configuration: `src/configs/best.json` (tuned params), `src/configs/blend.json`
(components and weights), `src/configs/blend_postproc.json` (calibration).

---

## 5. Results

| Model | CV RMSE (raw, s) | Notes |
|---|---:|---|
| Constant baseline (predict the mean) | 546.36 | Trivial reference |
| Baseline LightGBM (21 features) | 344.07 | `src/run_baseline.py` |
| Tuned global LightGBM | 337.16 | out-of-fold |
| Per-airport LightGBM | 335.01 | out-of-fold, best single |
| **Blend (tuned + per-airport)** | **333.02** | out-of-fold |
| **Blend + `mi2_affine_s50`** | **331.00** | leave-one-month-out (honest) |

Season-matched (Jan + Jul) CV of the final model: **452.90 s** — the fair
CV analog of the 2026 leaderboard period. Leaderboard: **392.59 s**,
rank **#170 / 232** (best of five submissions; full submission history in
`submissions/SUBMISSION_LOG.md`).

---

## 6. Reproducing the results

After `uv sync` and the data/external fetch steps above:

```bash
# 1. Baseline pipeline (21 features). ~4 minutes on a laptop.
#    Expected: overall raw CV RMSE ≈ 344.07 s, constant baseline 546.36 s,
#    season-matched (Jan+Jul) ≈ 459.26 s.
uv run python src/run_baseline.py

# 2. Final blend. `oof` runs 12-fold CV for all four components (~25 min on
#    Apple M4 Pro); `weights` searches blend weights on the OOF table only.
#    Expected: blend overall raw CV RMSE = 333.02 s
#              (tuned 337.16, per-airport 335.01, xgb 352.22, cat 362.02).
uv run python src/run_blend.py --phase oof
uv run python src/run_blend.py --phase weights

# 3. Post-processing verdicts on the OOF table (leave-one-month-out).
#    Expected: accepted candidate `mi2_affine_s50`
#              overall 331.00 s, season-matched 452.90 s.
uv run python src/run_micro_iterations.py

# Optional: re-run the Optuna tuning search (stochastic, ~68 min).
uv run python src/run_tuning.py --phase all
```

The blend and calibration are deterministic (seed 42), so steps 2–3 reproduce
the numbers above exactly. The tuned parameters are committed in
`src/configs/best.json`; step 4 is optional and only regenerates them.

To build a submission parquet without retraining the blend:

```bash
uv run python src/run_blend.py --phase reblend --version 5 --candidate mi2_affine_s50
```

---

## 7. Repository layout

```
assets/figures/        Figures used in the reports and the paper
resources/             feature_inventory.md, external-data research, challenge notes
src/
  pdc26/               Library: data loading, CV, features, training, tuning, airports
  fetch/               External-data fetchers (METAR, holidays) + competition-data helper
  configs/             Model, blend, and calibration configs (JSON)
  notebooks/           02_eda.ipynb — exploratory data analysis
  reports/             Experiment reports (EDA, ablations, tuning, blending, errors)
  run_*.py             Experiment entry points (one per report)
  make_submission.py   Build a submission parquet (thin wrapper over pdc26)
  minio_transfer.py    Optional helper to move files via the organizer MinIO bucket
  verify_data.py       Data verification checks
submissions/           SUBMISSION_LOG.md — full submission history
pyproject.toml         Dependencies (uv)
uv.lock                Lockfile
LICENSE                GPL-3.0
```

Reports in `src/reports/` document each experiment: EDA, feature ablations
(external data, congestion, domain), tail/outlier handling, model comparison,
tuning, error analysis, feature pruning, and the blend. Figures are in
`assets/figures/`.

---

## 8. Environment

- Python 3.12, managed with [uv](https://docs.astral.sh/uv/).
- Main dependencies: LightGBM, XGBoost, CatBoost, scikit-learn, Optuna, Polars,
  pandas, PyArrow, matplotlib, seaborn, holidays.
- `minio_transfer.py` needs the `minio` package, run ad hoc with
  `uv run --with minio python src/minio_transfer.py ...`.

---

## 9. License

Released under the **GNU General Public License v3.0** — see `LICENSE`.

## 10. Citation

A paper describing this work is in preparation (JOAS). Citation details will
be added here once available.

## Acknowledgements

Data provided by the PRC-OpenSky Data Challenge 2026 organizers. METAR data
from the Iowa State University Mesonet archive. Airport coordinates from the
`airportsdata` package.
