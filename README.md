# PRC-OpenSky Data Challenge 2026 — Team `thoughtful-glove`

Open solution for the [PRC-OpenSky Data Challenge 2026](https://ansperformance.eu/study/data-challenge/dc2026/):
predict **departure taxi time** (seconds) for flights at 10 major European
airports. This repository contains the full modelling pipeline, the experiment
reports, and the figures behind our final submission and the accompanying JOAS
paper.

**Final result:** leaderboard RMSE **392.59 s**, rank **#170 / 232 teams**
(best of six submissions — five model iterations plus one validation probe; official scoring API, 2026-10-07).
Month-blocked cross-validation of the same model: **333.02 s** (blended) and
**331.00 s** with the accepted post-processing calibration.

The October 2026 **final phase** (Jan+Feb+Jun+Jul 2026) was submitted
2026-10-12 as `thoughtful-glove_final.parquet`; the final files carry no
leaderboard, so that result is pending — see [§6](#6-final-phase-october-2026).

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

**Final-phase inputs (October 2026).** The final phase re-issued the
evaluation files as `final_ranking.parquet` (1,341,197 rows) and
`final_submitting.parquet` (670,790 departure rows to predict) for the four
months Jan/Feb/Jun/Jul 2026. They come from the same organizer bucket and go
in `data/raw/`. The final model and its reproduction are in §6.

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

The **final-phase** submission (`thoughtful-glove_final.parquet`, the same
frozen stack applied to the 4-month Jan/Feb/Jun/Jul 2026 files) was uploaded
2026-10-12; its result is pending because the final files are not scored on a
leaderboard. Section 6 documents that work and its validation.

---

## 6. Final phase (October 2026)

The competition's final phase, announced 2026-10-08, replaced the 2-month
Jan/Jul evaluation with a **4-month** one and re-issued the input files.

| | Ranking phase (v5) | Final phase |
|---|---|---|
| Input files | `ranking.parquet` (689,534 rows) | `final_ranking.parquet` (1,341,197 rows) + `final_submitting.parquet` |
| Months | Jan + Jul 2026 | **Jan, Feb, Jun, Jul 2026** |
| Departures to predict | 344,841 | **670,790** |
| Takeoff times | as distributed | re-dithered **±12 s** (91.83% of Jan/Jul movements moved) |
| Targets | blanked on DEP | blanked on DEP (`TAXITIME_SEC_mvt` 100% null) |

**What changed in the model: nothing.** The frozen v5 stack (tuned +
per-airport LightGBM, blend weights, `mi2_affine_s50`) is loaded from disk
unchanged — no retraining, no new features, no recalibration refits. Only the
serving path was adapted: features are rebuilt from `final_ranking.parquet`
and predictions are written in `final_submitting.parquet` order.

**Final submission:** `thoughtful-glove_final.parquet`, 670,790 rows,
8,641,587 bytes, SHA256 `e326deac…c632`, uploaded once 2026-10-12 03:51:28
UTC. The final files carry **no leaderboard**, so the result is unknown until
the organizers publish the final rankings.

### 6.1 What we verified

Three GO/NO-GO checks (`src/verify_final_data.py`; report
`src/reports/final_data_verification.md`):

| Check | Result | Verdict |
|---|---|---|
| Runway anchor (`AOBT_3_flt` family) survives | 98.6468% pooled populated (vs 98.4660% on `ranking.parquet`) | GO |
| Hard rows (null-operator) present | 1.3532% (9,077 / 670,790) | GO |
| External features join | METAR 99.9884% pooled (worst month 99.9659%); holidays 100% | GO |

Mean prediction shifts versus the 2025 analog are ≤ +7.2% per month. The
January and July means differ from v5's own predictions by ≤ 0.75% (median
per-row difference ≈ 7–9 s, the ±12 s dither). See
`src/reports/final_pipeline_validation.md`.

### 6.2 Honest validation story

There is no leaderboard on the final months, so validation rests on three
things:

1. **The 2025 analog.** The month-blocked CV of the same model on
   Jan+Feb+Jun+Jul 2025 (the closest available analog) gives **420.37 s**
   (raw blend). This is the reference the final score is expected to beat by a
   margin similar to the ranking phase: v5 scored 392.59 s on the leaderboard
   versus a 454.25 s 2-month analog, a −13.6% favourable gap.
2. **A serving-regression probe.** One validation-only submission (v6) on the
   Jan/Jul subset scored **393.2793 s vs v5's 392.5911 s (+0.18%)** —
   rebuilding every feature from the dithered final files reproduces v5's
   leaderboard performance, inside the pre-registered 2% gate.
3. **A drift audit (adversarial validation).** A classifier separating
   2025-analog from 2026-final rows returns a high AUC (0.9998 row-split;
   0.9678 day-grouped), but the separability comes from METAR weather
   (family AUC 0.86) and the congestion-count features (0.999), not the
   predictive core (anchors 0.56, temporal 0.52, flight characteristics
   0.50). The congestion signal is a **known latent units bug** in the frozen
   v5 feature code (`MIN_NS` nanoseconds applied to microsecond timestamps →
   windows ~1000× too wide). It is consistent between training and serving,
   so it does not change predictions; it is disclosed here rather than fixed
   (a serve-only fix would break train/serve consistency, and a real fix
   requires retraining). Full analysis: `src/reports/final_pipeline_validation.md` §6.

### 6.3 Hard-row experiment (pre-registered, CV-gated)

About 1.35% of final departures have null operator metadata ("hard rows").
On the 2025 analog these rows carry 64.7% of the squared error, but their
targets are largely corrupted (LIRF median ≈ 3,922 s; ~24 h rollovers). We
pre-registered a time-boxed experiment (`src/reports/hardrow_experiment.md`),
then ran it:

- A dedicated LightGBM on only the hard rows **passed the signal test**
  (beat the blend on held-out hard rows in 3 of 4 analog folds).
- It cleared the full gate with candidate `replace`: 4-month analog
  **391.38 s vs 420.37 s (+6.90%)**, hard-row RMSE 2,604.66 vs 2,922.74 s,
  base rows byte-identical → **ACCEPT (within-2025)**.

But the gate measures fitting the *corrupted 2025 tail*, and the evidence
says the 2026 scored truth is cleaner (the ranking-phase leader scored
213.18 s, below the 338.1 s floor a perfect base model would hit if the
corruption transferred). Swapping in the hard-row treatment (`final_v2`)
moves 9,056 of 9,077 hard rows by RMS 1,084 s, with an expected effect of
≈ **+21 s if the truth is clean** versus ≈ **−25 s if the corruption
transfers**. **We shipped v1.** `final_v2` (SHA256 `a0a3b54a…`) stays as the
documented experiment artifact — an honest, reproducible negative on
transfer.

### 6.4 Reproducing the final submission

Prerequisites: `uv sync`; all 2025 training files, `ranking.parquet`,
`submitting.parquet`, and the two final-phase files placed in `data/raw/`;
METAR and holiday features fetched (see §2). The frozen models are **not**
shipped (`models/` is gitignored), so step 2 retrains them.

```bash
# 1. Verify the final-phase input files.
#    Expected: ALL CHECKS GO; anchor 98.6468%; hard rows 1.3532% = 9,077;
#    METAR 99.9884%.
uv run python src/verify_final_data.py

# 2. Train the frozen v5 stack (12-fold OOF, ~25 min), then retrain on full
#    2025 and save the models the wrapper loads.
#    Expected: blend OOF 333.02 s (tuned 337.16, per-airport 335.01).
uv run python src/run_blend.py --phase oof
uv run python src/run_blend.py --phase weights
uv run python src/run_blend.py --phase serve --version 5
#    -> saves models/blend/tuned_lgbm_full.txt + per_airport_lgbm_<AP>_full.txt

# 3. Build the final submission from the frozen models.
#    Expected: 670,790 rows, SHA256 e326deac…c632, mean 1016.2 s.
uv run python src/make_final_submission.py

# 4. Validate it.
#    Expected: all pre-upload assertions PASS; two-build rebuild
#    byte-identical; 4-month analog 420.37 s.
uv run python src/validate_final_pipeline.py

# 5. Re-run the pre-registered hard-row experiment.
#    Expected: gate ACCEPT within-2025, 391.38 s vs 420.37 s.
uv run python src/run_hardrow_experiment.py
#    Optional: rebuild the experiment-only final_v2 artifact.
#    Expected: SHA256 a0a3b54a…, base rows byte-identical to v1.
uv run python src/make_final_submission.py --hardrow-v2
```

---

## 7. Reproducing the results

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

## 8. Repository layout

```
assets/figures/        Figures used in the reports and the paper
resources/             feature_inventory.md, external-data research, challenge notes
src/
  pdc26/               Library: data loading, CV, features, training, tuning, airports
  fetch/               External-data fetchers (METAR, holidays) + competition-data helper
  configs/             Model, blend, and calibration configs (JSON)
  notebooks/           02_eda.ipynb — exploratory data analysis
  reports/             Experiment reports (EDA, ablations, tuning, blending, errors,
                       final-phase verification / validation / hard-row experiment)
  run_*.py             Experiment entry points (one per report; includes
                       run_hardrow_experiment.py)
  make_submission.py   Build a submission parquet (thin wrapper over pdc26)
  make_final_submission.py   Build the final-phase (4-month) submission from frozen v5 models
  validate_final_pipeline.py Final-phase validation battery (assertions, rebuild, drift check)
  minio_transfer.py    Optional helper to move files via the organizer MinIO bucket
  verify_data.py       Data verification checks
  verify_final_data.py Final-phase input-file verification (anchors, hard rows, external joins)
submissions/           SUBMISSION_LOG.md — full submission history
pyproject.toml         Dependencies (uv)
uv.lock                Lockfile
LICENSE                GPL-3.0
```

Reports in `src/reports/` document each experiment: EDA, feature ablations
(external data, congestion, domain), tail/outlier handling, model comparison,
tuning, error analysis, feature pruning, and the blend, plus the final-phase
data verification, pipeline validation, and hard-row experiment. Figures are
in `assets/figures/`.

---

## 9. Environment

- Python 3.12, managed with [uv](https://docs.astral.sh/uv/).
- Main dependencies: LightGBM, XGBoost, CatBoost, scikit-learn, Optuna, Polars,
  pandas, PyArrow, matplotlib, seaborn, holidays.
- `minio_transfer.py` needs the `minio` package, run ad hoc with
  `uv run --with minio python src/minio_transfer.py ...`.

---

## 10. License

Released under the **GNU General Public License v3.0** — see `LICENSE`.

## 11. Contributing / Reproducibility

This repository is a competition artifact, not an actively maintained library.
The README's reproduction sections (§6.4 and §7) provide every command needed to
rebuild the results from source. Bug reports or questions can be filed as GitHub
issues on the repository.

## 12. Citation

A paper describing this work is in preparation (JOAS). Citation details will
be added here once available.

## Acknowledgements

Data provided by the PRC-OpenSky Data Challenge 2026 organizers. METAR data
from the Iowa State University Mesonet archive. Airport coordinates from the
`airportsdata` package.
