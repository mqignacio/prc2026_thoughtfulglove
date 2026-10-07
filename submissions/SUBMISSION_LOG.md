# Submission Log — PRC-OpenSky Data Challenge 2026

Team: `thoughtful-glove` · Bucket: `opensky/prc-2026-thoughtful-glove` · Started by Task `05-first_submission`.

---

## Submissions

| Version | Date (2026) | Model | CV RMSE (s) | Leaderboard RMSE (s) | Notes |
|---|---|---|---:|---:|---|
| v1 | 09-09 | `models/baseline_lgbm.txt` (LightGBM, 500 trees, 21 baseline features, no congestion) | 249.27 | **591.45** | First submission. Scored `Succeeded`, 344,841/344,841 pairs used. Pre-upload assertions clean, 0 clipped. Large CV→leaderboard gap investigated and fully attributed — see "Material Finding" + "Gap Attribution" below. |
| v2 | 09-09 | same booster, **categorical vocabulary corrected at score time** | 249.27 | **496.73** | Same model, no retraining. Only change: ranking categoricals rebound to the training vocabulary (`--fit-vocab` sidecar). 479 unseen cells → code −1; **74 values clipped** at 7,200 s. **−94.72 s (−16.0%) vs v1.** Beats the season-matched raw-target 2025 comparator (500.88 s) by 0.8% → **no measurable 2025→2026 drift.** |
| v3 | 10-03 | **blend** (Task 3.4): `tuned_lgbm` 0.4329 (global D4, `best.json`) + `per_airport_lgbm` 0.5671 (10 D4 models, family defaults) | 333.81 (blend OOF, all-2025; season-matched 455.11) | **401.73** | First **D4-feature** submission (62 features vs v1/v2's 21). Blend weights fit on OOF predictions only — leaderboard never used. Retrained on full 2025 at median fold `best_iteration`. 344,841 rows, 0 nulls, all finite, min floored to 1 s, no upper clip. **−94.99 s (−19.1%) vs v2.** Leaderboard is **−11.73% vs the season-matched 2025 CV comparator** (455.11 s) — favourable, no drift escalation. See "v3 — Blend" below. |
| v4 | 10-03 | **same blend as v3** + per-airport affine recalibration (slope shrunk 50%, `mi2_affine_s50`) | 331.30 (LOT-out OOF, all-2025; season-matched 453.67) | **396.74** | Model unchanged from v3 — **serving-time post-processing only**, accepted by leave-one-month-out CV (overall −2.52 s, season-matched −1.44 s). 344,841 rows, 0 nulls, all finite, min floored to 1 s. **−4.99 s (−1.24%) vs v3.** Leaderboard −12.54% vs the season-matched CV analog. Feature pruning of the 9 lowest-gain features was **rejected** (degrades CV). See "v4" below. |
| v5 | 10-07 | **same blend + FLIGHT_PREFIX feature** (airline code parsed from the flight number) + `mi2_affine_s50` | 331.01 (LOT-out OOF, all-2025; season-matched 452.90) | **392.59** | **FINAL SUBMISSION.** One new categorical feature (`FLIGHT_PREFIX`, 1,762 values) added to the D4 set (63 features total); blend weights and the affine recalibration refit. Every component improved: tuned 337.76→337.16, per-airport 336.12→335.01, xgb 354.31→352.22, cat 362.28→362.02. 344,841 rows, 0 nulls, all finite, min floored to 1 s. **−4.15 s (−1.05%) vs v4.** See "v5 — Final submission" below. |

`src/make_submission.py` is the reusable generator for the single-model versions:
`python src/make_submission.py --model <path> --version N` → `submissions/thoughtful-glove_vN.parquet`.
Multi-model blends use `src/run_blend.py` (see below).

---

## v3 — Blend (Task 3.4, 2026-10-03)

Task `11-blend_submission`. The first submission built from the **D4 feature set** (baseline + METAR + congestion + domain, 62 features), and the first multi-model blend. Blend weights were fit on **out-of-fold predictions only** — the leaderboard was never used to tune them.

**Components** (12-fold month-blocked CV, E-winsorize-errors target, raw unclipped RMSE):

| Component | Family | Mode | Config | OOF RMSE (s) |
|-----------|--------|------|--------|-------------:|
| `tuned_lgbm` | LightGBM | global | `best.json`: 221 leaves, lr 0.0623, min_child 12, feature_fraction 0.919, bagging 0.823, 500 rounds (median best_iter 418) | 337.76 |
| `per_airport_lgbm` | LightGBM | 10 per-airport | family defaults (63 leaves, lr 0.05), 500 rounds (median best_iter 172–420) | 336.12 |
| `xgb` | XGBoost | global | default (depth 6, lr 0.05), 500 rounds (median 498) | 354.31 |
| `cat` | CatBoost | global | default (depth 6, lr 0.05, 300 iters) | 362.28 |

**Reproduction gate:** every component's OOF RMSE reproduced its Task 3.3 value within 0.20% (tuned 337.54→337.76; per-airport 335.98→336.12; xgb 355.00→354.31; cat 362.24→362.28).

**Blend search** (simplex-constrained, all 15 non-empty subsets, multi-start SLSQP):

- Simple average of all 4: **340.33 s**
- Best single: `per_airport_lgbm` **336.12 s**
- **Selected: `tuned_lgbm` = 0.4329, `per_airport_lgbm` = 0.5671 → 333.81 s** (beats best single by **2.30 s**)
- `xgb` and `cat` received zero weight: their residual errors correlate ≥ 0.92 with the LightGBM models (tuned↔per-airport **0.964**; xgb↔cat 0.970), so they add nothing once the two LightGBM variants are in.

**Retrain + serving:** components retrained on full 2025 at their median fold `best_iteration` (tuned 418; per-airport 172–420). Ranking D4 features built with training-fitted vocabularies (unseen → −1), full-train operator target encoding, congestion from `ranking.parquet` DEP+ARR (689,534 rows), METAR for 2026-01/07. Blend floored at 1 s, no upper clip. Output: 344,841 rows, 0 nulls, all finite, range 1.0–39,905.4 s, mean 1,035.8 s.

**CV vs leaderboard**

| Quantity | RMSE (s) |
|---|---:|
| Blend OOF, all 2025 | 333.81 |
| Blend OOF, Jan+Jul 2025 (season-matched comparator) | 455.11 |
| **v3 leaderboard (Jan+Jul 2026)** | **401.73** |
| v2 leaderboard | 496.73 |
| v1 leaderboard | 591.45 |

The honest comparator is the **season-matched** 2025 figure (455.11 s): the leaderboard is **−11.73% (−53.38 s) better** than the 2025 Jan+Jul CV analog — no adverse drift, clearing the 5% escalation threshold in the favourable direction. (The naive all-2025 OOF comparison 333.81 → 401.73 = +20.35% mixes season regimes and is not the right comparator — see Gap Attribution, which established Jan+Jul as the correct CV analog.)

**Reproducibility:** `uv run python src/run_blend.py --phase oof` → `--phase weights` → `--phase serve --version 3`. Config `src/configs/blend.json`; report `src/reports/blend_results.md`; OOF table `data/processed/oof_predictions.parquet` (consumed by Task 4.1).

---

## v4 — Post-processing recalibration (Task 4.1, 2026-10-03)

Task `13-final_iterations`. The model is **unchanged from v3** (same blend, same 62 D4 features); the only change is a **per-airport affine recalibration** applied to the blended ranking predictions at serving time.

**Error analysis first** (`src/reports/error_analysis.md`): the blend's squared error is dominated by a tiny set of metadata-missing special flights — 1.08% of rows (null `AIRCRAFT_OPERATOR_flt`) carry **55.4% of all squared error**, and `LIRF/25` alone is 47.99% of total SE (RMSE 877.6 s; 409.8 s excluding the missing-operator rows). The accepted recalibration targets the resulting systematic compression of the upper tail.

**Accepted micro-iteration** (leave-one-month-out on the 2025 OOF — the honest estimate, not in-sample):

| Candidate | Overall OOF (s) | Δ | Season OOF (s) | Δ | Verdict |
|---|---:|---:|---:|---:|---|
| baseline blend | 333.815 | — | 455.110 | — | reference |
| per-airport weights (`mi1`) | 334.162 | −0.347 | 455.633 | −0.523 | reject |
| per-airport affine, full slope | 330.976 | +2.839 | 456.341 | −1.231 | reject (season degrades) |
| **per-airport affine, slope shrunk 50% (`mi2_affine_s50`)** | **331.296** | **+2.519** | **453.666** | **+1.443** | **ACCEPT** |
| per airport-runway affine | 334.410 | −0.595 | 466.672 | −11.562 | reject |
| weights + affine | 331.380 | +2.435 | 456.850 | −1.741 | reject |

The full-slope affine improves overall but degrades the season-matched metric (tail overfit); shrinking the slope 50% toward 1 gives a robust gain that passes both gates.

**Deltas:**

| Comparison | RMSE (s) |
|---|---:|
| v4 leaderboard (Jan+Jul 2026) | **396.74** |
| v3 leaderboard | 401.73 |
| **v4 vs v3** | **−4.99 (−1.24%)** |
| v4 season-matched CV analog (affine OOF, Jan+Jul 2025) | 453.67 |
| leaderboard vs CV analog | **−12.54%** (favourable) |

**Rejected micro-iteration:** feature pruning (drop the 9 lowest-gain of 62 features) **degrades** CV — overall 334.73 (+0.274%), season 458.28 (+3.17 s) — so the full 62-feature set is retained. See `src/reports/feature_pruning.md`.

**Reproducibility:** `uv run python src/run_blend.py --phase reblend --version 4 --candidate mi2_affine_s50` (applies `blend_postproc.json` to the existing v3 ranking predictions; no retraining). Config `src/configs/blend_postproc.json`; report `src/reports/micro_iterations.md`.

**Upload mechanism (2026-10-03):** the MacBook relay is **retired**. The Mac Mini now holds MinIO credentials (`~/.mc/config.json`, mode 600, outside the repo) and uploads/fetches directly with `uv run --with minio python src/minio_transfer.py up|ls|get opensky/<bucket>/<object>`. No human step for future submissions (vFINAL, Task 4.2).

---

## v5 — Final submission (Task 4.2, 2026-10-07)

Task `14-final_submission`. On the human user's request, one extra model/feature round was run before submitting. The binding constraints stood: month-blocked CV is the selection signal, the leaderboard is confirmation only, one change per version, best-submission-wins.

**Diagnosis first.** The blend's squared error is dominated by the **metadata-missing anchor rows**: the 1.08% of DEP rows with a null `AIRCRAFT_OPERATOR_flt` (and an entirely null `_flt` metadata block) carry **55.4% of all squared error** (blend RMSE 2,393.6 s on those rows vs 224.1 s on the other 98.92%). A hypothetical *perfect* predictor on those rows alone would take the overall OOF from 333.8 s to **222.9 s** — i.e. essentially top-of-leaderboard. So the whole gap to the leaders lives in that tail.

**Post-processing battery (rejected).** A LOT-out battery of serving-time recalibrations (`src/run_improvement_round.py` → `src/reports/improvement_round.md`) tested isotonic per-airport/global, per-(airport×predicted-decile) affine, per-airport quantile mapping, and per-airport tail caps. Isotonic and quantile-mapping degraded the season-matched metric (overfit); caps were strongly negative. The existing per-airport affine (`mi2_affine_s50`) remained the best accepted rule — the blend is already close to RMSE-calibrated.

**Feature change — `FLIGHT_PREFIX` (accepted).** The metadata-missing rows have no operator id, but they *do* carry `FLIGHT_mvt` (the flight number, e.g. `RYR975`, `NO`, `ISR344`), whose leading 2-3 letters are the airline ICAO code. That code is present in both training and ranking (99.2% coverage) and is the only operator proxy available for those rows. A LOT-out test on the hard rows showed the prefix is genuinely informative there. Adding it as a native LightGBM categorical (1762 values; unseen → −1) to the D4 feature set (62 → 63 features) improved **every** component:

| Component | OOF before | OOF after | Δ |
|---|---:|---:|---:|
| `tuned_lgbm` | 337.76 | 337.16 | −0.60 |
| `per_airport_lgbm` | 336.12 | 335.01 | −1.11 |
| `xgb` | 354.31 | 352.22 | −2.09 |
| `cat` | 362.28 | 362.02 | −0.26 |

Refit blend (`tuned_lgbm` 0.4094 + `per_airport_lgbm` 0.5906): OOF **333.02** (Δ **+0.79 s, +0.24%**), season-matched **454.25** (Δ +0.86 s). The change clears the ≥0.1% overall feature gate with the season-matched metric also improving. Post-processing was refit on the new OOF: `mi2_affine_s50` accepted again → LOT-out overall **331.01**, season-matched **452.90**.

**Honest scope note:** the prefix did *not* materially change the hard-row RMSE in the full pipeline (anchor-missing 2,393.6 → 2,389.0) — the global/per-airport models already smooth those rows, so the large dedicated-model gain seen in isolation did not transfer. The realised gain is the modest, CV-verified +0.24% overall.

**Deltas.**

| Comparison | RMSE (s) |
|---|---:|
| v5 leaderboard (Jan+Jul 2026) | **392.59** |
| v4 leaderboard | 396.74 |
| **v5 vs v4** | **−4.15 (−1.05%)** |
| v5 season-matched CV analog (affine OOF, Jan+Jul 2025) | 452.90 |
| leaderboard vs CV analog | −13.3% (favourable) |

**Final standing (official REST API `datacomp.opensky-network.org`, competitionId `bb3693e1-26bc-4a9e-8619-4fe78b4eab0c`).** Best submission **v5 = 392.5911 s**, **rank #170 of 232 teams** (4,766 scored submissions; API fetched 2026-10-07). Note: the community Vercel board linked below lists only 130 teams and **omits thoughtful-glove entirely**; the official API is authoritative. All five submissions are present in the bucket with their `_result`/`_persist` sidecars.

**Reproducibility:** `uv run python src/run_blend.py --phase oof --ignore-gate` → `--phase weights` → `uv run python src/run_micro_iterations.py` → `--phase serve --version 5` → `--phase reblend --version 5 --candidate mi2_affine_s50`. Feature added in `src/pdc26/features.py` (`FLIGHT_PREFIX` in `CAT_FEATURES`, `_flight_prefix()`); configs `src/configs/blend.json` + `blend_postproc.json`; reports `src/reports/blend_results.md`, `micro_iterations.md`, `improvement_round.md`.

**Artifact:** `submissions/thoughtful-glove_v5.parquet`, 4,707,505 bytes, SHA256 `72dfbfae816f99faab2b71536b87db56edbe35adce0d461d44c162df74bac4da`; 344,841 rows = `submitting.parquet` (order-exact `MVT_ID_mvt`), 0 nulls, all finite, range 1.0–42,905.8 s, mean 1,038.5 s. Archived at `submissions/final/thoughtful-glove_v5.parquet` and committed to the repo.

---

## Material Finding — categorical vocabulary skew (fixed in the serving path for v2)

**Part of the CV→leaderboard degradation is a train/serve skew bug** — initially thought to be the whole story; the full attribution is in the next section.

`pdc26/features.py::build_features()` encodes the 7 native-LightGBM categoricals with

```python
pdf[col] = pdf[col].astype("category").cat.codes.astype("int32")
```

Pandas builds `categories` from the **sorted unique values present in the DataFrame passed in**. `run_baseline.py` calls `build_features()` twice — once on the full 2025 training set, once on `ranking.parquet` — so each call invents its own value→code mapping. Codes are therefore **dataframe-local**, and the two mappings disagree. The saved booster splits on *training* codes; scoring feeds it *ranking* codes.

**Measured code mismatch on values present in both datasets (ranking DEP vs full 2025 DEP):**

| Categorical | Train uniq | Rank uniq | Shared | Codes disagree | % |
|---|---:|---:|---:|---:|---:|
| `RUNWAY_mvt` | 53 | 50 | 50 | 50 | **100.0%** |
| `AIRCRAFT_TYPE_mvt` | 270 | 200 | 192 | 191 | **99.5%** |
| `STAND_mvt` | 1,900 | 1,698 | 1,668 | 1,645 | **98.6%** |
| `AIRCRAFT_OPERATOR_flt` | 677 | 580 | 550 | 534 | **97.1%** |
| `ADEP_mvt` | 10 | 10 | 10 | 0 | 0.0% |
| `MARKET_SEGMENT_flt` | 9 | 9 | 9 | 0 | 0.0% |
| `FLIGHT_TYPE_flt` | 6 | 6 | 6 | 0 | 0.0% |

Four high-cardinality features — carrying most of the airport/stand/runway/operator signal — arrive at scoring time **relabelled**. The three unaffected features are low-cardinality and stable across years, which is why the score degrades rather than collapsing.

**Controlled quantification** (identical 2025-07 rows, known truth, same model — only the vocabulary source differs; 190,713 rows):

| Encoding | RMSE (s) |
|---|---:|
| Global vocabulary (as the model was fed at training) | 269.85 |
| Dataframe-local vocabulary (the bug) | 440.70 |
| **Isolated bug cost** | **+170.85 s (+63.3% relative)** |

*Caveat: both arms are in-sample (July sits in the training year), so the absolute levels are optimistic and the +63% figure overstates the real cost. Measured against the actual scoring rows in v1→v2, the true cost is **+94.72 s (−16.0%)** — see Gap Attribution.*

On the actual scoring rows, buggy vs corrected predictions differ by RMSE 280.6 s (Pearson r = 0.754); correcting the mapping also pushes some predictions outside [0, 7200] s, so the clip-and-log branch in `make_submission.py` becomes load-bearing (74 rows clipped in v2).

**Fix implemented for the serving path** (Task 2.2a scope, done here because it needed no retraining): `make_submission.py --fit-vocab` derives the vocabulary from the training frame, **self-verifies that it reproduces `build_features()`'s training codes exactly for all 7 categoricals**, and persists it to `models/baseline_lgbm_vocab.json` (60.8 KB, beside the booster). Scoring loads that file and rebinds the ranking codes, sending unseen values to −1 (LightGBM's missing marker for native categoricals). Scoring **refuses to run** without it unless `--no-vocab` is passed, so the skew cannot silently return. Unseen rates in v2: `STAND_mvt` 284 (0.08%), `AIRCRAFT_OPERATOR_flt` 177 (0.05%), `AIRCRAFT_TYPE_mvt` 18 (0.01%), all others 0.

---

## Gap Attribution — resolved: there is no 2026 distribution shift

The v1 gap (+342.18 s over CV) decomposes **exactly** into four terms. The decisive experiment: score the **existing OOF predictions** (2,085,047 rows, already out-of-fold) against **raw unclipped** truth instead of the clipped target the CV harness used.

| Cause | Size | Evidence |
|---|---:|---|
| Categorical encoding bug | +94.72 s | v1 591.45 → v2 496.73, same booster |
| **CV scores a clipped target; leaderboard scores raw** | +115.91 s | 2025 OOF: **249.27 s** vs clipped · **365.18 s** vs raw |
| Season mix (all-year CV → Jan+Jul scoring) | +135.70 s | 2025 Jan+Jul OOF vs raw = **500.88 s** |
| **Genuine 2025→2026 drift** | **−4.15 s** | v2 496.73 vs 500.88 comparator = **−0.83%** |
| **Total** | **+342.18 s** | matches the observed gap exactly |

**The headline CV number was measuring an easier quantity than the competition scores.** `build_features()` clips the target to [1, 14400] s, and CV RMSE was computed on that clipped column, while the ranking script evaluates raw `TAXITIME_SEC_mvt`. Raw truth in the 2025 training year runs from **−12 s to 131,167 s** (36 hours — plainly a data error), with 122 rows above 14,400 s and 388 below 1 s.

**Independent confirmation that inputs did not shift**: scoring-time feature distributions are near-identical to training (`anchor_aobt_sec` mean 1,003.9 → 1,036.4 s, nulls 1.08% → 1.53%, `wake_cat_ordinal` 2.2 → 2.2). The one non-trivial move is anchor spread (std 377 → 444 s, +18%), far too small to explain a 76% error increase — consistent with drift contributing ~0.

**Consequences**

1. The model **generalizes to 2026**. Nothing about the year-over-year assumption in `TODO.md` Risks needs changing beyond recording the real cause. The 5%-gap escalation stays satisfied — attributed, not open.
2. **The CV harness must report RMSE against the raw target**, or at minimum alongside it. Until then every CV number in `baseline_results.md` understates achievable leaderboard error by ~46%, and any comparison that turns on outlier treatment is invalid. Affects Tasks 2.4, 3.1, 3.2, 3.3, 3.4.
3. **The dominant remaining lever is the tail, not features.** 721 rows (0.03% of training) carry **60.3% of total squared error**; excluding them, RMSE is 230.00 s vs 365.18 s overall. Halving tail error projects 496.73 s → **~415 s**. Promote outlier handling (Task 3.3's clip/winsorize work) above further feature engineering, and test whether training on raw rather than clipped targets lets the model express the learnable part of the tail.
4. The [0, 7200] s prediction clip is directionally wrong for this metric — it is tighter than the training target's own clip and truncates genuinely long taxi-out predictions (74 rows moved in v2). Cheap to relax; bundle with the tail work rather than spending a slot alone.

---

## Submission Mechanics (Task 05 Goal 4)

**(a) Result-file latency — near-instant, ~22 s.** Both stamps below come from the same bucket clock, so the delta is reliable:

| Event | Bucket clock | Δ from upload |
|---|---|---:|
| `thoughtful-glove_v1.parquet` uploaded | 11:31:46 | — |
| `thoughtful-glove_v1.parquet_result.json` appeared | 11:32:08 | **+22 s** |
| `thoughtful-glove_v1.parquet_persist.json` appeared | 11:32:23 | +37 s |

Scoring is effectively synchronous. Two sidecar files are written per submission:
- `_result.json` → `{"status":"Succeeded","used_pairs":344841,"score":591.4486,"inputs":{"test_set_bucket":"prc-2026-testsets","test_set_file":"truthing.parquet"}}`
- `_persist.json` → `{"status":"Succeeded","team":"thoughtful-glove",...}`

No polling loop is needed — a single check ~1 min after upload suffices. Retrieval: `mc cp opensky/prc-2026-thoughtful-glove/<file>_result.json` and read `score`. The hidden truth set is `prc-2026-testsets/truthing.parquet`.

**(b) Scoring rule — BEST submission, not latest.** This determines final-submission strategy (Task 4.2 / `tasks/14-final_submission.md`). Quote, [ranking page](https://prc-data-challenge-2026.netlify.app/ranking.html):

> "Teams will be ranked based on the best Root Mean Square Error (RMSE) achieved across their submissions."

Corroborated by the [leaderboard](https://prc-challenge-2026.vercel.app/), whose header reads "Best submission per team." with columns `# | Team | Best | Subs | Last activity`.

**Consequence:** there is **no downside risk to experimenting** — a failed submission cannot overwrite a good score, and submitting early/often is free. Strategy: keep every version uploaded; no end-of-competition "sacrificial" submission concern. Our current best (and only) stands at 591.45 s until v2 lands.

**(c) Rate limit — 5 submissions/day, 1 GB/bucket.** Quote:

> "There is currently a limit of 5 submissions per day (and a total max of 1GB per bucket.)"

v1 consumed 4.5 MiB of the 1 GB budget, so ~200+ submissions fit by volume — **the daily cadence is the binding constraint, not storage**. Plan ≤5 scoring runs/day; batch candidate models and submit deliberately.

**Validation rules** (the ranking script errors on any of): an `MVT_ID_mvt` that does not match, missing rows, or extra rows. `make_submission.py` asserts row count, null-freeness, and finiteness before upload, which covers all three.

**(d) CV-vs-leaderboard gap:** v1 +137.3% relative vs overall CV, +109.1% vs season-matched CV — exceeded the 5% threshold and was escalated. **Fully attributed** in the Gap Attribution section: encoding bug + clipped-vs-raw metric + season mix, with genuine drift ≈ 0. Re-measured on v2: 496.73 s sits *below* the honest 2025 season-matched comparator (500.88 s), so the serving path now tracks CV expectations once the metric is stated on the raw target.

**Also noted on the ranking page** — monitoring of submissions for exploitation of the ranking process:

> "We will monitor submissions for attempts to learn from or exploit the ranking process. We consider such practices unfair and inconsistent with the purposes and goals of the Data Challenge."

Reinforces the project rule already in TODO: model selection is driven by month-blocked CV, with the leaderboard used only as confirmation — not as a feedback oracle for repeated score-seeking.

---

## Reproducibility Notes

- v1 asserted clean before upload: 344,841 rows = `submitting.parquet` row count, 0 nulls, all finite, **0 values clipped** (observed range 125.5–6,919.3 s, mean 987.7 s, std 327.3 s).
- Output dtype is `float64` for `TAXITIME_SEC_mvt` (template declares `Int32`); accepted by the scorer, which returned `Succeeded` on all 344,841 pairs. Retained as float for later versions.
- v2 asserted clean before upload on the same four checks; only the clip branch differs — 74 values above 7,200 s clipped (raw model range 1.0–15,891.6 s; after clip mean 1,028.2 s, median 959.1 s, std 406.8 s). Both v1 and v2 scored `Succeeded` with `used_pairs: 344841`.
- Reproducing v2: `python src/make_submission.py --fit-vocab` once (writes `models/baseline_lgbm_vocab.json`), then `python src/make_submission.py --version 2`.
- v3 asserted clean before upload on the same four checks: 344,841 rows = `submitting.parquet`, 0 nulls, all finite, min floored to 1.0 s (no upper clip); observed range 1.0–39,905.4 s, mean 1,035.8 s, std 492.3 s. Uploaded via the Mac Mini → MacBook TCP relay (SHA256 `d962e4a0…`, 4,707,497 bytes, verified on both machines) and scored from the MacBook (MinIO credentials live there).
- `.gitignore`: `submissions/` was replaced with `submissions/*` + `!submissions/*.md` so this log is tracked while parquet payloads stay ignored. Verified after the change that `thoughtful-glove_v1.parquet`, `data/`, `models/` and `resources/credentials.json` remain ignored (Task 01 criteria still satisfied).
- Diagnostics run during this session: `src/make_submission.py` (submission build) plus throwaway scripts under `/tmp` (`diag_enc.py`, `diag_controlled.py`, `diag2.py`, `drift.py`, `clip_effect.py`) — not retained as deliverables; their results are reproduced in the tables above. `clip_effect.py` is the one worth promoting into the repo (raw-target CV metric), tracked as Open Item 1.

## Open Items

1. ~~**CV harness must report raw-target RMSE** (and win rate vs a constant / anchor baseline) so model selection optimises the quantity the competition scores. Highest priority — every ranking so far used the clipped metric.~~ **CLOSED by Task 2.2a:** `cv.py` now reports `overall_rmse_raw`/`per_month.rmse_raw` alongside the clipped figures, plus a `constant_baseline_rmse_raw` for interpretability. `run_baseline.py`—re-run — raw RMSE **365.26s** (clipped 249.28s; constant baseline 546.36s). Both metrics in the report per-month table. Season-matched (Jan+Jul) raw RMSE **500.32s** computed as the CV analog of the leaderboard.
2. ~~**Finish Task 2.2a in the pipeline itself:** move the fit-on-train / apply-on-score vocabulary into `pdc26/features.py` and persist it from `run_baseline.py`, so CV, blending and every feature task share one mapping instead of only the submission script. Add the raw-target CV metric in the same pass, then re-run the 12 folds to get a trustworthy baseline.~~ **CLOSED by Task 2.2a:** `fit_categorical_vocab()`/`apply_categorical_vocab()`/`persist_vocab()`/`load_vocab()` live in `pdc26/features.py`. `run_baseline.py` fits vocab once, passes it to both training and ranking `build_features()` calls, and persists to `models/baseline_lgbm_vocab.json`. `make_submission.py` is now a thin wrapper calling the library functions. 12 folds re-run: overall raw RMSE 365.26s (matches reference), season-matched 500.32s. Unseen rates match v2 exactly (STAND 284, AIRCRAFT_TYPE 18, AIRCRAFT_OPERATOR 177).
3. ~~**Tail / outlier handling is the main remaining lever** (60.3% of squared error in 0.03% of rows): test raw vs clipped training targets, relax the [0, 7200] s prediction clip, and consider winsorising only the demonstrable data errors. → promote within Task 3.3.~~ **CLOSED by Task 2.2b:** E-winsorize-errors target treatment adopted. Raw CV RMSE **344.07s** (from 365.07s baseline, −5.75%). Prediction clip relaxed to floor-at-1s, no upper bound. See `src/reports/tail_outliers.md` for full diagnosis and experiment comparison table.
4. ~~`models/baseline_lgbm_vocab.json` lives under gitignored `models/`, matching how the booster itself is handled — Task 4.3 (public repo) must ship the vocabulary-generation step, not just the booster, for reproducibility.~~ **MITIGATED by Task 2.2a:** the vocabulary is now generated by `fit_categorical_vocab()` in `pdc26/features.py`, which is called automatically by `run_baseline.py`. The public repo (Task 4.3) ships the generation code, not the artifact. Re-verified after 2.2a: no new tracked secrets.
5. 3 of 5 submission slots remain today; v1 and v2 both count toward the bucket's 1 GB budget (9.0 MiB used). **Updated 2026-10-03 (Task 3.4):** v3 uploaded (401.73 s). v1/v2 were 09-09 and v3 is 10-03, so the 5/day counter belongs to a new day; bucket volume is now ~13.5 MiB of the 1 GB cap.
6.  **Closed 2026-10-07 (Task 4.2):** v5 (final, FLIGHT_PREFIX) uploaded — 392.59 s, rank #170/232. Bucket volume ~23.5 MiB of 1 GB; 1 of 5 daily slots used on 10-07.
