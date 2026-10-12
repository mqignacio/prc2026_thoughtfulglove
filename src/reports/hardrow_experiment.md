# Hard-Row Experiment (Task 19) — Pre-Registered Protocol

**Status:** protocol pre-registered 2026-10-11 16:46 +0800 (Asia/Manila; the `%z` offset is authoritative — macOS labels the zone `PST`, see TODO audit A3) · **Runner:** `src/run_hardrow_experiment.py` (not yet run at registration time) · **Approval:** human approved the session plan 2026-10-11; this protocol is frozen before any model run.

> This file was committed with ONLY this protocol section. All result sections below the rule were appended **after** the runs, and every number in them is script output. No gate rule, fold definition, metric definition, or candidate was changed after the first run; any deviation is disclosed inline.

---

## 1. Hypothesis

A dedicated treatment of the hard rows (DEP rows whose `AIRCRAFT_OPERATOR_flt` is null; equivalently the fully-null `_flt` block, verified identical in Task 17) reduces their out-of-fold raw-target RMSE enough to improve the 4-month analog (Jan+Feb+Jun+Jul 2025 OOF folds) by **≥ 1%**, **and** that gain reflects signal learnable from ranking-available columns rather than irreducible corrupted-target noise.

The corrupting structure (recomputed 2026-10-11 from the 2025 OOF + training parquets, values matching `resources/external_research-resubmission.md` §3 and TODO): hard rows are **1.08%** of 2025 DEP rows (22,470 / 2,085,047) and **1.34%** of the 4-month analog rows, carrying **64.7%** of analog squared error (hard raw RMSE **2,922.74 s** vs base **251.43 s**). At LIRF the 1,488 hard rows have median 3,922 s, mean 6,530.6 s, p99 60,264.5 s, max 131,167 s; 0.87% exceed 86,400 s (block-time date-rollover artifacts).

## 2. Signal test — HARD STOP (runs first, before any acceptance modeling)

**Procedure.** Train a dedicated LightGBM on **only** the 22,470 null-operator training DEP rows (2025), leave-one-month-out through the authoritative harness (`pdc26.tuning.month_blocked_cv_model`, family `lgbm`, repo-default params `family_default_params("lgbm", 42)`; ≤ 500 rounds, early stopping 50 — **no tuning toward the gate**). Features are the single global D4 matrix built once by `run_blend.BlendData` (fixed global categorical vocabulary, fixed feature order, within-fold operator target encoding) restricted to hard rows — compliance with lesson #1 in `~/AGENT-LESSONS.md`. Training target = `winsorize_errors` applied to the hard-row subset (the pipeline's target treatment, which caps the ~24 h rollover artifacts at the airport P99 of the hard subset); evaluation metric = **raw-target RMSE** on the held-out month's hard rows. All 12 folds are run; the gate uses the 4 analog folds (Jan/Feb/Jun/Jul 2025).

**Comparator.** The existing global blend's OOF hard-row RMSE, per fold, on the identical rows. Reconstructed from `data/processed/oof_predictions.parquet` with the recorded `src/configs/blend.json` active weights; reproduction asserted before use (all-2025 333.0208 s, 4-month analog 420.3667 s, both within 0.01 s of the recorded/quoted values).

Pre-registered comparator values (recomputed 2026-10-11):

| Analog fold | n hard rows | blend hard-row RMSE (s) |
|---|---:|---:|
| 2025-01 | 1,435 | 3,323.13 |
| 2025-02 | 1,110 | 3,698.58 |
| 2025-06 | 2,502 | 2,500.04 |
| 2025-07 | 3,938 | 2,763.96 |
| **Pooled 4-month** | **8,985** | **2,922.74** |

**HARD-STOP RULE.** If the dedicated model does **not** beat the blend's hard-row RMSE in **≥ 3 of the 4** analog folds, **STOP and REJECT immediately** — the corruption is treated as irreducible/untransferable, no acceptance modeling is attempted, and no candidate (b)/(c) is run.

## 3. Primary and secondary metrics

- **Primary:** LOT-out 4-month analog RMSE — pooled Jan+Feb+Jun+Jul 2025 OOF folds, raw target. Baseline (recomputed): **420.3667 s** on 671,293 rows (per fold 383.74 / 399.15 / 366.90 / 503.96).
- **Secondary:** 2-month analog (Jan+Jul, baseline 454.2545 s); all-2025 (baseline 333.0208 s); per-fold hard-row RMSE; hard-row share of squared error under the treatment; post-affine 4-month analog (baseline 417.52 s — context only, **not** a gate).

## 4. Acceptance gate (all four must hold; base case = REJECT)

Let "treated OOF" = the blend OOF with hard-row predictions replaced by the dedicated model's OOF predictions; base rows are mathematically unchanged by construction.

1. **Primary gain:** treated 4-month analog **≤ 416.1663 s** (≥ 1.0%, i.e. ≥ 4.2037 s improvement vs 420.3667 s).
2. **Fold consistency:** per-fold treated analog RMSE is strictly lower than the per-fold baseline in **≥ 3 of the 4** analog folds.
3. **Hard-row gain:** pooled 4-month-analog hard-row RMSE **< 2,922.74 s**.
4. **Base-row invariance:** the treated base-row prediction vector is byte-identical to the baseline blend's base-row vector (asserted by array equality, not prose).

**REJECT if any condition fails.** No post-hoc changes to the gate or fold definition. Do not tune toward a pass.

## 5. Candidate treatments (in order; stop at first gate clear or time-out)

- **(a) Dedicated hard-row sub-model** — the signal-test model, served on final rows with the same null-operator signature. On hard rows, **REPLACE** (the dedicated prediction is the prediction) is primary; if REPLACE fails the gate, one pre-registered fallback is a convex blend `w · dedicated + (1 − w) · blend` with **w ∈ {0.25, 0.5, 0.75}** selected by the gate metric.
- **(b) Corruption-aware hard-row features** — only if (a) shows signal at the hard stop but falls short of the full gate, AND ≥ 45 min of the time-box remain. Indicators derived from ranking-available columns only, added for hard rows via the sub-model; same gate.
- **(c) Two-part (hurdle) treatment** — only if (a) shows signal but is unstable (fold RMSE swing > 30%). Same gate.

Candidate (a) is fully evaluated before (b)/(c). No parameter search beyond the pre-registered weight grid.

## 6. Serving design (applies only if ACCEPT)

- Train the dedicated model on **all** 2025 hard rows (22,470), with rounds = the median CV `best_iteration` across the 12 folds (repo convention).
- `final_v2` = the Task 18 `final_v1` values with the operator-null final rows replaced by the dedicated prediction floored at 1 s (no upper clip). Base rows byte-identical to `final_v1`.
- Re-run Task 18's pre-upload assertions (670,790 rows, order-exact vs `final_submitting`, 0 nulls, finite, ≥ 1 s), distribution sanity on the changed rows, and the two-build SHA256 reproduction gate.
- `run_blend.py` and `pdc26/` remain byte-identical; the Task 18 default build is unchanged (the v2 path is an additive flag).

## 7. Honest-transfer disclosure (fixed text, applies to either verdict)

The 2025 analog target contains the corrupted tail. A gate pass therefore demonstrates **within-2025 learnability** of that tail, **not** transfer to the 2026 scored truth. The leaderboard floor arithmetic in the audit (`resources/external_research-resubmission.md` §3; TODO) is the reason to expect near-zero final-phase gain: if the scored truth carried the same tail, a perfect base model would floor at ≈ 338.1 s (4-month analog), yet the Jan/Jul leader is 213.1814 s, below that floor — so the scored truth is most likely materially cleaner. If the truth is clean, replacing ~1,000 s predictions with corruption-fitted ~3,000 s predictions on ~1.35% of rows would add ≈ +68 s RMSE to the ≈ 363 s expected baseline. The final artifact choice (v1 vs v2) belongs to Task 20, informed by this report.

## 8. Constraints honored

Time-box ≤ 3 wall-clock hours (signal test ≤ 45 min). No leaderboard tuning. No EUROCONTROL aggregates and no scored-month aggregate in the treatment (decision D2/Q2-A). No retraining of the base models (base-row predictions must remain byte-identical). Authoritative CV harness only. Trained on 2025 data only, described honestly for the repo/paper.

---

*Result sections appended below after the runs.*

---

### A. Reproduction gate (run before any experiment number was trusted)

LOT-out baselines recomputed from `data/processed/oof_predictions.parquet` with the recorded `blend.json` weights, asserted within 0.01 s of the pre-registered values (script output, `logs/hardrow_experiment.log`):

| Baseline | Recomputed (s) | Pre-registered (s) |
|---|---:|---:|
| all-2025 blend RMSE | 333.0208 | 333.0208 |
| 4-month analog RMSE | 420.3667 | 420.3667 |
| 2-month analog RMSE | 454.2545 | 454.2545 |

2025 hard rows: **22,470 / 2,085,047 (1.0777%)**. Global D4 matrix: 2,085,047 rows × 63 cols. Hard-row CV target mean: raw 1,384.4 s / winsorised 1,363.0 s. The run completed in **2.3 min** (well inside the 45-min signal-test budget).


### B. Signal test (hard stop) -- dedicated hard-row LightGBM vs the blend

Leave-one-month-out; trained only on the 22,470 null-operator 2025 rows; raw-target RMSE on the held-out month's hard rows.

| Fold | n hard | blend hard RMSE (s) | dedicated hard RMSE (s) | Δ (s) | dedicated wins |
|---|---:|---:|---:|---:|:---:|
| 2025-01 * | 1,435 | 3,323.13 | 3,353.19 | -30.06 | no |
| 2025-02 * | 1,110 | 3,698.58 | 3,382.25 | +316.32 | **yes** |
| 2025-03 | 1,082 | 1,995.76 | 1,394.45 | +601.31 | **yes** |
| 2025-04 | 1,085 | 1,426.99 | 940.14 | +486.85 | **yes** |
| 2025-05 | 1,605 | 2,382.38 | 2,448.56 | -66.18 | no |
| 2025-06 * | 2,502 | 2,500.04 | 2,298.48 | +201.56 | **yes** |
| 2025-07 * | 3,938 | 2,763.96 | 2,191.06 | +572.90 | **yes** |
| 2025-08 | 2,330 | 1,684.12 | 1,433.23 | +250.88 | **yes** |
| 2025-09 | 2,465 | 2,073.04 | 1,666.02 | +407.02 | **yes** |
| 2025-10 | 1,913 | 946.99 | 831.63 | +115.37 | **yes** |
| 2025-11 | 1,356 | 2,832.83 | 2,121.05 | +711.78 | **yes** |
| 2025-12 | 1,649 | 1,917.49 | 1,692.15 | +225.35 | **yes** |
| **Pooled analog** | 8,985 | 2,922.74 | 2,604.66 | +318.08 | **yes** |
| _All 2025_ | 22,470 | 2,389.01 | 2,082.15 | -- | -- |

*\* = analog fold. Pre-registered signal gate: the dedicated model must beat the blend in **≥ 3 of the 4** analog folds (Jan/Feb/Jun/Jul).*

**Signal-test result: PASS — dedicated wins 3/4 analog folds (10/12 all folds).**

### C. Acceptance gate

Treated OOF = blend OOF with hard rows replaced by the dedicated model's OOF prediction (or a pre-registered convex blend `w·dedicated + (1−w)·blend`). Base rows unchanged by construction.

| Candidate | 4-month analog (s) | vs base | gain | folds improved | hard RMSE (s) | base identical |
|---|---:|---:|---:|---:|---:|:---:|
| replace | 391.38 | -28.99 | +6.90% | 3/4 | 2,604.66 | yes |
| blend_w0.25 | 409.61 | -10.75 | +2.56% | 4/4 | 2,806.35 | yes |
| blend_w0.50 | 401.11 | -19.26 | +4.58% | 4/4 | 2,712.98 | yes |
| blend_w0.75 | 394.99 | -25.38 | +6.04% | 3/4 | 2,645.09 | yes |

Pre-registered gate (all must hold): analog ≤ **416.16 s**; improvement in ≥ 3/4 folds; hard-row RMSE < **2,922.74 s**; base rows byte-identical.

### D. Secondary metrics (chosen candidate: **replace**)

| Metric | Baseline | Treated | Δ |
|---|---:|---:|---:|
| 4-month analog RMSE (s) | 420.3667 | 391.3768 | +28.9899 |
| 2-month analog RMSE (s) | 454.2545 | 418.0023 | +36.2522 |
| all-2025 RMSE (s) | 333.0208 | 310.0275 | +22.9933 |
| hard-row RMSE, analog (s) | 2,922.74 | 2,604.66 | +318.08 |
| hard-row SE share, analog | 64.7% | 48.6% | -- |
| 4-month analog, post-affine variant (s) | 417.52 | 390.86 | +26.66 |

Per-fold 4-month analog RMSE (raw blend space):

| Fold | baseline (s) | treated (s) | Δ (s) |
|---|---:|---:|---:|
| 2025-01 | 383.74 | 386.17 | -2.43 |
| 2025-02 | 399.15 | 376.85 | +22.29 |
| 2025-06 | 366.90 | 348.43 | +18.47 |
| 2025-07 | 503.96 | 441.99 | +61.97 |

### E. Verdict

**ACCEPT** — chosen candidate `replace` (selected by the pre-registered order: replace, then w ∈ {0.25, 0.5, 0.75}). Gate conditions for the chosen candidate:

1. 4-month analog ≤ 416.16 s: PASS (391.38 s)
2. Improvement in ≥ 3/4 folds: PASS (3/4)
3. Hard-row RMSE < 2,922.74 s: PASS (2,604.66 s)
4. Base rows byte-identical: PASS

**ACCEPT (within-2025).** The gate clears on the 2025 analog — see the honest transfer disclosure in the protocol (§7): this is within-2025 learnability of the corrupted tail, not evidence the 2026 scored truth carries it. The final artifact choice (v1 vs v2) belongs to Task 20.


### F. Serving sanity (final_v2 vs final_v1)

final_v1 SHA256 `e326deac578d92329a28a08d3acb752e136ae66d6ce335f7cc7afe3dea4fc632` · final_v2 SHA256 `a0a3b54adbebc78ab24379085000b6ce4b9901c8062301bac3fa445a93fc4871` · rows 670,790 · changed rows 9,056 of 9,077 hard rows (21 hard rows identical); base rows changed: 0 · base rows byte-identical: PASS.

Delta (v2 − v1) on the 9,077 hard rows:

| stat | mean | median | std | RMS | p1 | p5 | p95 | p99 | max |Δ | |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Δ s | -1.638 | -12.984 | 1084.225 | 1084.226 | -1472.5 | -644.0 | +553.5 | +2566.0 | 31808.0 |

Per-month hard-row predictions (v1 vs v2):

| Month | n hard | v1 mean | v2 mean | mean Δ | RMS Δ |
|---|---:|---:|---:|---:|---:|
| 2026-01 | 2,453 | 1505.6 | 1559.1 | +53.5 | 1149.5 |
| 2026-02 | 1,320 | 1333.8 | 1384.5 | +50.7 | 926.0 |
| 2026-06 | 2,467 | 1278.8 | 1254.6 | -24.2 | 1032.5 |
| 2026-07 | 2,837 | 1473.9 | 1419.8 | -54.1 | 1137.0 |

Per-airport hard-row predictions (v1 vs v2):

| Airport | n hard | v1 mean | v2 mean | mean Δ | RMS Δ |
|---|---:|---:|---:|---:|---:|
| EDDF | 766 | 1027.3 | 1094.6 | +67.3 | 849.8 |
| EDDM | 449 | 1200.7 | 1219.4 | +18.7 | 802.0 |
| EGLL | 471 | 1621.6 | 1610.5 | -11.0 | 483.7 |
| EHAM | 1,894 | 876.0 | 908.9 | +32.9 | 968.8 |
| LEBL | 759 | 1114.2 | 1100.7 | -13.5 | 616.1 |
| LEMD | 438 | 1140.7 | 1109.7 | -31.0 | 273.3 |
| LFPG | 1,305 | 1186.6 | 1179.8 | -6.7 | 510.9 |
| LIRF | 619 | 6144.6 | 5961.3 | -183.3 | 3319.1 |
| LSZH | 1,047 | 793.6 | 820.6 | +27.0 | 598.0 |
| LTFM | 1,329 | 1138.5 | 1127.8 | -10.7 | 253.6 |

Overall prediction stats — v1: mean 1016.2 median 947.3 std 465.9; v2: mean 1016.2 median 946.8 std 465.0.

**Interpretation (honest; the aggregate mean hides the row-level change).** The swap changes 9,056 of 9,077 hard rows by RMS **1,084 s** per row, yet the marginal means barely move (v1 1016.2 vs v2 1016.2) — the reallocation is near-symmetric (mean -1.6 s). Which direction helps depends on the unknown 2026 hard-row truth, and there is no leaderboard feedback for the final months:

- If the 2025 corruption mechanism transfers, the analog gain is 6.90%, i.e. ≈ **−25.0 s** on a 363 s expected final.
- If the 2026 truth is clean and the reallocation is uncorrelated with the residual (the audit's lean: leader 213.18 s < the 338.1 s 4-month floor), it adds ≈ 15,907 MSE/row → ≈ **+21.3 s** on the 363 s baseline.

The two are similar in magnitude and of opposite sign, so this is a genuine judgment call for Task 20, not a mechanical 'v2 is better'.

---

### G. Summary, artifact recommendation, and reviewer checklist

*(Author prose; all numbers above are script output. This section makes no new numeric claims.)*

**What the experiment found.** The dedicated hard-row LightGBM clears the pre-registered signal test (beats the blend on held-out hard rows in 3 of 4 analog folds — losing January by 30 s, winning Feb/Jun/Jul by 202–573 s) and clears the full 4-month-analog gate with the `replace` candidate: **391.38 s vs 420.37 s (+6.90%)**, hard-row RMSE 2,604.66 s vs 2,922.74 s, base rows byte-identical. So, by the pre-registered rule, the verdict is **ACCEPT (within 2025)**.

**Why that is not a green light to submit v2.** The 2025 analog target is the corrupted one (LIRF median 3,922 s, ~24 h rollovers). The gate rewards fitting that corruption, and the audit's floor arithmetic says the scored 2026 truth is probably cleaner (leader 213.18 s < the 338.1 s 4-month floor a perfect base model would hit if the tail held). On the actual final rows the swap is not small: it moves 9,056 of 9,077 hard rows by RMS 1,084 s, even though the aggregate mean barely changes. Under the audit's lean (clean truth, reallocation ≈ noise) the expected effect is ≈ **+21 s**; only if the 2025 corruption mechanism transfers does it become ≈ **−25 s**. The v5 leaderboard evidence (392.59 s scored vs 454.25 s analog, −13.6%) further indicates the 2026 scored truth does not behave like the 2025 analog.

**Recommendation for Task 20 (the artifact choice belongs there).** Submit **`thoughtful-glove_final_v1.parquet`** (Task 18), not v2. The gate's ACCEPT is real but measures within-2025 learnability of a corruption that the evidence says does not transfer; swapping in v2 is a near-symmetric bet with a mildly negative expected value and an unbounded downside (if the truth is clean, the corrupted-tail predictions add error). Keep v2 built and documented as the experiment's artifact so the reasoning is inspectable; do not let a gate designed on corrupted targets override the transfer evidence. If you prefer to honour the gate literally, v2 is ready and validated at `submissions/final/thoughtful-glove_final_v2.parquet`.

**What a reviewer should check.**

1. The protocol in §1–§8 was committed (`0572474`) before the runner existed; `git log` shows the protocol commit precedes every result. No gate rule, fold set, or candidate changed after the first run.
2. The signal test is the harness model (`pdc26.tuning.month_blocked_cv_model`) on the one global D4 matrix from `run_blend.BlendData` — one fixed vocabulary, one fixed feature order (lesson #1). The 4-month-analog baselines are reproduced to 0.01 s (§A) before any experiment number is used.
3. The gate arithmetic is per-fold out-of-fold (§B, §D), not in-sample; the January fold is shown losing, and the ≥ 3/4 rule is what passes.
4. Base-row invariance is asserted mechanically (`np.array_equal`), not by prose: 661,713 base rows byte-identical; the v2 SHA256 is unchanged across two builds (§F).
5. The honest-transfer framing (§7 and §G) is the intended reading — an ACCEPT here is a statement about the 2025 analog, not a prediction of a final-phase gain. The at-best-ambiguous transfer is the task's declared base case, now measured.

**Deviation disclosure.** None from the pre-registered protocol. The only mid-session correction was to the serving-sanity *text*: a first draft called the hard-row swap "negligible" from the aggregate means; the per-row RMS (1,084 s) contradicted that, and the section was rewritten before it was committed. The gate and its numbers were never re-touched.
