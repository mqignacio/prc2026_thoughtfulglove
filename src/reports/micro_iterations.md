# Micro-Iterations — Task 4.1 (leave-one-month-out CV verdicts)

Generated: 2026-10-07 12:36 PST · Source: 2025 OOF table, LOT-out (12 folds) · Baseline (reproduction gate): overall **333.02s**, season-matched **454.25s**.

## Verdicts

| Candidate | Change | Params | Overall RMSE | Δ vs base | Season RMSE | Δ season | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| `baseline_global` | recorded global blend (reproduction gate) | 0 | 333.02 | +0.000 | 454.25 | +0.000 | **reference** |
| `mi1_weights` | per-airport 2-component blend weights | 10 | 333.61 | -0.587 | 454.86 | -0.606 | **reject** |
| `mi2_affine` | per-airport affine recalibration of the blend | 20 | 330.77 | +2.251 | 454.72 | -0.462 | **reject** |
| `mi2_affine_s50` | per-airport affine, slope shrunk 50% | 20 | 331.00 | +2.016 | 452.90 | +1.356 | **ACCEPT** |
| `mi2_affine_rwy` | per airport-runway affine recalibration | n/a | 333.06 | -0.036 | 459.70 | -5.445 | **reject** |
| `mi3_weights_affine` | per-airport weights + per-airport affine | 30 | 331.34 | +1.677 | 455.28 | -1.022 | **reject** |

_Acceptance rule (approved): post-processing kept if overall improves and season-matched does not degrade (> −0.01s); feature/model changes need ≥0.1% overall gain. LOT-out is the honest estimate; in-sample fits are not used for selection._

## Per-month RMSE (LOT-out)

| Candidate | 2025-01 | 2025-02 | 2025-03 | 2025-04 | 2025-05 | 2025-06 | 2025-07 | 2025-08 | 2025-09 | 2025-10 | 2025-11 | 2025-12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline_global` | 383.7 | 399.1 | 257.7 | 226.1 | 300.0 | 366.9 | 504.0 | 289.7 | 323.6 | 236.7 | 326.9 | 280.2 |
| `mi1_weights` | 384.2 | 402.1 | 258.6 | 226.4 | 300.1 | 367.2 | 504.7 | 289.9 | 323.8 | 237.0 | 327.4 | 280.2 |
| `mi2_affine` | 377.6 | 391.6 | 251.0 | 228.5 | 301.9 | 362.9 | 508.4 | 287.2 | 317.5 | 244.6 | 319.7 | 273.0 |
| `mi2_affine_s50` | 380.3 | 395.1 | 253.9 | 226.2 | 300.4 | 364.3 | 503.9 | 287.1 | 319.7 | 239.6 | 322.9 | 276.0 |
| `mi2_affine_rwy` | 385.5 | 392.4 | 251.4 | 228.6 | 301.9 | 365.3 | 511.7 | 291.0 | 316.8 | 253.5 | 316.0 | 277.0 |
| `mi3_weights_affine` | 378.1 | 394.8 | 251.8 | 228.8 | 302.0 | 363.1 | 509.0 | 287.4 | 317.7 | 244.9 | 320.3 | 273.0 |

## Full-12-month fits written for accepted candidates

- `mi2_affine_s50`: per-airport affine, slope shrunk 50% (Δ overall +2.016s, Δ season +1.356s)

## Interpretation

- The blend's squared error is dominated by a small number of metadata-missing (no operator ID) special flights at LIRF (see `error_analysis.md` §6).
- Per-airport affine recalibration tests whether the model systematically compresses the heavy upper tail; LOT-out decides if any such expansion generalises.
- If no candidate passes, the blend is already RMSE-calibrated and the tail is treated as irreducible with the ranking-available features.
