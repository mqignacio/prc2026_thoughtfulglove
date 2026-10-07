# Improvement Round — post-processing calibration battery (Task 4.2)

Generated: 2026-10-07 11:55 PST · LOT-out on the 2025 OOF (12 folds) · Baseline overall **333.815s**, season-matched **455.110s**.

| Candidate | Overall RMSE | Δ overall | Season RMSE | Δ season | Verdict |
|---|---:|---:|---:|---:|---|
| `baseline_global` | 333.815 | +0.000 | 455.110 | +0.000 | **reference** |
| `v4_affine_s50` | 331.296 | +2.519 | 453.666 | +1.443 | **ACCEPT** |
| `affine_s25` | 332.297 | +1.518 | 453.880 | +1.230 | **ACCEPT** |
| `affine_s75` | 330.855 | +2.960 | 454.488 | +0.621 | **ACCEPT** |
| `iso_airport` | 333.395 | +0.420 | 461.825 | -6.715 | **reject** |
| `iso_global` | 333.217 | +0.598 | 461.702 | -6.592 | **reject** |
| `decile_affine_ap` | 330.970 | +2.844 | 453.889 | +1.221 | **ACCEPT** |
| `qmap_airport` | 646.500 | -312.685 | 1094.330 | -639.220 | **reject** |
| `cap_ap_p995` | 395.884 | -62.069 | 544.774 | -89.664 | **reject** |

_Acceptance rule: overall improves AND season-matched does not degrade._
