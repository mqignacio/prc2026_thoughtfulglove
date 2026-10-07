# Feature-Pruning Check — Task 4.1

Generated: 2026-10-03 20:08 PST · Drop list derived from saved full-retrain gainer gain importance (62 features).

## Dropped features (individual gain share < 0.05%)

| Feature | Gain share (%) |
|---|---:|
| `local_month` | 0.0347 |
| `flight_type_code` | 0.0333 |
| `FLIGHT_TYPE_flt` | 0.0228 |
| `metar_wind_gust_kt` | 0.0127 |
| `nm_matched` | 0.0121 |
| `metar_flight_cat_ord` | 0.0109 |
| `diversion_flag` | 0.0018 |
| `flight_rule_ifr` | 0.0017 |
| `local_weekend` | 0.0009 |

**9 features dropped, cumulative gain 0.131%** (kept 53).

## CV verdict (two active components, month-blocked)

| Model | Overall RMSE (s) | Δ vs baseline | Season RMSE (s) | Δ vs baseline |
|---|---:|---:|---:|---:|
| baseline blend (62 features) | 333.815 | — | 455.110 | — |
| pruned blend (53 features) | 334.730 | -0.915 | 458.284 | -3.174 |
| pruned + affine s50 (LOT-out) | 332.615 | +1.199 | 457.470 | -2.361 |

**Verdict: REJECT — degrades too much**

_Acceptance rule: pruning kept if OOF degrades <= 0.05% overall and season-matched._
