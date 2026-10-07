"""pdc26 — PRC-OpenSky Data Challenge 2026 baseline pipeline.

Modules:
    data      — Load and prepare training/ranking parquet data.
    features  — Baseline feature engineering (temporal + anchors + categoricals).
    cv        — Month-blocked temporal cross-validation harness.
    train     — LightGBM trainer and ranking scorer.
"""

SEED = 42
