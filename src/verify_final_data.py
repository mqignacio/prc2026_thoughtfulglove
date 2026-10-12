"""Final-phase data verification — PRC-OpenSky Data Challenge 2026 (Task 17).

Runs every scripted check from tasks/17-final_data_verification.md against the
final-phase files (``final_ranking.parquet`` / ``final_submitting.parquet``) and
prints a full report to stdout. Paste the output into
``src/reports/final_data_verification.md``.

The most important check is (4): whether the near-oracle anchor columns
(``AOBT_3_flt`` and the other NM off-block times) survive in the final files.
If they do not, the pipeline design is invalid and the session must STOP and
escalate to the human user.

Usage:
    uv run python src/verify_final_data.py
"""

from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")

FINAL_RANKING = os.path.join(RAW, "final_ranking.parquet")
FINAL_SUBMITTING = os.path.join(RAW, "final_submitting.parquet")
RANKING = os.path.join(RAW, "ranking.parquet")
SUBMITTING = os.path.join(RAW, "submitting.parquet")
METAR = os.path.join(PROCESSED, "metar_features.parquet")
HOLIDAY = os.path.join(PROCESSED, "holiday_features.parquet")

# Object sizes observed in the bucket listing (2026-10-08 11:04 UTC upload).
BUCKET_SIZES = {
    "final_ranking.parquet": 85_853_399,
    "final_submitting.parquet": 3_003_722,
}
EXPECTED_MONTHS = ["2026-01", "2026-02", "2026-06", "2026-07"]

ANCHOR_COLS = ["AOBT_3_flt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "ARVT_1_flt"]
FLT_COLS = [
    "LOBT_flt", "CALLSIGN_flt", "ADEP_flt", "ADES_flt", "ADES_FILED_flt",
    "MARKET_SEGMENT_flt", "IOBT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt",
    "AIRCRAFT_TYPE_flt", "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
    "EOBT_1_flt", "ARVT_1_flt", "AOBT_3_flt", "ARVT_3_flt",
]

# 2025 reference rates (from src/reports/data_verification.md + error_analysis.md)
REF_AOBT_NULL_PCT = 1.53      # ranking DEP null rate, recomputed in-script too
REF_OPERATOR_NULL_PCT = 1.08  # 2025 OOF missing-operator share


def _hdr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _months(s: pd.Series) -> pd.Series:
    mt = pd.to_datetime(s, errors="coerce", utc=True)
    return mt.dt.strftime("%Y-%m")


def _null_pct(s: pd.Series) -> float:
    return float(s.isna().mean() * 100.0)


def _pop_pct(s: pd.Series) -> float:
    return float(s.notna().mean() * 100.0)


def main() -> int:
    for p in (FINAL_RANKING, FINAL_SUBMITTING):
        if not os.path.exists(p):
            print(f"ERROR: {p} not found — download not done yet.", file=sys.stderr)
            return 1

    # ---------------------------------------------------------------- (0)
    _hdr("(0) FILE INVENTORY — SIZE, SHA256, BUCKET MATCH")
    for name, path in [("final_ranking.parquet", FINAL_RANKING),
                       ("final_submitting.parquet", FINAL_SUBMITTING)]:
        size = os.path.getsize(path)
        exp = BUCKET_SIZES[name]
        print(f"  {name:26s} {_size_mb(path):8.3f} MB  {size:>12,} B  "
              f"bucket={exp:,} B  match={size == exp}")
        print(f"  {'':26s} sha256={_sha256(path)}")

    fr = pq.read_table(FINAL_RANKING).to_pandas()
    fs = pq.read_table(FINAL_SUBMITTING).to_pandas()
    rank = pq.read_table(RANKING).to_pandas()
    sub = pq.read_table(SUBMITTING).to_pandas()

    fr["__ym"] = _months(fr["MVT_TIME_UTC_mvt"])
    rank["__ym"] = _months(rank["MVT_TIME_UTC_mvt"])
    fr_dep = fr[fr["PHASE_mvt"] == "DEP"]

    # ---------------------------------------------------------------- (1)
    _hdr("(1) STRUCTURE — ROWS, MONTHS, PHASE, PER-AIRPORT")
    print(f"  final_ranking rows   = {len(fr):,}")
    print(f"  final_ranking cols   = {fr.shape[1] - 1}")  # minus helper __ym
    months = sorted(fr["__ym"].dropna().unique().tolist())
    print(f"  months present       = {months}")
    print(f"  months == expected   = {months == EXPECTED_MONTHS}")
    print(f"  MVT_TIME null (all)  = {_null_pct(fr['MVT_TIME_UTC_mvt']):.4f}%")
    print("\n  PHASE counts per month:")
    tab = fr.groupby(["__ym", "PHASE_mvt"]).size().unstack(fill_value=0)
    with pd.option_context("display.max_rows", None):
        print(tab.to_string())
    print("\n  Per-reporting-airport rows (DEP by ADEP_mvt, ARR by ADES_mvt):")
    dep_by_ap = fr_dep.groupby("ADEP_mvt").size().sort_values(ascending=False)
    arr_by_ap = fr[fr["PHASE_mvt"] == "ARR"].groupby("ADES_mvt").size().sort_values(ascending=False)
    ap_tab = pd.DataFrame({"DEP": dep_by_ap, "ARR": arr_by_ap}).fillna(0).astype(int)
    ap_tab["TOTAL"] = ap_tab.sum(axis=1)
    print(ap_tab.to_string())
    print(f"\n  distinct DEP ADEP_mvt = {fr_dep['ADEP_mvt'].nunique()} "
          f"(expected 10, match={fr_dep['ADEP_mvt'].nunique() == 10})")

    # ---------------------------------------------------------------- (2)
    _hdr("(2) SCHEMA — final_ranking vs ranking (names + dtypes)")
    f_cols = [c for c in fr.columns if c != "__ym"]
    r_cols = [c for c in rank.columns if c != "__ym"]
    print(f"  name set equal   : {set(f_cols) == set(r_cols)}")
    print(f"  name order equal : {f_cols == r_cols}")
    print(f"  missing in final : {sorted(set(r_cols) - set(f_cols))}")
    print(f"  extra in final   : {sorted(set(f_cols) - set(r_cols))}")
    dtype_mismatch = []
    for c in f_cols:
        if c in rank.columns and str(fr[c].dtype) != str(rank[c].dtype):
            dtype_mismatch.append((c, str(rank[c].dtype), str(fr[c].dtype)))
    print(f"  dtype mismatches : {len(dtype_mismatch)}")
    for c, rd, fd in dtype_mismatch:
        print(f"    {c:28s} ranking={rd:10s} final={fd}")
    print(f"  FINAL SCHEMA ({len(f_cols)} cols):")
    for c in f_cols:
        print(f"    {c:28s} {str(fr[c].dtype)}")

    # ---------------------------------------------------------------- (3)
    _hdr("(3) FINAL_SUBMITTING — ROWS, NULLS, ID ORDER vs final_ranking DEP")
    print(f"  final_submitting rows      = {len(fs):,}")
    print(f"  expected (final DEP rows)  = {len(fr_dep):,}   match={len(fs) == len(fr_dep)}")
    print(f"  columns                    = {list(fs.columns)}")
    print(f"  MVT_ID_mvt nulls           = {int(fs['MVT_ID_mvt'].isna().sum())} "
          f"(expected 0)")
    print(f"  TAXITIME_SEC_mvt null%     = {_null_pct(fs['TAXITIME_SEC_mvt']):.2f}% "
          f"(template placeholder — expected 100%)")
    dep_ids_ordered = fr_dep["MVT_ID_mvt"].tolist()
    fs_ids_ordered = fs["MVT_ID_mvt"].tolist()
    print(f"  row order exact vs final DEP = {fs_ids_ordered == dep_ids_ordered}")
    print(f"  id set equal vs final DEP    = {set(fs_ids_ordered) == set(dep_ids_ordered)}")
    # Jan/Jul overlap with the original submitting.parquet template
    jj_ids = set(fr_dep.loc[fr_dep["__ym"].isin(["2026-01", "2026-07"]), "MVT_ID_mvt"])
    sub_ids = set(sub["MVT_ID_mvt"])
    print(f"\n  final Jan/Jul DEP ids   = {len(jj_ids):,}")
    print(f"  submitting.parquet ids  = {len(sub_ids):,}")
    print(f"  submitting ⊆ final J/J  = {sub_ids.issubset(jj_ids)}")
    print(f"  set equality            = {sub_ids == jj_ids}")
    print(f"  in submitting not final : {len(sub_ids - jj_ids):,}")
    print(f"  in final J/J not submit : {len(jj_ids - sub_ids):,}")

    # ---------------------------------------------------------------- (4)
    _hdr("(4) ANCHOR AVAILABILITY [CRITICAL] — populated% on final DEP rows")
    print(f"  ranking DEP reference (recomputed):")
    rank_dep = rank[rank["PHASE_mvt"] == "DEP"]
    for c in ANCHOR_COLS:
        r_rate = _pop_pct(rank_dep[c])
        print(f"    {c:12s} ranking DEP populated = {r_rate:.4f}%  "
              f"(null {100 - r_rate:.4f}%)")
    print("\n  final DEP populated% per month:")
    rows = []
    for c in ANCHOR_COLS:
        rates = {ym: _pop_pct(fr_dep.loc[fr_dep["__ym"] == ym, c]) for ym in EXPECTED_MONTHS}
        rates["POOLED"] = _pop_pct(fr_dep[c])
        rows.append({"column": c, **rates})
    anchor_tab = pd.DataFrame(rows).set_index("column")
    print(anchor_tab.round(4).to_string())

    # verdict rule: pooled within 2 points of ranking AND every month >= 90%
    aobt_pool = _pop_pct(fr_dep["AOBT_3_flt"])
    aobt_rank = _pop_pct(rank_dep["AOBT_3_flt"])
    aobt_month_min = min(_pop_pct(fr_dep.loc[fr_dep["__ym"] == ym, "AOBT_3_flt"])
                         for ym in EXPECTED_MONTHS)
    anchor_ok = (abs(aobt_pool - aobt_rank) <= 2.0) and (aobt_month_min >= 90.0)

    # ---------------------------------------------------------------- (5)
    _hdr("(5) HARD-ROW STRUCTURE — null operator / fully-null _flt block (final DEP)")
    op_null = fr_dep["AIRCRAFT_OPERATOR_flt"].isna()
    print(f"  pooled operator-null on final DEP = {op_null.mean() * 100:.4f}%  "
          f"({int(op_null.sum()):,}/{len(fr_dep):,})")
    print(f"  2025 OOF reference                = {REF_OPERATOR_NULL_PCT:.2f}%")
    print("\n  operator-null% per month:")
    for ym in EXPECTED_MONTHS:
        m = fr_dep["__ym"] == ym
        print(f"    {ym}: {fr_dep.loc[m, 'AIRCRAFT_OPERATOR_flt'].isna().mean() * 100:8.4f}%")
    block_null = fr_dep[FLT_COLS].isna().all(axis=1)
    print(f"\n  fully-null _flt block (all {len(FLT_COLS)} cols null) = "
          f"{block_null.mean() * 100:.4f}% ({int(block_null.sum()):,})")
    print("\n  operator-null% per airport (final DEP):")
    ap = fr_dep.groupby("ADEP_mvt")["AIRCRAFT_OPERATOR_flt"].apply(
        lambda s: s.isna().mean() * 100).sort_values(ascending=False)
    for code, val in ap.items():
        print(f"    {code:6s} {val:8.4f}%")
    hard_ok = op_null.mean() * 100 >= 1.0

    # ---------------------------------------------------------------- (6)
    _hdr("(6) BLANK DISCIPLINE — block/taxi/mvt time on final DEP and ARR")
    fr_arr = fr[fr["PHASE_mvt"] == "ARR"]
    for c in ["BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "MVT_TIME_UTC_mvt"]:
        print(f"  {c:22s} DEP null%={_null_pct(fr_dep[c]):8.4f}  "
              f"ARR null%={_null_pct(fr_arr[c]):8.4f}")
    blank_ok = (_null_pct(fr_dep["BLOCK_TIME_UTC_mvt"]) == 100.0
                and _null_pct(fr_dep["TAXITIME_SEC_mvt"]) == 100.0
                and _pop_pct(fr_arr["BLOCK_TIME_UTC_mvt"]) == 100.0
                and _pop_pct(fr_arr["TAXITIME_SEC_mvt"]) == 100.0
                and _null_pct(fr_dep["MVT_TIME_UTC_mvt"]) == 0.0)

    # ---------------------------------------------------------------- (7)
    _hdr("(7) DITHER QUANTIFICATION — final vs ranking MVT_TIME on Jan/Jul DEP")
    rank_jj = rank[(rank["PHASE_mvt"] == "DEP")
                   & (rank["__ym"].isin(["2026-01", "2026-07"]))][["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]]
    final_jj = fr_dep[fr_dep["__ym"].isin(["2026-01", "2026-07"])][["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]]
    print(f"  ranking Jan/Jul DEP ids = {len(rank_jj):,}")
    print(f"  final   Jan/Jul DEP ids = {len(final_jj):,}")
    print(f"  id set equality         = {set(rank_jj['MVT_ID_mvt']) == set(final_jj['MVT_ID_mvt'])}")
    merged = final_jj.merge(rank_jj, on="MVT_ID_mvt", suffixes=("_final", "_rank"))
    print(f"  joined rows             = {len(merged):,}")
    t_final = pd.to_datetime(merged["MVT_TIME_UTC_mvt_final"], utc=True)
    t_rank = pd.to_datetime(merged["MVT_TIME_UTC_mvt_rank"], utc=True)
    delta = (t_final - t_rank).dt.total_seconds()
    print(f"  delta min/max           = {delta.min():.1f}s / {delta.max():.1f}s")
    print(f"  |delta| <= 12s          = {(delta.abs() <= 12).mean() * 100:.2f}%  (expect ~92%)")
    print(f"  delta == 0              = {(delta == 0).mean() * 100:.2f}%")
    print(f"  0 < |delta| <= 12s      = {((delta != 0) & (delta.abs() <= 12)).mean() * 100:.2f}%")
    print(f"  |delta| > 12s           = {(delta.abs() > 12).mean() * 100:.4f}%  (expect ~0)")
    print("  delta distribution (seconds):")
    bins = [-np.inf, -6.5, -0.5, 0.5, 6.5, np.inf]
    labels = ["<=-7", "-6..-1", "0", "1..6", ">=7"]
    print(pd.cut(delta, bins=bins, labels=labels).value_counts().reindex(labels).to_string())
    dither_ok = (delta.abs() <= 12).mean() * 100 >= 90.0

    # ---------------------------------------------------------------- (8)
    _hdr("(8) EXTERNAL FEATURE COVERAGE — METAR (60 min back) + holiday (final DEP)")
    metar = pd.read_parquet(METAR)
    metar["valid"] = pd.to_datetime(metar["valid"], utc=True)
    metar["station"] = metar["station"].astype("string")
    metar_sorted = metar.sort_values(["valid"]).reset_index(drop=True)

    mvt = pd.DataFrame({
        "__idx": np.arange(len(fr_dep)),
        "__time": pd.to_datetime(fr_dep["MVT_TIME_UTC_mvt"], utc=True),
        "__airport": fr_dep["ADEP_mvt"].astype("string").values,
        "__ym": fr_dep["__ym"].values,
    })
    mvt_sorted = mvt.sort_values("__time", kind="mergesort").reset_index(drop=True)
    merged_metar = pd.merge_asof(
        mvt_sorted, metar_sorted,
        left_on="__time", right_on="valid",
        left_by="__airport", right_by="station",
        direction="backward", tolerance=pd.Timedelta("60min"),
    ).sort_values("__idx").reset_index(drop=True)
    matched = merged_metar["valid"].notna()
    print(f"  METAR observations loaded = {len(metar):,}")
    print(f"  METAR join rate pooled    = {matched.mean() * 100:.4f}%  "
          f"({int(matched.sum()):,}/{len(fr_dep):,})")
    print("  METAR join rate per month:")
    mr = []
    for ym in EXPECTED_MONTHS:
        m = merged_metar["__ym"] == ym
        mr.append((ym, float(matched[m].mean() * 100)))
        print(f"    {ym}: {matched[m].mean() * 100:8.4f}%")
    metar_rate_min = min(r for _, r in mr)

    hol = pd.read_parquet(HOLIDAY)
    hol["date"] = pd.to_datetime(hol["date"]).dt.date
    hol["airport"] = hol["airport"].astype("string")
    hol_lookup = hol.set_index(["airport", "date"])[["is_public_holiday"]]
    mvt_date = pd.to_datetime(fr_dep["MVT_TIME_UTC_mvt"], utc=True).dt.date
    keys = list(zip(fr_dep["ADEP_mvt"].astype("string"), mvt_date))
    idx = pd.MultiIndex.from_tuples(keys, names=["airport", "date"])
    hol_matched = hol_lookup.reindex(idx)["is_public_holiday"]
    print(f"\n  holiday rows loaded       = {len(hol):,}; "
          f"airports={hol['airport'].nunique()}")
    print(f"  holiday join rate pooled  = {hol_matched.notna().mean() * 100:.4f}%")
    print("  holiday join rate per month:")
    for ym in EXPECTED_MONTHS:
        m = (fr_dep["__ym"] == ym).values
        print(f"    {ym}: {hol_matched[m].notna().mean() * 100:8.4f}%")
    holiday_rate = float(hol_matched.notna().mean() * 100)
    ext_ok = (metar_rate_min >= 99.0) and (holiday_rate >= 99.0)

    # ---------------------------------------------------------------- (9)
    _hdr("(9) GO / NO-GO VERDICTS")
    print("  (a) ANCHOR SURVIVES [CRITICAL]:")
    print(f"      final pooled AOBT_3_flt populated = {aobt_pool:.4f}%  "
          f"vs ranking = {aobt_rank:.4f}%  (delta {aobt_pool - aobt_rank:+.4f} pts)")
    print(f"      worst final month populated       = {aobt_month_min:.4f}%")
    print(f"      VERDICT: {'GO' if anchor_ok else 'NO-GO — STOP AND ESCALATE'}")
    print("  (b) HARD-ROW STRUCTURE PRESENT (Task 19 hypothesis):")
    print(f"      final DEP operator-null = {op_null.mean() * 100:.4f}%  "
          f"(reference {REF_OPERATOR_NULL_PCT:.2f}%)")
    print(f"      VERDICT: {'GO' if hard_ok else 'NO-GO'}")
    print("  (c) EXTERNAL FEATURES JOIN >= 99% OF FINAL DEP ROWS:")
    print(f"      worst-month METAR = {metar_rate_min:.4f}%; holiday = {holiday_rate:.4f}%")
    print(f"      VERDICT: {'GO' if ext_ok else 'NO-GO'}")
    print(f"\n  blank-discipline check: {'PASS' if blank_ok else 'FAIL'}")
    print(f"  dither (<=12s >= 90%): {'PASS' if dither_ok else 'FAIL'}")

    all_ok = anchor_ok and hard_ok and ext_ok and blank_ok and dither_ok
    print("\n  OVERALL: " + ("ALL CHECKS GO" if all_ok else "AT LEAST ONE CHECK FAILED"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
