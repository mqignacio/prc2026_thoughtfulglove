"""Data acquisition verification — PRC-OpenSky Data Challenge 2026 (Task 01).

Runs the mechanical checks (a)-(g) from tasks/01-data_acquisition.md against the
parquet files in data/raw/. Prints a report to stdout; re-run after download and
paste the output into src/reports/data_verification.md.

Usage:
    uv run python src/verify_data.py
"""

from __future__ import annotations

import glob
import os
import sys

import pandas as pd
import pyarrow.parquet as pq

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
EXPECTED_TRAINING_ROWS = 4_167_797
EXPECTED_ADEPS = {"EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LTFM", "LSZH"}
TIME_COLS = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]


def _size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def _open(path: str) -> pd.DataFrame:
    # pyarrow engine validates readability; row group metadata gives cheap count.
    return pq.read_table(path).to_pandas()


def main() -> int:
    if not os.path.isdir(RAW):
        print(f"ERROR: {RAW} not found — download not done yet.", file=sys.stderr)
        return 1

    training = sorted(glob.glob(os.path.join(RAW, "training_2025-*.parquet")))
    ranking_p = os.path.join(RAW, "ranking.parquet")
    submitting_p = os.path.join(RAW, "submitting.parquet")

    print("=" * 72)
    print("(0) FILE INVENTORY")
    print("=" * 72)
    print(f"training files: {len(training)} (expected 12)")
    for name, ok in [("ranking.parquet", os.path.exists(ranking_p)),
                     ("submitting.parquet", os.path.exists(submitting_p))]:
        print(f"  {name}: {'present' if ok else 'MISSING'}")

    # ---- load + per-file sizes/rows (check a, b) ----
    total_rows = 0
    print("\n" + "=" * 72)
    print("(a) PER-FILE SIZE & ROWS")
    print("=" * 72)
    frames = {}
    for p in training:
        df = _open(p)
        frames[os.path.basename(p)] = df
        total_rows += len(df)
        print(f"  {os.path.basename(p):40s} {_size_mb(p):6.1f} MB  {len(df):>9,} rows")
    train_all = pd.concat(frames.values(), ignore_index=True) if frames else pd.DataFrame()

    rank_df = _open(ranking_p) if os.path.exists(ranking_p) else pd.DataFrame()
    sub_df = _open(submitting_p) if os.path.exists(submitting_p) else pd.DataFrame()
    if os.path.exists(ranking_p):
        print(f"  {'ranking.parquet':40s} {_size_mb(ranking_p):6.1f} MB  {len(rank_df):>9,} rows")
    if os.path.exists(submitting_p):
        print(f"  {'submitting.parquet':40s} {_size_mb(submitting_p):6.1f} MB  {len(sub_df):>9,} rows")

    print("\n" + "=" * 72)
    print("(b) TOTAL TRAINING ROWS")
    print("=" * 72)
    print(f"  total = {total_rows:,}   expected ≈ {EXPECTED_TRAINING_ROWS:,}   "
          f"match={abs(total_rows - EXPECTED_TRAINING_ROWS) < 5000}")

    # ---- (c) reporting airports in training (DEP ADEP_mvt / ARR ADES_mvt) ----
    # The "10 airports" are the REPORTING airport: ADEP_mvt on DEP rows,
    # ADES_mvt on ARR rows. Counting ADEP_mvt across all rows is wrong because
    # ARR rows carry the origin of arriving flights (worldwide).
    print("\n" + "=" * 72)
    print("(c) UNIQUE REPORTING AIRPORTS IN TRAINING (DEP ADEP_mvt)")
    print("=" * 72)
    if {"ADEP_mvt", "ADES_mvt", "PHASE_mvt"}.issubset(train_all.columns):
        dep_adeps = set(train_all.loc[train_all["PHASE_mvt"] == "DEP", "ADEP_mvt"].dropna().unique())
        arr_ades = set(train_all.loc[train_all["PHASE_mvt"] == "ARR", "ADES_mvt"].dropna().unique())
        print(f"  DEP ADEP_mvt count = {len(dep_adeps)}   expected = 10   match={len(dep_adeps) == 10}")
        print(f"  DEP ADEP_mvt set   = {sorted(dep_adeps)}")
        print(f"  ARR ADES_mvt set == DEP ADEP_mvt set: {arr_ades == dep_adeps}")
        print(f"  extra/unexpected vs official: {sorted(dep_adeps - EXPECTED_ADEPS)}")
        print(f"  missing vs official:          {sorted(EXPECTED_ADEPS - dep_adeps)}")
        # for reference only (NOT the check): raw ADEP_mvt across all rows
        print(f"  [ref] distinct ADEP_mvt over ALL rows (incl. ARR origins): {train_all['ADEP_mvt'].nunique()}")
    else:
        print("  ADEP_mvt / ADES_mvt / PHASE_mvt column(s) not found!")

    # ---- (d) ranking month coverage ----
    print("\n" + "=" * 72)
    print("(d) RANKING MONTH COVERAGE (expect Jan 2026 + Jul 2026)")
    print("=" * 72)
    if "MVT_TIME_UTC_mvt" in rank_df.columns and len(rank_df):
        mt = pd.to_datetime(rank_df["MVT_TIME_UTC_mvt"], errors="coerce", utc=True)
        covered = sorted({f"{y}-{m:02d}" for y, m in zip(mt.dt.year, mt.dt.month) if pd.notna(y)})
        print(f"  covered (year-month) from MVT_TIME_UTC_mvt: {covered}")
    else:
        print("  MVT_TIME_UTC_mvt missing or ranking empty (see check g).")

    # ---- (e) submitting MVT_IDs vs ranking DEP MVT_IDs ----
    print("\n" + "=" * 72)
    print("(e) SUBMITTING MVT_IDs vs RANKING DEP MVT_IDs")
    print("=" * 72)
    if {"MVT_ID_mvt"}.issubset(sub_df.columns) and "MVT_ID_mvt" in rank_df.columns and "PHASE_mvt" in rank_df.columns:
        sub_ids = set(sub_df["MVT_ID_mvt"])
        rank_dep_ids = set(rank_df.loc[rank_df["PHASE_mvt"] == "DEP", "MVT_ID_mvt"])
        print(f"  submitting ids      = {len(sub_ids):,}")
        print(f"  ranking DEP ids     = {len(rank_dep_ids):,}")
        print(f"  submitting ⊆ ranking DEP : {sub_ids.issubset(rank_dep_ids)}")
        print(f"  exact set equality       : {sub_ids == rank_dep_ids}")
        print(f"  in submitting not in rank DEP: {len(sub_ids - rank_dep_ids):,}")
        print(f"  in rank DEP not submitting    : {len(rank_dep_ids - sub_ids):,}")
    else:
        print("  required id/phase columns missing.")

    # ---- (f) ranking column availability for PHASE=DEP ----
    print("\n" + "=" * 72)
    print("(f) RANKING COLUMN NULL-RATE FOR PHASE=DEP (availability table)")
    print("=" * 72)
    if "PHASE_mvt" in rank_df.columns and len(rank_df):
        dep = rank_df[rank_df["PHASE_mvt"] == "DEP"]
        rows = []
        for col in dep.columns:
            null_rate = dep[col].isna().mean() * 100.0
            rows.append((col, len(dep), null_rate))
        avail = pd.DataFrame(rows, columns=["column", "dep_rows", "null_pct"]).sort_values("null_pct")
        pd.set_option("display.max_rows", None)
        print(avail.to_string(index=False))
        fully_null = [c for c, _, nr in rows if nr == 100.0]
        print(f"\n  FULLY-NULL (100%) DEP columns: {fully_null}")
        print(f"  BLOCK_TIME_UTC_mvt null%: {dep['BLOCK_TIME_UTC_mvt'].isna().mean()*100:.2f}"
              if "BLOCK_TIME_UTC_mvt" in dep else "  BLOCK_TIME_UTC_mvt: absent")
        print(f"  TAXITIME_SEC_mvt   null%: {dep['TAXITIME_SEC_mvt'].isna().mean()*100:.2f}"
              if "TAXITIME_SEC_mvt" in dep else "  TAXITIME_SEC_mvt: absent")

    # ---- (g) CRITICAL: MVT_TIME_UTC_mvt populated in ranking DEP? ----
    print("\n" + "=" * 72)
    print("(g) CRITICAL VERDICT — MVT_TIME_UTC_mvt POPULATED IN RANKING DEP ROWS")
    print("=" * 72)
    if "MVT_TIME_UTC_mvt" in rank_df.columns and "PHASE_mvt" in rank_df.columns:
        dep_mt = rank_df.loc[rank_df["PHASE_mvt"] == "DEP", "MVT_TIME_UTC_mvt"]
        populated = dep_mt.notna().mean() * 100.0
        verdict = "CONFIRMED (populated)" if populated > 99.0 else ("PARTIAL" if populated > 0 else "REFUTED (all null)")
        print(f"  DEP rows = {len(dep_mt):,}   non-null = {dep_mt.notna().sum():,}   "
              f"populated% = {populated:.2f}")
        print(f"  VERDICT: {verdict}")
        print("  -> TODO locked decision #5 stands only if CONFIRMED. If REFUTED/PARTIAL,")
        print("     congestion-feature strategy needs re-planning (fallback: SCHED_TIME_UTC_mvt + NM times).")
    else:
        print("  VERDICT: cannot determine — MVT_TIME_UTC_mvt and/or PHASE_mvt absent in ranking.")

    print("\n" + "=" * 72)
    print("SCHEMA (ranking): list of columns")
    print("=" * 72)
    print(list(rank_df.columns))
    return 0


if __name__ == "__main__":
    sys.exit(main())
