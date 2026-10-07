"""Place the PRC-2026 competition parquet files into ``data/raw/``.

The competition datasets are distributed by the challenge organizers to
registered participants and are **not redistributed** in this repository.
This script only moves files you are already entitled to hold: it mirrors the
organizer dataset bucket (or a local download directory) into ``data/raw/``
and then prints the expected file list so you can verify the copy.

It does **not** contain or fetch credentials. If you use the MinIO mirror, you
must already have an ``mc`` alias configured for the challenge endpoint
(``mc alias set <name> <endpoint> <access-key> <secret-key>``); keys live in
``~/.mc/config.json`` and are never passed on the command line.

Usage:
    # From a local download directory (e.g. the challenge "download all" zip):
    uv run python src/fetch/fetch_competition_data.py --from-dir ~/Downloads/prc2026

    # From the organizer MinIO bucket via an existing mc alias:
    uv run python src/fetch/fetch_competition_data.py --mc-alias opensky

    # Then verify row counts and set equality:
    uv run python src/verify_data.py
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import shutil
import subprocess
import sys

log = logging.getLogger("fetch_competition_data")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEST = os.path.join(REPO_ROOT, "data", "raw")
BUCKET_PREFIX = "prc-2026-datasets"

EXPECTED_FILES = [
    "training_2025-01-01_2025-02-01.parquet",
    "training_2025-02-01_2025-03-01.parquet",
    "training_2025-03-01_2025-04-01.parquet",
    "training_2025-04-01_2025-05-01.parquet",
    "training_2025-05-01_2025-06-01.parquet",
    "training_2025-06-01_2025-07-01.parquet",
    "training_2025-07-01_2025-08-01.parquet",
    "training_2025-08-01_2025-09-01.parquet",
    "training_2025-09-01_2025-10-01.parquet",
    "training_2025-10-01_2025-11-01.parquet",
    "training_2025-11-01_2025-12-01.parquet",
    "training_2025-12-01_2026-01-01.parquet",
    "ranking.parquet",
    "submitting.parquet",
]


def _copy_from_dir(src: str) -> int:
    os.makedirs(DEST, exist_ok=True)
    copied = 0
    for name in EXPECTED_FILES:
        path = os.path.join(src, name)
        if not os.path.exists(path):
            log.warning("missing in source dir: %s", name)
            continue
        shutil.copy2(path, os.path.join(DEST, name))
        copied += 1
        log.info("copied %s", name)
    return copied


def _mirror_from_mc(alias: str) -> int:
    remote = f"{alias}/{BUCKET_PREFIX}/"
    os.makedirs(DEST, exist_ok=True)
    log.info("mc mirror --overwrite %s %s", remote, DEST)
    proc = subprocess.run(
        ["mc", "mirror", "--overwrite", remote, DEST],
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"`mc mirror` failed (exit {proc.returncode}). "
            f"Check the alias '{alias}' and that you have access to {BUCKET_PREFIX}."
        )
    return len(glob.glob(os.path.join(DEST, "*.parquet")))


def _report() -> int:
    present = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(DEST, "*.parquet")))]
    missing = [n for n in EXPECTED_FILES if n not in present]
    log.info("data/raw contains %d parquet files", len(present))
    if missing:
        log.error("missing %d expected file(s): %s", len(missing), ", ".join(missing))
        return 1
    log.info("all %d expected files present", len(EXPECTED_FILES))
    log.info("next: uv run python src/verify_data.py")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--from-dir", type=str, help="Local directory holding the organizer parquet files.")
    group.add_argument("--mc-alias", type=str, help="Configured mc alias for the challenge endpoint.")
    args = parser.parse_args()

    copied = _copy_from_dir(args.from_dir) if args.from_dir else _mirror_from_mc(args.mc_alias)
    log.info("placed %d file(s) into %s", copied, DEST)
    return _report()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    sys.exit(main())
