#!/usr/bin/env python3
"""MinIO (S3-compatible) transfer for PRC-2026 submissions.

Credentials are read from the MinIO client config at ``~/.mc/config.json``
(override with the ``MC_CONFIG`` env var). Nothing is passed on the command
line, so keys never land in shell history or process listings.

Requires the ``minio`` package; run with an ephemeral env:

    uv run --with minio python src/minio_transfer.py ls    opensky/prc-2026-thoughtful-glove/
    uv run --with minio python src/minio_transfer.py up    submissions/thoughtful-glove_v4.parquet opensky/prc-2026-thoughtful-glove/
    uv run --with minio python src/minio_transfer.py get   opensky/prc-2026-thoughtful-glove/thoughtful-glove_v4.parquet_result.json /tmp/v4_result.json

Arguments use ``<alias>/<bucket>/<object...>`` form (the alias resolves to the
endpoint in the mc config).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("minio_transfer")

MC_CONFIG = os.environ.get("MC_CONFIG", os.path.expanduser("~/.mc/config.json"))


def load_alias(alias: str | None) -> tuple[str, str, str, str]:
    """Return (alias_name, endpoint_host, access_key, secret_key) from the mc config."""
    if not os.path.exists(MC_CONFIG):
        raise SystemExit(f"no mc config at {MC_CONFIG}; pass MC_CONFIG or provision credentials")
    with open(MC_CONFIG) as f:
        cfg = json.load(f)
    aliases = cfg.get("aliases", {})
    if not aliases:
        raise SystemExit(f"no aliases in {MC_CONFIG}")
    name = alias if alias in aliases else ("opensky" if "opensky" in aliases else next(iter(aliases)))
    a = aliases[name]
    url = a["url"]
    host = url.split("://", 1)[-1].rstrip("/")
    return name, host, a["accessKey"], a["secretKey"]


def make_client(alias: str | None):
    from minio import Minio
    name, host, ak, sk = load_alias(alias)
    log.info("Using alias '%s' -> %s", name, host)
    return Minio(host, access_key=ak, secret_key=sk, secure=True)


def split_target(spec: str) -> tuple[str, str, str]:
    """``alias/bucket[/object]`` -> (alias, bucket, object)."""
    parts = spec.split("/", 2)
    if len(parts) < 2:
        raise SystemExit(f"bad target '{spec}'; expected <alias>/<bucket>[/<object>]")
    alias, bucket = parts[0], parts[1]
    obj = parts[2] if len(parts) > 2 else ""
    return alias, bucket, obj


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_ls = sub.add_parser("ls", help="list objects under <alias>/<bucket>/[prefix]")
    p_ls.add_argument("target")

    p_up = sub.add_parser("up", help="upload <local> to <alias>/<bucket>/[object]")
    p_up.add_argument("local")
    p_up.add_argument("target")

    p_get = sub.add_parser("get", help="download <alias>/<bucket>/<object> to <local>")
    p_get.add_argument("target")
    p_get.add_argument("local")

    args = p.parse_args()

    if args.cmd == "ls":
        alias, bucket, prefix = split_target(args.target)
        client = make_client(alias)
        for o in client.list_objects(bucket, prefix=prefix or None, recursive=False):
            print(f"{o.size:>12,d}  {o.last_modified}  {o.object_name}")
        return

    if args.cmd == "up":
        alias, bucket, obj = split_target(args.target)
        obj = obj or os.path.basename(args.local)
        client = make_client(alias)
        client.fput_object(bucket, obj, args.local)
        log.info("Uploaded %s -> %s/%s", args.local, bucket, obj)
        return

    if args.cmd == "get":
        alias, bucket, obj = split_target(args.target)
        if not obj:
            raise SystemExit("get requires an object name")
        client = make_client(alias)
        client.fget_object(bucket, obj, args.local)
        log.info("Downloaded %s/%s -> %s", bucket, obj, args.local)
        return


if __name__ == "__main__":
    main()
