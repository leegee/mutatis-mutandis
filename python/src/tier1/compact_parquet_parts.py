#!/usr/bin/env python
"""
compact_parquet_parts.py — Merge small hive part files per year

After a Tier 1 run (or an older unbuffered write), each year=* directory
may contain many tiny part-*.parquet files. This tool concatenates them
into fewer larger parts to reduce NTFS/AV pressure and speed up DuckDB globs.

Safe algorithm (per year directory)
-----------------------------------
1. List part-*.parquet (ignore .tmp and hidden files)
2. Optionally skip files already larger than --min-file-bytes
3. Read selected parts with PyArrow, concatenate
4. Write one new part-*.parquet via temp rename
5. Delete the source parts that were merged

Usage
-----
    python compact_parquet_parts.py --store /data/tier1_parquet
    python compact_parquet_parts.py --store /data/tier1_parquet --year 1664
    python compact_parquet_parts.py --store /data/tier1_parquet --dry-run
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

from tier1.observation_store_api import (
    configure_store_backend,
    resolve_store_path,
)


from lib.corpus_logging import logger

def list_year_dirs(root: Path, year: Optional[int] = None) -> list[Path]:
    if year is not None:
        d = root / f"year={year}"
        return [d] if d.is_dir() else []
    return sorted(
        p for p in root.glob("year=*") if p.is_dir() and p.name.startswith("year=")
    )


def list_parts(year_dir: Path) -> list[Path]:
    parts = []
    for p in sorted(year_dir.glob("part-*.parquet")):
        if p.name.startswith("."):
            continue
        parts.append(p)
    return parts


def compact_year_dir(
    year_dir: Path,
    *,
    min_file_bytes: int = 64 * 1024 * 1024,
    target_rows: int = 250_000,
    dry_run: bool = False,
) -> dict:
    """
    Merge parts smaller than min_file_bytes into larger files.

    Large parts are left untouched. Small parts are grouped until
    cumulative rows approach target_rows, then written as one part.
    """
    parts = list_parts(year_dir)
    if len(parts) <= 1:
        return {"year_dir": str(year_dir), "parts_in": len(parts), "merged": 0, "written": 0}

    small: list[tuple[Path, int]] = []
    kept_large = 0
    for p in parts:
        sz = p.stat().st_size
        if sz < min_file_bytes:
            # row count from metadata when cheap
            try:
                meta = pq.ParquetFile(p).metadata
                nrows = int(meta.num_rows) if meta is not None else 0
            except Exception:
                nrows = 0
            small.append((p, nrows))
        else:
            kept_large += 1

    if not small:
        return {
            "year_dir": str(year_dir),
            "parts_in": len(parts),
            "merged": 0,
            "written": 0,
            "kept_large": kept_large,
        }

    groups: list[list[tuple[Path, int]]] = []
    cur: list[tuple[Path, int]] = []
    cur_rows = 0
    for item in small:
        cur.append(item)
        cur_rows += item[1]
        if cur_rows >= target_rows and len(cur) >= 1:
            groups.append(cur)
            cur, cur_rows = [], 0
    if cur:
        groups.append(cur)

    written = 0
    merged = 0
    for group in groups:
        if len(group) < 2:
            # A lone small part has nothing to compact with. Leaving it alone makes repeated compaction runs convergent.
            continue

        paths = [g[0] for g in group]
        merged += len(paths)
        if dry_run:
            written += 1
            continue

        # Too much in memory? Go via ParquetWriter?
        tables = [pq.read_table(p) for p in paths]
        table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]

        part_name = f"part-{uuid.uuid4().hex[:12]}.parquet"
        dest = year_dir / part_name
        tmp = year_dir / f".{part_name}.tmp"
        try:
            from parquet_observation_backend import write_observation_parquet
        except ImportError:
            from lib.parquet_observation_backend import write_observation_parquet  # type: ignore
        write_observation_parquet(table, tmp)
        tmp.rename(dest)
        written += 1
        for p in paths:
            try:
                p.unlink()
            except OSError:
                pass

    return {
        "year_dir": str(year_dir),
        "parts_in": len(parts),
        "small": len(small),
        "merged": merged,
        "written": written,
        "kept_large": kept_large,
        "dry_run": dry_run,
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Compact small Parquet part files under a hive-partitioned observation store"
    )

    p.add_argument("--store", type=str, default=None)
    p.add_argument("--masked", action="store_true")
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=1)

    p.add_argument("--year", type=int, default=None, help="Only compact this year")
    p.add_argument(
        "--min-file-bytes",
        type=int,
        default=64 * 1024 * 1024,
        help="Parts smaller than this are candidates for merge (default 64MiB)",
    )
    p.add_argument(
        "--target-rows",
        type=int,
        default=250_000,
        help="Aim for about this many rows per merged part (default 500000)",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    configure_store_backend(
        "parquet",
        num_shards=args.num_shards,
    )

    root = resolve_store_path(
        store_backend="parquet",
        masked=args.masked,
        store=args.store,
        shard=args.shard,
        num_shards=args.num_shards,
    )

    if not root.is_dir():
        raise SystemExit(f"store not found: {root}")

    years = list_year_dirs(root, args.year)
    if not years:
        logger.warning("no year=* directories found")
        return 0

    total_merged = 0
    total_written = 0
    for yd in years:
        stats = compact_year_dir(
            yd,
            min_file_bytes=args.min_file_bytes,
            target_rows=args.target_rows,
            dry_run=args.dry_run,
        )
        total_merged += int(stats.get("merged", 0))
        total_written += int(stats.get("written", 0))
        logger.info(
            f"{stats['year_dir']}: parts_in={stats.get('parts_in')} "
            f"small={stats.get('small', 0)} merged={stats.get('merged', 0)} "
            f"written={stats.get('written', 0)}"
            + (" [dry-run]" if args.dry_run else "")
        )

    logger.info(f"done: merged_files={total_merged} new_parts={total_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
