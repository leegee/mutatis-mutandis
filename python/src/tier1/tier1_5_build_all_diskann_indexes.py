#!/usr/bin/env python
"""
tier1_5_build_all_diskann_indexes.py

Build DiskANN indices for every (year, scale) partition present in the
Tier 1 Parquet observation store.

The Parquet observation layer remains the canonical embedding store.
DiskANN indices are disposable derived artefacts.

This script is intentionally a thin orchestrator. All index-building logic
remains in tier1_5_build_diskann_index.build_one().
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.dataset as ds

import lib.corpus_config as config
from lib.corpus_logging import logger
from tier1_5_build_diskann_index import build_one

SCALES = ("local", "medium", "broad")
DEFAULT_DIMENSIONS = 768


def discover_years(store: Path) -> list[int]:
    """
    Return every publication year present in the Parquet observation store.
    """

    dataset = ds.dataset(
        store,
        format="parquet",
        partitioning="hive",
    )

    years = (
        dataset
        .scanner(columns=["year"])
        .to_table()
        .column("year")
        .to_pylist()
    )

    return sorted(set(int(year) for year in years))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build DiskANN indices for every year and embedding scale."
        )
    )

    parser.add_argument(
        "--store",
        type=Path,
        default=config.EVENTSTORE_T1_PATH,
        help="Root of the Tier 1 Parquet observation store.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=config.DISKANN_INDEXES_DIR,
        help="Root directory for DiskANN indices.",
    )

    parser.add_argument(
        "--dimensions",
        type=int,
        default=DEFAULT_DIMENSIONS,
    )

    parser.add_argument(
        "--complexity",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--graph-degree",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--search-memory-gb",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--build-memory-gb",
        type=float,
        default=8.0,
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--pq-disk-bytes",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild indices that already exist.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    years = discover_years(args.store)

    logger.info(f"Discovered {len(years)} publication years.")
    logger.info()

    total = len(years) * len(SCALES)
    completed = 0

    for year in years:
        for scale in SCALES:

            output_directory = (
                args.output
                / f"year={year}"
                / scale
            )

            if output_directory.exists() and not args.overwrite:
                logger.info(
                    f"[SKIP] year={year} scale={scale}"
                )
                completed += 1
                continue

            logger.info(
                "=" * 70
            )
            logger.info(
                f"[{completed + 1}/{total}] "
                f"year={year} scale={scale}"
            )

            build_one(
                store=args.store,
                output_root=args.output,
                year=year,
                scale=scale,
                dimensions=args.dimensions,
                complexity=args.complexity,
                graph_degree=args.graph_degree,
                search_memory_gb=args.search_memory_gb,
                build_memory_gb=args.build_memory_gb,
                num_threads=args.num_threads,
                pq_disk_bytes=args.pq_disk_bytes,
            )

            completed += 1

    logger.info()
    logger.info("=" * 70)
    logger.info("Tier 1.5 build complete.")
    logger.info(f"Built or verified {completed} index partitions.")


if __name__ == "__main__":
    main()
