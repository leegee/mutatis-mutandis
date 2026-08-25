#!/usr/bin/env python
"""
tier1_5_build_diskann_index.py

Build a DiskANN index from a 50-year Parquet Tier 1 observation partition.

The Parquet observation layer is the source of truth. DiskANN is a disposable
derived geometric index and stores no semantic provenance.

Physical indexes are partitioned by year bucket and embedding scale:

    year=1650-1699/local
    year=1650-1699/medium
    year=1650-1699/broad

The bucket size is deliberately configurable so retrieval experiments can
compare temporal partitioning strategies without changing Tier 1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from retrieval.parquet_embeddings import load_embeddings
from retrieval.diskann_builder import build_diskann_index
import lib.corpus_config as config
from lib.corpus_logging import logger


SCALES = ("local", "medium", "broad")
DEFAULT_DIMENSIONS = 768
DEFAULT_YEAR_BUCKET_SIZE = 50


def year_bucket(
    year: int,
    bucket_size: int,
) -> tuple[int, int]:
    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")

    start = (year // bucket_size) * bucket_size
    end = start + bucket_size - 1

    return start, end


def build_one(
    *,
    store: Path,
    output_root: Path,
    bucket_start: int,
    bucket_end: int,
    scale: str,
    dimensions: int,
    complexity: int,
    graph_degree: int,
    search_memory_gb: float,
    build_memory_gb: float,
    num_threads: int,
    pq_disk_bytes: int,
) -> Path:
    logger.info(
        "Loading Parquet observations: "
        "years=%d-%d scale=%s",
        bucket_start,
        bucket_end,
        scale,
    )

    event_ids, vectors = load_embeddings(
        store,
        year_start = bucket_start,
        year_end   = bucket_end,
        scale      = scale,
        dimensions = dimensions,
    )

    logger.info(
        "Loaded %d observations with vectors of shape %s",
        len(event_ids),
        vectors.shape,
    )

    if len(event_ids) == 0:
        logger.warning(
            "No observations for years=%d-%d scale=%s; "
            "skipping index",
            bucket_start,
            bucket_end,
            scale,
        )
        return None

    output_directory = (
        output_root
        / f"year={bucket_start}-{bucket_end}"
        / scale
    )

    logger.info(
        "Building DiskANN index: %s",
        output_directory,
    )

    event_ids_path = build_diskann_index(
        vectors=vectors,
        event_ids=event_ids,
        index_directory=output_directory,
        dimensions=dimensions,
        complexity=complexity,
        graph_degree=graph_degree,
        search_memory_gb=search_memory_gb,
        build_memory_gb=build_memory_gb,
        num_threads=num_threads,
        pq_disk_bytes=pq_disk_bytes,
        index_prefix=scale,
    )

    logger.info(
        "DiskANN build complete: %s",
        output_directory,
    )

    logger.info(
        "Event-ID mapping: %s",
        event_ids_path,
    )

    return output_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a temporal-bucketed DiskANN index from "
            "the Parquet observation store."
        )
    )

    parser.add_argument(
        "--store",
        type=Path,
        default=config.EVENTSTORE_T1_PATH,
        help="Root of the Parquet observation store.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=config.DISKANN_INDEXES_DIR,
        help="Root directory for derived DiskANN indices.",
    )

    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help=(
            "Year identifying the bucket to build. "
            "The bucket containing this year is indexed."
        ),
    )

    parser.add_argument(
        "--bucket-size",
        type=int,
        default=DEFAULT_YEAR_BUCKET_SIZE,
        help=(
            f"Temporal bucket size in years "
            f"(default: {DEFAULT_YEAR_BUCKET_SIZE})."
        ),
    )

    parser.add_argument(
        "--scale",
        choices=SCALES,
        required=True,
        help="Embedding scale to index.",
    )

    parser.add_argument(
        "--dimensions",
        type=int,
        default=DEFAULT_DIMENSIONS,
        help=f"Embedding dimensionality (default: {DEFAULT_DIMENSIONS}).",
    )

    parser.add_argument(
        "--complexity",
        type=int,
        default=100,
        help="DiskANN build complexity (default: 100).",
    )

    parser.add_argument(
        "--graph-degree",
        type=int,
        default=64,
        help="DiskANN graph degree (default: 64).",
    )

    parser.add_argument(
        "--search-memory-gb",
        type=float,
        default=4.0,
        help="DiskANN search-memory budget in GB (default: 4).",
    )

    parser.add_argument(
        "--build-memory-gb",
        type=float,
        default=8.0,
        help="DiskANN build-memory budget in GB (default: 8).",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help="DiskANN build threads; 0 means all available threads.",
    )

    parser.add_argument(
        "--pq-disk-bytes",
        type=int,
        default=0,
        help="PQ bytes per vector; 0 disables PQ compression.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bucket_start, bucket_end = year_bucket(
        args.year,
        args.bucket_size,
    )

    logger.info(
        "Building temporal bucket %d-%d",
        bucket_start,
        bucket_end,
    )

    build_one(
        store=args.store,
        output_root=args.output,
        bucket_start=bucket_start,
        bucket_end=bucket_end,
        scale=args.scale,
        dimensions=args.dimensions,
        complexity=args.complexity,
        graph_degree=args.graph_degree,
        search_memory_gb=args.search_memory_gb,
        build_memory_gb=args.build_memory_gb,
        num_threads=args.num_threads,
        pq_disk_bytes=args.pq_disk_bytes,
    )


if __name__ == "__main__":
    main()
