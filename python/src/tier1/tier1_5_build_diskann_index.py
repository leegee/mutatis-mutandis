#!/usr/bin/env python
"""
tier1_5_build_diskann_index.py

Build one DiskANN index from a Parquet Tier 1 observation partition.

The Parquet observation layer is the source of truth. DiskANN is a disposable
derived geometric index and stores no semantic provenance.

This first pass deliberately builds one (year, scale) index at a time. That
keeps peak memory bounded and gives us a simple integration point to test
before adding corpus-wide orchestration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from retrieval.parquet_embeddings import load_embeddings
from retrieval.diskann_builder import build_diskann_index
import lib.corpus_config as config
from lib.corpus_logging import logger

SCALES = ("local", "medium", "broad")
DEFAULT_DIMENSIONS = 768


def build_one(
    *,
    store: Path,
    output_root: Path,
    year: int,
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
        f"Loading Parquet observations: "
        f"year={year} scale={scale}"
    )

    event_ids, vectors = load_embeddings(
        store,
        year=year,
        scale=scale,
        dimensions=dimensions,
    )

    logger.info(
        f"Loaded {len(event_ids)} observations "
        f"with vectors of shape {vectors.shape}"
    )

    output_directory = (
        output_root
        / f"year={year}"
        / scale
    )

    logger.info(
        f"Building DiskANN index: {output_directory}"
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
        f"DiskANN build complete: {output_directory}"
    )
    logger.info(
        f"Event-ID mapping: {event_ids_path}"
    )

    return output_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one DiskANN index from a Parquet Tier 1 "
            "observation partition."
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
        help="Publication year to index.",
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
        help=(
            f"Embedding dimensionality "
            f"(default: {DEFAULT_DIMENSIONS})."
        ),
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
        help=(
            "DiskANN search-memory budget in GB "
            "(default: 4)."
        ),
    )

    parser.add_argument(
        "--build-memory-gb",
        type=float,
        default=8.0,
        help=(
            "DiskANN build-memory budget in GB "
            "(default: 8)."
        ),
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help=(
            "DiskANN build threads; "
            "0 means all available threads."
        ),
    )

    parser.add_argument(
        "--pq-disk-bytes",
        type=int,
        default=0,
        help=(
            "PQ bytes per vector; "
            "0 disables PQ compression."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    build_one(
        store=args.store,
        output_root=args.output,
        year=args.year,
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
