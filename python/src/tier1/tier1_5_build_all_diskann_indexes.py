#!/usr/bin/env python
"""
tier1_5_build_all_diskann_indexes.py

Build DiskANN indices for every (bucket, scale) partition present in the
Tier 1 Parquet observation store.

The Parquet observation layer remains the canonical embedding store.
DiskANN indices are disposable derived artefacts.

Bucketing strategy
-------------------
Buckets are FIXED 50-year calendar windows, not per-year and not
density-equalised. This matches the retrieval algorithm's fixed step size:
searches walk backward taking the top-k from one 50-year window and then
searching the previous 50-year window. Bucket width has to equal query-step
width or a single logical search step either fans out across multiple
physical indices (bucket too narrow) or silently pulls in results outside
the intended window (bucket too wide).

Buckets never span a model-boundary year (--model-boundary-year), since
embeddings on either side of that boundary are produced by different models
and are not comparable in a single L2/cosine index. This is currently a
no-op (model_boundary_year defaults to None) because the corpus is still
single-model; the hook exists so that when the historical/modern embedding
alignment work lands, indices built afterward are correct by construction
rather than by convention.

Per-bucket build parameters (pq_disk_bytes, memory budgets, complexity,
graph_degree) are density-sized from each bucket's observation count. The
corpus is not uniformly dense across centuries, so a flat set of build
parameters either wastes resources on sparse buckets or fails to fit dense
ones in the available RAM. Sizing is done ONCE per bucket and applied
identically to all three scales (local/medium/broad) within that bucket:
retrieval sometimes queries a single scale alone (not always as an
ensemble via RRF), so no scale can be built to a lower quality bar than
its siblings on the assumption something else will compensate for it.

This script is intentionally a thin orchestrator. All index-building logic
remains in tier1_5_build_diskann_index.build_one().
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pyarrow.dataset as ds

import lib.corpus_config as config
from lib.corpus_logging import logger
from tier1_5_build_diskann_index import build_one, year_bucket

SCALES = ("local", "medium", "broad")
DEFAULT_DIMENSIONS = 768
DEFAULT_YEAR_BUCKET_SIZE = 50

# RAM sizing defaults. These are starting heuristics, not measured
# constants -- tune against actual build/query behaviour once a few
# buckets have run.
DEFAULT_AVAILABLE_RAM_GB = 64.0
DEFAULT_RAM_HEADROOM_GB = 8.0


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


def compute_buckets(
    years: list[int],
    bucket_size: int,
    model_boundary_year: int | None,
) -> list[tuple[int, int]]:
    """
    Compute fixed calendar buckets covering the discovered years, splitting
    any bucket that would otherwise straddle model_boundary_year.
    """

    buckets: set[tuple[int, int]] = set()

    for year in years:
        buckets.add(year_bucket(year, bucket_size))

    ordered = sorted(buckets)

    if model_boundary_year is None:
        return ordered

    split: list[tuple[int, int]] = []

    for start, end in ordered:
        if start < model_boundary_year <= end:
            logger.warning(
                "Bucket year=%d-%d spans model boundary %d; "
                "splitting into year=%d-%d and year=%d-%d",
                start,
                end,
                model_boundary_year,
                start,
                model_boundary_year - 1,
                model_boundary_year,
                end,
            )
            split.append((start, model_boundary_year - 1))
            split.append((model_boundary_year, end))
        else:
            split.append((start, end))

    return split


def count_bucket_observations(
    store: Path,
    bucket_start: int,
    bucket_end: int,
) -> int:
    """
    Cheap row count for a bucket's year range, used only for sizing build
    parameters. Deliberately does not touch embedding columns or per-scale
    null filtering -- an approximate count is sufficient here; the exact
    per-scale count is recomputed by load_embeddings() at build time.
    """

    dataset = ds.dataset(
        store,
        format="parquet",
        partitioning="hive",
    )

    return dataset.scanner(
        columns=["event_id"],
        filter=(
            (ds.field("year") >= bucket_start)
            & (ds.field("year") <= bucket_end)
        ),
    ).count_rows()


def size_build_parameters(
    *,
    observation_count: int,
    dimensions: int,
    available_ram_gb: float,
    ram_headroom_gb: float,
) -> dict:
    """
    Pick pq_disk_bytes / memory budgets / complexity / graph_degree from a
    bucket's observation count.

    Heuristic, not measured: raw vector footprint is compared against a
    usable RAM budget. Buckets whose raw float32 vectors fit comfortably
    skip PQ compression entirely; buckets that don't fit get PQ
    compression sized to the degree of overflow, with complexity nudged up
    to offset the recall cost of compression. Treat these tiers as a
    starting point to validate against actual build times and recall, not
    as fixed truth.
    """

    raw_gb = (observation_count * dimensions * 4) / (1024 ** 3)
    usable_gb = max(available_ram_gb - ram_headroom_gb, 1.0)

    build_memory_gb = round(min(usable_gb, max(raw_gb * 1.5, 2.0)), 2)
    search_memory_gb = round(min(usable_gb * 0.5, max(raw_gb * 0.5, 1.0)), 2)

    if raw_gb <= usable_gb * 0.5:
        # Comfortably fits; no compression needed.
        pq_disk_bytes = 0
        complexity = 100
        graph_degree = 64
    elif raw_gb <= usable_gb:
        # Fits, but tightly -- light compression as a safety margin.
        pq_disk_bytes = 32
        complexity = 100
        graph_degree = 64
    else:
        # Does not fit raw; compression is required, scaled to overflow.
        overflow_ratio = raw_gb / usable_gb
        pq_disk_bytes = 64 if overflow_ratio <= 4 else 96
        complexity = 128
        graph_degree = 64

    return {
        "pq_disk_bytes": pq_disk_bytes,
        "search_memory_gb": search_memory_gb,
        "build_memory_gb": build_memory_gb,
        "complexity": complexity,
        "graph_degree": graph_degree,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build DiskANN indices for every fixed 50-year bucket and "
            "embedding scale."
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
        "--bucket-size",
        type=int,
        default=DEFAULT_YEAR_BUCKET_SIZE,
        help=(
            "Fixed calendar bucket width in years. Must match the "
            "retrieval algorithm's search-window step size."
        ),
    )

    parser.add_argument(
        "--model-boundary-year",
        type=int,
        default=None,
        help=(
            "Year at which the embedding model changes (e.g. MacBERTh -> "
            "modern BERT). Buckets spanning this year are split so no "
            "single index ever mixes embeddings from two models. "
            "Defaults to None (no split) until a validated cross-model "
            "alignment exists."
        ),
    )

    # These four default to None: when left unset, parameters are
    # density-sized per bucket via size_build_parameters(). Passing any of
    # them explicitly switches that parameter to a fixed value applied
    # uniformly across every bucket and scale (manual override mode).
    parser.add_argument("--complexity", type=int, default=None)
    parser.add_argument("--graph-degree", type=int, default=None)
    parser.add_argument("--search-memory-gb", type=float, default=None)
    parser.add_argument("--build-memory-gb", type=float, default=None)
    parser.add_argument("--pq-disk-bytes", type=int, default=None)

    parser.add_argument(
        "--available-ram-gb",
        type=float,
        default=DEFAULT_AVAILABLE_RAM_GB,
        help="RAM budget used for auto-sizing build parameters.",
    )

    parser.add_argument(
        "--ram-headroom-gb",
        type=float,
        default=DEFAULT_RAM_HEADROOM_GB,
        help="RAM reserved (not used for indexing) when auto-sizing.",
    )

    parser.add_argument(
        "--num-threads",
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

    buckets = compute_buckets(
        years,
        args.bucket_size,
        args.model_boundary_year,
    )
    logger.info(f"Computed {len(buckets)} fixed calendar buckets.")
    logger.info("-" * 70)

    total = len(buckets) * len(SCALES)
    completed = 0

    for bucket_start, bucket_end in buckets:
        observation_count = count_bucket_observations(
            args.store,
            bucket_start,
            bucket_end,
        )

        auto_params = size_build_parameters(
            observation_count=observation_count,
            dimensions=args.dimensions,
            available_ram_gb=args.available_ram_gb,
            ram_headroom_gb=args.ram_headroom_gb,
        )

        params = {
            "complexity": (
                args.complexity
                if args.complexity is not None
                else auto_params["complexity"]
            ),
            "graph_degree": (
                args.graph_degree
                if args.graph_degree is not None
                else auto_params["graph_degree"]
            ),
            "search_memory_gb": (
                args.search_memory_gb
                if args.search_memory_gb is not None
                else auto_params["search_memory_gb"]
            ),
            "build_memory_gb": (
                args.build_memory_gb
                if args.build_memory_gb is not None
                else auto_params["build_memory_gb"]
            ),
            "pq_disk_bytes": (
                args.pq_disk_bytes
                if args.pq_disk_bytes is not None
                else auto_params["pq_disk_bytes"]
            ),
        }

        logger.info(
            "Bucket year=%d-%d: observations=%d params=%s",
            bucket_start,
            bucket_end,
            observation_count,
            params,
        )

        for scale in SCALES:
            output_directory = (
                args.output / f"year={bucket_start}-{bucket_end}" / scale
            )

            if output_directory.exists():
                complete = output_directory / "_COMPLETE"

                if complete.exists() and not args.overwrite:
                    logger.info(
                        f"[SKIP] year={bucket_start}-{bucket_end} scale={scale}"
                    )
                    completed += 1
                    continue

                if args.overwrite:
                    logger.info(
                        f"[OVERWRITE] removing existing index: {output_directory}"
                    )
                else:
                    logger.warning(
                        f"[REBUILD] removing incomplete index: {output_directory}"
                    )

                shutil.rmtree(output_directory)

            logger.info("=" * 70)
            logger.info(
                f"[{completed + 1}/{total}] "
                f"year={bucket_start}-{bucket_end} scale={scale}"
            )

            build_one(
                store=args.store,
                output_root=args.output,
                bucket_start=bucket_start,
                bucket_end=bucket_end,
                scale=scale,
                dimensions=args.dimensions,
                complexity=params["complexity"],
                graph_degree=params["graph_degree"],
                search_memory_gb=params["search_memory_gb"],
                build_memory_gb=params["build_memory_gb"],
                num_threads=args.num_threads,
                pq_disk_bytes=params["pq_disk_bytes"],
            )

            completed += 1

    logger.info("=" * 70)
    logger.info("Tier 1.5 build complete.")
    logger.info(f"Built or verified {completed} index partitions.")


if __name__ == "__main__":
    main()
