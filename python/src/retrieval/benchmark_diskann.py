#!/usr/bin/env python
"""
benchmark_diskann_layouts.py

Compare DiskANN retrieval layouts for the Tier 1 observation corpus.

Layouts
-------

1. Per-year:
       year=1640/local
       year=1640/medium
       year=1640/broad
       ...
       year=1665/broad

   Every physical index is searched independently and the results are fused.

2. Per-scale:
       local/
       medium/
       broad/

   Each scale contains the entire corpus and only three physical indexes are
   searched.

The benchmark separates:

    * index construction/loading time
    * single-query latency
    * batch-query latency
    * end-to-end retrieval latency
    * candidate/result counts

Optional exact-search evaluation can be added later to measure recall@k.

The benchmark deliberately does not treat the DiskANN distances from different
physical indexes as semantic scores. DiskANN is only responsible for candidate
retrieval. Result fusion should remain explicit and reproducible.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from lib.corpus_logging import logger

from retrieval.diskann_observation_index import (
    DiskANNObservationIndex,
)
from retrieval.diskann_observation_index_store import (
    DiskANNObservationIndexStore,
)
from retrieval.models import Float32Array, SearchResult, SearchSpace
from retrieval.observation_index import ObservationIndex


DEFAULT_DIMENSIONS = 768

SCALES = (
    "local",
    "medium",
    "broad",
)


@dataclass(slots=True)
class TimingSummary:
    """Summary statistics for a sequence of elapsed times."""

    count: int
    minimum_ms: float
    median_ms: float
    mean_ms: float
    p95_ms: float
    maximum_ms: float

    @classmethod
    def from_seconds(
        cls,
        values: list[float],
    ) -> "TimingSummary":
        if not values:
            raise ValueError("cannot summarise an empty timing sequence")

        milliseconds = sorted(
            value * 1000.0
            for value in values
        )

        count = len(milliseconds)

        p95_index = min(
            count - 1,
            int(np.ceil(count * 0.95)) - 1,
        )

        return cls(
            count=count,
            minimum_ms=milliseconds[0],
            median_ms=float(
                statistics.median(milliseconds)
            ),
            mean_ms=float(
                statistics.fmean(milliseconds)
            ),
            p95_ms=milliseconds[p95_index],
            maximum_ms=milliseconds[-1],
        )

    def log(
        self,
        label: str,
    ) -> None:
        logger.info(
            "%s: "
            "n=%d "
            "min=%.3fms "
            "median=%.3fms "
            "mean=%.3fms "
            "p95=%.3fms "
            "max=%.3fms",
            label,
            self.count,
            self.minimum_ms,
            self.median_ms,
            self.mean_ms,
            self.p95_ms,
            self.maximum_ms,
        )


@dataclass(slots=True)
class BenchmarkResult:
    """Results for one retrieval layout."""

    name: str
    index_count: int
    load_seconds: float
    single_query: TimingSummary
    batch_query: TimingSummary | None
    result_count: int

    def log(self) -> None:
        logger.info("")
        logger.info(
            "========== %s ==========",
            self.name,
        )

        logger.info(
            "Physical indexes: %d",
            self.index_count,
        )

        logger.info(
            "Index load time: %.3fs",
            self.load_seconds,
        )

        logger.info(
            "Results returned per query: %d",
            self.result_count,
        )

        self.single_query.log(
            "Single-query latency",
        )

        if self.batch_query is not None:
            self.batch_query.log(
                "Batch-query latency",
            )


def normalise(
    vectors: np.ndarray,
) -> np.ndarray:
    """
    Return unit-normalised float32 vectors.

    DiskANN uses L2 distance. For unit vectors, L2 ordering is monotonic
    with cosine similarity:

        ||a - b||² = 2 - 2 cos(a, b)
    """

    array = np.asarray(
        vectors,
        dtype=np.float32,
    )

    if array.ndim == 1:
        norm = np.linalg.norm(array)

        if not np.isfinite(norm) or norm == 0:
            raise ValueError(
                "vector must have a finite, non-zero norm"
            )

        return array / norm

    if array.ndim != 2:
        raise ValueError(
            "vectors must be one- or two-dimensional"
        )

    norms = np.linalg.norm(
        array,
        axis=1,
        keepdims=True,
    )

    if np.any(~np.isfinite(norms)):
        raise ValueError(
            "vectors must have finite norms"
        )

    if np.any(norms == 0):
        raise ValueError(
            "vectors must not contain zero vectors"
        )

    return array / norms


def load_queries(
    path: Path | None,
    *,
    dimensions: int,
    query_count: int,
    seed: int,
) -> Float32Array:
    """
    Load benchmark queries or generate deterministic synthetic unit vectors.

    Synthetic vectors are suitable for comparing latency between layouts.
    They are not suitable for measuring semantic recall.
    """

    if path is not None:
        logger.info(
            "Loading benchmark queries from %s",
            path,
        )

        queries = np.load(path)

        queries = np.asarray(
            queries,
            dtype=np.float32,
        )

        if queries.ndim != 2:
            raise ValueError(
                "query file must contain a two-dimensional array"
            )

        if queries.shape[1] != dimensions:
            raise ValueError(
                f"query dimension {queries.shape[1]} "
                f"does not match expected dimension "
                f"{dimensions}"
            )

        if len(queries) == 0:
            raise ValueError(
                "query file contains no queries"
            )

        return normalise(queries)

    logger.warning(
        "No query file supplied; generating %d synthetic "
        "unit vectors. Latency comparisons are valid, but "
        "semantic recall measurements are not.",
        query_count,
    )

    rng = np.random.default_rng(seed)

    queries = rng.standard_normal(
        size=(query_count, dimensions),
        dtype=np.float32,
    )

    return normalise(queries)


def discover_per_scale_indexes(
    root: Path,
    *,
    dimensions: int,
    scales: tuple[str, ...],
) -> list[ObservationIndex]:
    """
    Load one corpus-wide DiskANN index per scale.

    Expected layout:

        root/
            local/
                local_event_ids.npy
                ...
            medium/
                medium_event_ids.npy
                ...
            broad/
                broad_event_ids.npy
                ...
    """

    indexes: list[ObservationIndex] = []

    for scale in scales:
        index_directory = root / scale

        event_ids_path = (
            index_directory
            / f"{scale}_event_ids.npy"
        )

        if not index_directory.is_dir():
            logger.warning(
                "Missing scale index directory: %s",
                index_directory,
            )
            continue

        if not event_ids_path.is_file():
            logger.warning(
                "Missing event-ID mapping: %s",
                event_ids_path,
            )
            continue

        logger.info(
            "Loading corpus-wide %s index from %s",
            scale,
            index_directory,
        )

        indexes.append(
            DiskANNObservationIndex(
                index_directory=index_directory,
                event_ids_path=event_ids_path,
                dimensions=dimensions,
                num_threads=0,
                batch_num_threads=0,
                search_complexity=100,
                beam_width=2,
                num_nodes_to_cache=0,
                index_prefix=scale,
            )
        )

    if not indexes:
        raise RuntimeError(
            f"no corpus-wide indexes found under {root}"
        )

    return indexes


def load_per_year_indexes(
    root: Path,
) -> list[ObservationIndex]:
    """
    Load all existing year/scale indexes using the current store.

    The current DiskANNObservationIndexStore discovers:

        root/
            year=1640/
                local/
                medium/
                broad/
            ...
    """

    store = DiskANNObservationIndexStore(
        indexes_root=root,
    )

    indexes = store.get(
        SearchSpace(
            years=None,
            scale=None,
        )
    )

    if not indexes:
        raise RuntimeError(
            f"no per-year DiskANN indexes found under {root}"
        )

    return indexes


def search_all_indexes(
    indexes: list[ObservationIndex],
    query: Float32Array,
    *,
    k: int,
) -> list[SearchResult]:
    """
    Search every physical index.

    This intentionally measures the architectural cost of fan-out.

    Fusion is not performed here because the benchmark's primary question is
    how many physical DiskANN searches are required. The caller can later add
    RRF or another ranking layer without changing the raw ANN measurements.
    """

    return [
        index.search(
            query,
            k=k,
        )
        for index in indexes
    ]


def count_results(
    responses: list[SearchResult],
) -> int:
    """Count raw candidates returned across all physical indexes."""

    return sum(
        len(response.event_ids)
        for response in responses
    )


def warm_indexes(
    indexes: list[ObservationIndex],
    queries: Float32Array,
    *,
    k: int,
    warmup_queries: int,
) -> None:
    """
    Warm Python and DiskANN search paths.

    DiskANN's page/cache state can materially affect disk-backed timings.
    Warmup is therefore separated from measured queries.
    """

    if warmup_queries <= 0:
        return

    count = min(
        warmup_queries,
        len(queries),
    )

    logger.info(
        "Warming %d indexes with %d queries",
        len(indexes),
        count,
    )

    for query in queries[:count]:
        search_all_indexes(
            indexes,
            query,
            k=k,
        )


def benchmark_single_queries(
    indexes: list[ObservationIndex],
    queries: Float32Array,
    *,
    k: int,
) -> tuple[TimingSummary, int]:
    """
    Measure end-to-end fan-out latency for one query.

    Each timing includes searching every physical index.
    """

    elapsed: list[float] = []
    result_count = 0

    for query in queries:
        started = time.perf_counter()

        responses = search_all_indexes(
            indexes,
            query,
            k=k,
        )

        elapsed.append(
            time.perf_counter() - started
        )

        result_count = count_results(
            responses
        )

    return (
        TimingSummary.from_seconds(elapsed),
        result_count,
    )


def benchmark_batch_queries(
    indexes: list[ObservationIndex],
    queries: Float32Array,
    *,
    k: int,
    batch_size: int,
) -> TimingSummary | None:
    """
    Measure batched fan-out retrieval.

    The timing is reported per query, not merely per batch, so results are
    comparable with single-query latency.
    """

    if batch_size <= 1:
        return None

    elapsed_per_query: list[float] = []

    for start in range(
        0,
        len(queries),
        batch_size,
    ):
        batch = queries[
            start:start + batch_size
        ]

        started = time.perf_counter()

        for index in indexes:
            index.batch_search(
                batch,
                k=k,
            )

        elapsed = time.perf_counter() - started

        per_query = elapsed / len(batch)

        elapsed_per_query.extend(
            [per_query] * len(batch)
        )

    return TimingSummary.from_seconds(
        elapsed_per_query
    )


def benchmark_layout(
    *,
    name: str,
    loader: Callable[[], list[ObservationIndex]],
    queries: Float32Array,
    k: int,
    warmup_queries: int,
    batch_size: int,
) -> BenchmarkResult:
    """
    Load, warm and benchmark one physical DiskANN layout.

    Index construction is deliberately outside this benchmark: rebuilding an
    index measures an offline preprocessing cost, whereas this script measures
    the runtime consequences of the resulting layout.
    """

    gc.collect()

    logger.info("")
    logger.info(
        "Loading layout: %s",
        name,
    )

    load_started = time.perf_counter()

    indexes = loader()

    load_seconds = (
        time.perf_counter()
        - load_started
    )

    logger.info(
        "%s loaded %d physical indexes in %.3fs",
        name,
        len(indexes),
        load_seconds,
    )

    warm_indexes(
        indexes,
        queries,
        k=k,
        warmup_queries=warmup_queries,
    )

    single_query, result_count = (
        benchmark_single_queries(
            indexes,
            queries,
            k=k,
        )
    )

    batch_query = benchmark_batch_queries(
        indexes,
        queries,
        k=k,
        batch_size=batch_size,
    )

    return BenchmarkResult(
        name=name,
        index_count=len(indexes),
        load_seconds=load_seconds,
        single_query=single_query,
        batch_query=batch_query,
        result_count=result_count,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark per-year versus corpus-wide "
            "DiskANN retrieval layouts."
        )
    )

    parser.add_argument(
        "--per-year-root",
        type=Path,
        required=True,
        help=(
            "Root containing year=<YYYY>/<scale> "
            "DiskANN indexes."
        ),
    )

    parser.add_argument(
        "--per-scale-root",
        type=Path,
        required=True,
        help=(
            "Root containing corpus-wide "
            "local/, medium/ and broad/ indexes."
        ),
    )

    parser.add_argument(
        "--queries",
        type=Path,
        default=None,
        help=(
            "Optional .npy file containing query vectors "
            "with shape (n, 768)."
        ),
    )

    parser.add_argument(
        "--query-count",
        type=int,
        default=100,
        help=(
            "Number of synthetic queries when --queries "
            "is not supplied (default: 100)."
        ),
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
        "--k",
        type=int,
        default=60,
        help=(
            "Neighbours requested from each physical "
            "index (default: 60)."
        ),
    )

    parser.add_argument(
        "--warmup-queries",
        type=int,
        default=10,
        help=(
            "Queries used for warmup before measurement "
            "(default: 10)."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help=(
            "Batch size for batch_search benchmarking "
            "(default: 32; <=1 disables batching)."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for synthetic queries "
            "(default: 42)."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dimensions <= 0:
        raise ValueError(
            "--dimensions must be positive"
        )

    if args.query_count <= 0:
        raise ValueError(
            "--query-count must be positive"
        )

    if args.k <= 0:
        raise ValueError(
            "--k must be positive"
        )

    if args.warmup_queries < 0:
        raise ValueError(
            "--warmup-queries must be non-negative"
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive"
        )

    queries = load_queries(
        args.queries,
        dimensions=args.dimensions,
        query_count=args.query_count,
        seed=args.seed,
    )

    logger.info(
        "Benchmark queries: shape=%s",
        queries.shape,
    )

    results: list[BenchmarkResult] = []

    results.append(
        benchmark_layout(
            name="PER-YEAR",
            loader=lambda: load_per_year_indexes(
                args.per_year_root
            ),
            queries=queries,
            k=args.k,
            warmup_queries=args.warmup_queries,
            batch_size=args.batch_size,
        )
    )

    results.append(
        benchmark_layout(
            name="PER-SCALE",
            loader=lambda: discover_per_scale_indexes(
                args.per_scale_root,
                dimensions=args.dimensions,
                scales=SCALES,
            ),
            queries=queries,
            k=args.k,
            warmup_queries=args.warmup_queries,
            batch_size=args.batch_size,
        )
    )

    logger.info("")
    logger.info(
        "========================================"
    )
    logger.info(
        "DiskANN layout benchmark summary"
    )
    logger.info(
        "========================================"
    )

    for result in results:
        result.log()

    if len(results) == 2:
        per_year = results[0]
        per_scale = results[1]

        logger.info("")
        logger.info(
            "========== COMPARISON =========="
        )

        logger.info(
            "Physical-index reduction: "
            "%d -> %d (%.1fx fewer)",
            per_year.index_count,
            per_scale.index_count,
            (
                per_year.index_count
                / per_scale.index_count
            )
            if per_scale.index_count
            else float("inf"),
        )

        logger.info(
            "Load-time ratio: %.2fx",
            (
                per_year.load_seconds
                / per_scale.load_seconds
            )
            if per_scale.load_seconds
            else float("inf"),
        )

        logger.info(
            "Median single-query ratio: %.2fx",
            (
                per_year.single_query.median_ms
                / per_scale.single_query.median_ms
            )
            if per_scale.single_query.median_ms
            else float("inf"),
        )

        logger.info(
            "P95 single-query ratio: %.2fx",
            (
                per_year.single_query.p95_ms
                / per_scale.single_query.p95_ms
            )
            if per_scale.single_query.p95_ms
            else float("inf"),
        )


if __name__ == "__main__":
    main()
