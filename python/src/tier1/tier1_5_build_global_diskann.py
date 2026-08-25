#!/usr/bin/env python
# tier1_5_build_global_diskann.py

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

import lib.corpus_config as config
from lib.corpus_logging import logger
from retrieval.diskann_builder import build_diskann_index
from retrieval.parquet_embeddings import load_embeddings

SCALES = ("local", "medium", "broad")
DEFAULT_DIMENSIONS = 768
DEFAULT_CHUNK_SIZE = 100_000


def available_years(
    store: Path,
) -> tuple[int, ...]:
    years: list[int] = []

    for path in store.glob("year=*"):
        if not path.is_dir():
            continue

        try:
            year = int( path.name.removeprefix("year=") )
        except ValueError:
            continue

        years.append(year)

    return tuple(sorted(set(years)))


def count_observations(
    *,
    store: Path,
    years: tuple[int, ...],
    scale: str,
    dimensions: int,
) -> int:
    total = 0

    for year in years:
        event_ids, vectors = load_embeddings(
            store,
            year_start=year,
            year_end=year,
            scale=scale,
            dimensions=dimensions,
        )

        if vectors.ndim != 2:
            raise ValueError( f"Expected two-dimensional vectors for year={year} scale={scale}" )

        if vectors.shape[1] != dimensions:
            raise ValueError( f"Unexpected dimensions for year={year} scale={scale}: {vectors.shape}" )

        if len(event_ids) != vectors.shape[0]:
            raise ValueError( f"Event/vector count mismatch for year={year} scale={scale}" )

        total += len(event_ids)

        del event_ids
        del vectors

    return total


def populate_memmaps(
    *,
    store: Path,
    years: tuple[int, ...],
    scale: str,
    dimensions: int,
    vectors_path: Path,
    event_ids_path: Path,
    count: int,
) -> None:
    vectors = np.memmap(
        vectors_path,
        dtype=np.float32,
        mode="w+",
        shape=(count, dimensions),
    )

    event_ids = np.memmap(
        event_ids_path,
        dtype=np.uint64,
        mode="w+",
        shape=(count,),
    )

    offset = 0

    try:
        for year in years:
            logger.info(
                "[tier1.5] loading year=%s scale=%s",
                year,
                scale,
            )

            year_event_ids, year_vectors = load_embeddings(
                store,
                year=year,
                scale=scale,
                dimensions=dimensions,
            )

            n = len(year_event_ids)

            if n != year_vectors.shape[0]:
                raise ValueError(
                    f"Event/vector count mismatch for "
                    f"year={year} scale={scale}"
                )

            end = offset + n

            if end > count:
                raise RuntimeError(
                    "Observation count changed between passes"
                )

            vectors[offset:end] = year_vectors
            event_ids[offset:end] = year_event_ids

            offset = end

            logger.info(
                "[tier1.5] populated year=%s scale=%s "
                "observations=%d total=%d",
                year,
                scale,
                n,
                offset,
            )

            del year_event_ids
            del year_vectors

        if offset != count:
            raise RuntimeError(
                f"Expected {count} observations but populated {offset}"
            )

        vectors.flush()
        event_ids.flush()

    finally:
        del vectors
        del event_ids


def build_one(
    *,
    store: Path,
    output_root: Path,
    work_root: Path,
    years: tuple[int, ...],
    scale: str,
    dimensions: int,
    complexity: int,
    graph_degree: int,
    search_memory_gb: float,
    build_memory_gb: float,
    num_threads: int,
    pq_disk_bytes: int,
    normalisation_chunk_size: int,
) -> Path:
    logger.info(
        "[tier1.5] counting observations: scale=%s",
        scale,
    )

    count = count_observations(
        store=store,
        years=years,
        scale=scale,
        dimensions=dimensions,
    )

    logger.info(
        "[tier1.5] scale=%s observations=%d "
        "raw_vectors=%.2f GiB",
        scale,
        count,
        count * dimensions * 4 / (1024 ** 3),
    )

    work_directory = (
        work_root
        / scale
    )

    if work_directory.exists():
        shutil.rmtree(work_directory)

    work_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    vectors_path = (
        work_directory
        / f"{scale}_vectors.dat"
    )

    event_ids_path = (
        work_directory
        / f"{scale}_event_ids.dat"
    )

    populate_memmaps(
        store=store,
        years=years,
        scale=scale,
        dimensions=dimensions,
        vectors_path=vectors_path,
        event_ids_path=event_ids_path,
        count=count,
    )

    vectors = np.memmap(
        vectors_path,
        dtype=np.float32,
        mode="r+",
        shape=(count, dimensions),
    )

    event_ids = np.memmap(
        event_ids_path,
        dtype=np.uint64,
        mode="r",
        shape=(count,),
    )

    output_directory = (
        output_root
        / scale
    )

    if output_directory.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing index: "
            f"{output_directory}"
        )

    logger.info(
        "[tier1.5] building global DiskANN: scale=%s "
        "observations=%d output=%s",
        scale,
        count,
        output_directory,
    )

    build_diskann_index(
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
        normalisation_chunk_size=normalisation_chunk_size,
    )

    del vectors
    del event_ids

    logger.info(
        "[tier1.5] global DiskANN complete: scale=%s",
        scale,
    )

    return output_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build corpus-wide DiskANN indexes, one per embedding scale."
        )
    )

    parser.add_argument(
        "--store",
        type=Path,
        default=config.EVENTSTORE_T1_PATH,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(config.DISKANN_INDEXES_DIR)
            / "global"
        ),
    )

    parser.add_argument(
        "--work",
        type=Path,
        default=(
            Path(config.DISKANN_INDEXES_DIR)
            / "_global_build"
        ),
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
        help="0 lets DiskANN choose its thread count.",
    )

    parser.add_argument(
        "--pq-disk-bytes",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--normalisation-chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
    )

    parser.add_argument(
        "--scale",
        choices=SCALES,
        action="append",
        help=(
            "Scale to build. May be supplied multiple times. "
            "Defaults to all scales."
        ),
    )

    parser.add_argument(
        "--year-from",
        type=int,
    )

    parser.add_argument(
        "--year-to",
        type=int,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    years = available_years(args.store)

    if args.year_from is not None:
        years = tuple(
            year
            for year in years
            if year >= args.year_from
        )

    if args.year_to is not None:
        years = tuple(
            year
            for year in years
            if year <= args.year_to
        )

    if not years:
        raise RuntimeError(
            f"No Parquet years found under {args.store}"
        )

    scales = (
        tuple(args.scale)
        if args.scale
        else SCALES
    )

    logger.info(
        "[tier1.5] global build years=%s",
        years,
    )

    logger.info(
        "[tier1.5] global build scales=%s",
        scales,
    )

    for scale in scales:
        build_one(
            store=args.store,
            output_root=args.output,
            work_root=args.work,
            years=years,
            scale=scale,
            dimensions=args.dimensions,
            complexity=args.complexity,
            graph_degree=args.graph_degree,
            search_memory_gb=args.search_memory_gb,
            build_memory_gb=args.build_memory_gb,
            num_threads=args.num_threads,
            pq_disk_bytes=args.pq_disk_bytes,
            normalisation_chunk_size=(
                args.normalisation_chunk_size
            ),
        )


if __name__ == "__main__":
    main()

