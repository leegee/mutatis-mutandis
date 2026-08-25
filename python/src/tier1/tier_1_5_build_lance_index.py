#!/usr/bin/env python
"""
tier1_5_build_lance_index.py

Build one LanceDB table per embedding scale (local/medium/broad), covering
every year in the Parquet observation store.

This replaces physical bucket-partitioned indices (tier1_5_build_diskann_index.py
/ tier1_5_build_all_diskann_indexes.py) with a single index per scale plus a
scalar range index on `year`. A 50-year search window becomes a query-time
filter (`year >= start AND year <= end`) instead of a directory you had to
pre-decide the boundaries of and rebuild on schema/boundary changes.

What this does NOT yet handle:
    - Cross-model separation. If a scale's embeddings span both the
      MacBERTh-derived and modern-BERT-derived eras, mixing them in one
      table's vector index is the same category of bug as mixing them in
      one DiskANN index -- see tier1_5_build_all_diskann_indexes.py's
      model_boundary_year handling for the reasoning. This script detects
      an `embedding_model` column IF one is already present in the Parquet
      schema and, if so, uses the real values. When the column is missing
      it fabricates a constant value of "macberth" for every row so that
      the column always exists and can be filtered/indexed. Once the real
      provenance column lands in the store, cross-era queries become safe.

Ingestion streams one year at a time via table.add(), so no single Python
process ever holds a full scale's vectors in memory -- unlike the pilot
script, which called load_embeddings() once for the whole requested range.

Index type is chosen from the corpus-wide observation count for that scale:
    - Small enough to fit comfortably in RAM at full precision -> IvfFlat
      (no PQ compression -- exact vectors, better recall, no quantization
      loss). This is very likely what you want for the sparser early
      centuries of the corpus.
    - Larger than that -> IvfPq, with num_partitions scaled to corpus size.
      Sub-vector count follows the same reasoning discussed in the DiskANN
      orchestrator: pick a size where dimensions divide evenly.

These thresholds are heuristic starting points, not measured constants --
validate against the recall/build-time pilot results for a given scale
before trusting them at full scale.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import lancedb
import pyarrow as pa
import pyarrow.dataset as ds
from lancedb.index import BTree, IvfFlat, IvfPq

import lib.corpus_config as config
from lib.corpus_logging import logger
from retrieval.parquet_embeddings import load_embeddings

SCALES = ("local", "medium", "broad")
DEFAULT_DIMENSIONS = 768
DEFAULT_MODEL = "macberth"

# Below this raw footprint, skip PQ compression entirely and use IvfFlat.
# 83,375 vectors at 768 dims (~245 MiB) measured comfortably fast in the
# pilot; this threshold gives real headroom under a 64 GB budget while
# still leaving room for query-time caches and the OS.
FLAT_INDEX_MAX_RAW_GB = 8.0


def discover_years(store: Path) -> list[int]:
    dataset = ds.dataset(store, format="parquet", partitioning="hive")
    years = (
        dataset
        .scanner(columns=["year"])
        .to_table()
        .column("year")
        .to_pylist()
    )
    return sorted(set(int(year) for year in years))


def detect_model_column(store: Path) -> str | None:
    """
    Return "embedding_model" if the Parquet schema already carries per-row
    model provenance, else None. Never fabricates this from year -- see
    module docstring.
    """
    dataset = ds.dataset(store, format="parquet", partitioning="hive")
    if "embedding_model" in dataset.schema.names:
        return "embedding_model"
    return None


def choose_index_config(
    *,
    observation_count: int,
    dimensions: int,
):
    """
    Density-aware index type/parameter selection, mirroring the reasoning
    in tier1_5_build_all_diskann_indexes.size_build_parameters(): don't pay
    for lossy compression where raw vectors already fit comfortably.
    """
    raw_gb = (observation_count * dimensions * 4) / (1024 ** 3)

    if raw_gb <= FLAT_INDEX_MAX_RAW_GB:
        logger.info(
            "Corpus footprint %.2f GiB <= %.2f GiB threshold: using IvfFlat (no PQ).",
            raw_gb,
            FLAT_INDEX_MAX_RAW_GB,
        )
        num_partitions = max(16, min(1024, round(math.sqrt(observation_count))))
        return IvfFlat(distance_type="l2", num_partitions=num_partitions)

    logger.info(
        "Corpus footprint %.2f GiB > %.2f GiB threshold: using IvfPq.",
        raw_gb,
        FLAT_INDEX_MAX_RAW_GB,
    )
    num_partitions = max(64, min(4096, round(math.sqrt(observation_count))))
    num_sub_vectors = dimensions // 8 if dimensions % 8 == 0 else dimensions // 4
    return IvfPq(
        distance_type="l2",
        num_partitions=num_partitions,
        num_sub_vectors=num_sub_vectors,
    )


def build_scale_table(
    *,
    store: Path,
    db_path: Path,
    scale: str,
    dimensions: int,
    years: list[int],
    model_column: str | None,
    default_model: str = DEFAULT_MODEL,
) -> None:
    db = lancedb.connect(str(db_path))
    table_name = scale
    tbl = None
    t0 = time.time()

    for year in years:
        event_ids, vectors = load_embeddings(
            store,
            year_start=year,
            year_end=year,
            scale=scale,
            dimensions=dimensions,
        )
        n = len(event_ids)
        if n == 0:
            logger.info("year=%d scale=%s: no observations, skipping", year, scale)
            continue

        data = {
            "event_id": pa.array(event_ids, type=pa.uint64()),
            "year": pa.array([year] * n, type=pa.int32()),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(vectors.reshape(-1), type=pa.float32()),
                dimensions,
            ),
        }

        # Always emit an embedding_model column
        if model_column:
            dataset = ds.dataset(store, format="parquet", partitioning="hive")
            model_tbl = (
                dataset
                .scanner(
                    columns=["event_id", model_column],
                    filter=ds.field("year") == year,
                )
                .to_table()
            )
            model_map = dict(
                zip(
                    model_tbl.column("event_id").to_pylist(),
                    model_tbl.column(model_column).to_pylist(),
                )
            )
            model_values = [model_map.get(eid, default_model) for eid in event_ids]
        else:
            model_values = [default_model] * n

        data["embedding_model"] = pa.array(model_values)

        batch = pa.table(data)

        if tbl is None:
            tbl = db.create_table(table_name, data=batch, mode="overwrite")
        else:
            tbl.add(batch)

        logger.debug( "year=%d scale=%s: appended %d rows (table total=%d)", year, scale, n, tbl.count_rows(), )

    if tbl is None:
        raise RuntimeError(f"No observations found for scale={scale}")

    ingest_seconds = time.time() - t0
    total_rows = tbl.count_rows()
    logger.info( "scale=%s: ingestion complete, %d rows in %.1fs", scale, total_rows, ingest_seconds, )

    # Vector index
    index_config = choose_index_config(
        observation_count=total_rows,
        dimensions=dimensions,
    )
    t0 = time.time()
    tbl.create_index("vector", config=index_config)
    logger.info("scale=%s: vector index built in %.1fs", scale, time.time() - t0)

    # Year scalar index
    t0 = time.time()
    tbl.create_index("year", config=BTree())
    logger.info("scale=%s: year scalar index built in %.1fs", scale, time.time() - t0)

    # Model provenance index (always present)
    t0 = time.time()
    tbl.create_index("embedding_model", config=BTree())
    logger.info(
        "scale=%s: embedding_model scalar index built in %.1fs",
        scale,
        time.time() - t0,
    )


def query_window(
    *,
    db_path: Path,
    scale: str,
    query_vector,
    year_start: int,
    year_end: int,
    k: int = 10,
    nprobes: int = 20,
    model: str | None = None,
) -> list[dict]:
    """
    Example of the thing this whole migration is for: a 50-year (or any)
    search window as a query-time filter against a single scale-wide
    table, replacing "find and merge results across N physical bucket
    indices."
    """
    db = lancedb.connect(str(db_path))
    tbl = db.open_table(scale)

    where = f"year >= {year_start} AND year <= {year_end}"
    if model is not None:
        where += f" AND embedding_model = '{model}'"

    return (
        tbl.search(query_vector)
        .where(where)
        .nprobes(nprobes)
        .limit(k)
        .to_list()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one LanceDB table per scale across the full corpus."
    )
    parser.add_argument("--store", type=Path, default=config.EVENTSTORE_T1_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(config.LANCE_INDEXES_DIR),
        help="Root directory for the Lance database (one table per scale).",
    )
    parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    parser.add_argument(
        "--scale",
        choices=SCALES,
        action="append",
        help="Scale to build. May be supplied multiple times. Defaults to all scales.",
    )
    parser.add_argument("--year-from", type=int)
    parser.add_argument("--year-to", type=int)
    parser.add_argument(
        "--default-model",
        type=str,
        default=DEFAULT_MODEL,
        help="Value written for embedding_model when the column is absent from the store.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    years = discover_years(args.store)
    if args.year_from is not None:
        years = [y for y in years if y >= args.year_from]
    if args.year_to is not None:
        years = [y for y in years if y <= args.year_to]
    if not years:
        raise RuntimeError(f"No Parquet years found under {args.store}")

    model_column = detect_model_column(args.store)
    if model_column:
        logger.info("Detected embedding model provenance column: %s", model_column)
    else:
        logger.info( "No embedding_model column in source schema; defaulting every row to '%s'.", args.default_model, )

    scales = tuple(args.scale) if args.scale else SCALES
    logger.info( "Building Lance tables for scales=%s, years=%d-%d", scales, min(years), max(years), )

    for scale in scales:
        logger.info("=" * 70)
        logger.info("scale=%s", scale)
        build_scale_table(
            store=args.store,
            db_path=args.output,
            scale=scale,
            dimensions=args.dimensions,
            years=years,
            model_column=model_column,
            default_model=args.default_model,
        )

    logger.info("=" * 70)
    logger.info("Lance build complete: %s", args.output)


if __name__ == "__main__":
    main()
