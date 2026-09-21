# tier1/vector_writer.py
"""
Shared Lance vector writer for all Tier 1 scales (local / medium / broad).

Schema is deliberately lean. PostgreSQL's `events` table is the single
source of truth for provenance — corpus, doc_id, token, token_idx, and
each scale's window_id / window_token_pos (see db_observation_backend.py).
Lance stores only what's needed to bucket, filter, and search vectors
without a join:

    event_id         -- joins back to events.event_id
    pub_year          -- canonical name; matches events.pub_year exactly
    embedding_model
    vector

Anything else (window_id, start_idx, end_idx, doc_id, corpus, ...) lives
in Postgres and should be looked up via event_id, never duplicated here.
Duplicating it invites drift: a partial failure between insert_events()
and table.add() would leave two copies of the same fact free to disagree.

One VectorWriter instance = one scale. Table names are namespaced as
"{scale}__{model_name}__{year_start:04d}_{year_end:04d}", so local,
medium, and broad tables can never collide even when multiple writers
run concurrently against the same lance_root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import lancedb
import numpy as np
import pyarrow as pa

from lib.corpus_logging import logger

VECTOR_INDEX_TYPE = "IVF_FLAT"
VECTOR_INDEX_METRIC = "cosine"

MIN_VECTOR_INDEX_PARTITIONS = 1
MAX_VECTOR_INDEX_PARTITIONS = 256

# Names Lance assigns by default to create_index()/create_scalar_index()
# calls below (index_type + "_" + column + "_idx" pattern collapses to
# this for the unnamed calls we make). Kept as a named constant so
# build_indexes() has one place to update if that ever changes.
EXPECTED_INDEXES = {
    "vector_idx",
    "event_id_idx",
    "pub_year_idx",
    "embedding_model_idx",
}


def year_bucket(year: int, bucket_size: int) -> tuple[int, int]:
    start = (year // bucket_size) * bucket_size
    return start, start + bucket_size - 1


def lance_table_name(
    scale: str,
    model_name: str,
    year: int,
    bucket_size: int,
) -> str:
    start, end = year_bucket(year, bucket_size)
    return f"{scale}__{model_name}__{start:04d}_{end:04d}"


def vector_index_partitions(row_count: int) -> int:
    """
    Size num_partitions from the table's actual row count.

    sqrt(row_count) is a standard IVF starting heuristic: it keeps the
    average partition size (and therefore per-partition training/search
    cost) growing sublinearly as the table grows, without requiring more
    partitions than a small bucket has rows to support.
    """
    if row_count <= 0:
        raise ValueError("row_count must be positive")

    return max(
        MIN_VECTOR_INDEX_PARTITIONS,
        min(MAX_VECTOR_INDEX_PARTITIONS, int(row_count ** 0.5)),
    )


class VectorWriter:
    """
    Writes vectors for ONE scale to year-bucketed Lance tables.

    Used by both tier1_seeds2events.py (one instance per entry in
    ACTIVE_SCALES) and tier1_corpus2events.py (one instance for
    whichever --scale the worker was started with), so every scale
    goes through the exact same schema, dedup logic, and index build.
    """

    def __init__(
        self,
        lance_root: Path,
        *,
        scale: str,
        model_name: str,
        bucket_size: int,
        dry_run: bool = False,
    ) -> None:
        self.scale = scale
        self.model_name = model_name
        self.bucket_size = bucket_size
        self.dry_run = dry_run
        self.tables: dict[str, object] = {}

        if not dry_run:
            self.db = lancedb.connect(str(lance_root))

    def table_name(self, pub_year: int) -> str:
        return lance_table_name(
            self.scale, self.model_name, pub_year, self.bucket_size
        )

    def write(
        self,
        *,
        event_ids: Sequence[int],
        pub_year: int,
        vectors: np.ndarray,
    ) -> int:
        """
        Write one batch of vectors, all belonging to the same pub_year
        (i.e. the same Lance table). Rows whose event_id is already
        present are skipped, so repeated/overlapping calls are safe.
        """
        n = len(event_ids)
        assert n == len(vectors), "event_ids and vectors length mismatch"

        if n == 0:
            return 0

        if pub_year is None:
            raise ValueError(
                "pub_year is required to bucket a Lance write "
                f"(scale={self.scale})"
            )

        rows = [
            {
                "event_id": int(event_ids[i]),
                "pub_year": int(pub_year),
                "embedding_model": self.model_name,
                "vector": np.asarray(vectors[i], dtype=np.float32).tolist(),
            }
            for i in range(n)
        ]

        if self.dry_run:
            logger.info(
                "[vector_writer] dry-run: would write %d rows to %s",
                n, self.table_name(pub_year),
            )
            return n

        table = self._open_table(
            self.table_name(pub_year),
            dim=len(rows[0]["vector"]),
        )

        existing = self._existing_ids(
            table, {row["event_id"] for row in rows}
        )
        new_rows = [row for row in rows if row["event_id"] not in existing]

        if new_rows:
            table.add(new_rows, mode="append")

        return len(new_rows)

    def _existing_ids(self, table, event_ids: set[int]) -> set[int]:
        if not event_ids or table.count_rows() == 0:
            return set()

        arrow = table.to_arrow()
        return set(arrow.column("event_id").to_pylist()) & event_ids

    def _open_table(self, name: str, *, dim: int):
        if name in self.tables:
            return self.tables[name]

        existing = set(self.db.list_tables().tables)

        if name in existing:
            table = self.db.open_table(name)
        else:
            logger.info("[vector_writer] creating Lance table %s", name)
            schema = pa.schema([
                pa.field("event_id", pa.int64()),
                pa.field("pub_year", pa.int32()),
                pa.field("embedding_model", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), dim)),
            ])
            table = self.db.create_table(name, schema=schema)

        self.tables[name] = table
        return table

    def index_existing_tables(self) -> None:
        """Load every table already on disk for this scale, then index them."""
        prefix = f"{self.scale}__{self.model_name}__"

        for table_name in sorted(
            name for name in self.db.list_tables().tables
            if name.startswith(prefix)
        ):
            self.tables[table_name] = self.db.open_table(table_name)

        self.build_indexes()

    def build_indexes(self) -> None:
        """
        Rebuild indexes for tables that are missing an index or contain
        unindexed rows.

        Indexes are treated as derived acceleration structures. A table
        is complete only when all required indexes exist and cover
        every row.
        """
        for table_name, table in self.tables.items():
            row_count = table.count_rows()

            if row_count == 0:
                continue

            indices = {index.name: index for index in table.list_indices()}

            if (
                EXPECTED_INDEXES <= indices.keys()
                and all(
                    indices[name].num_unindexed_rows == 0
                    for name in EXPECTED_INDEXES
                )
            ):
                logger.info(
                    "[vector_writer] indexes already complete for %s (%d rows)",
                    table_name, row_count,
                )
                continue

            num_partitions = vector_index_partitions(row_count)

            logger.info(
                "[vector_writer] rebuilding indexes for %s (%d rows, %d partitions)",
                table_name, row_count, num_partitions,
            )

            table.create_index(
                metric=VECTOR_INDEX_METRIC,
                index_type=VECTOR_INDEX_TYPE,
                vector_column_name="vector",
                num_partitions=num_partitions,
                replace=True,
            )

            table.create_scalar_index("event_id", index_type="BTREE", replace=True)
            table.create_scalar_index("pub_year", index_type="BTREE", replace=True)
            table.create_scalar_index("embedding_model", index_type="BTREE", replace=True)
