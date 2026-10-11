
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lib.corpus_config import (
    LANCE_INDEXES_DIR,
    LANCE_BUCKET_SIZE,
    LANCE_MODEL_NAME,
)
from lib.corpus_logging import logger
from tier1.vector_writer import year_bucket, vector_index_partitions


# Local export produced from Colab's passages_*.parquet files.
INPUT_DIR = Path("../out/export/passages")

REPRESENTATIONS = ("l8", "last", "mean4")
VECTOR_DIM = 768

# Keep individual writes reasonably small.
WRITE_BATCH_SIZE = 512
READ_BATCH_SIZE = 256

# Index creation is deferred until all data has been imported.
BUILD_INDEXES = True


def passage_id(
    corpus: str,
    doc_id: str,
    word_start: int,
    word_end: int,
) -> str:
    """Stable identity for a passage, independent of its vector values."""
    identity = (
        f"{corpus}\0{doc_id}\0{word_start}\0{word_end}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def table_name(representation: str, pub_year: int) -> str:
    start, end = year_bucket(pub_year, LANCE_BUCKET_SIZE)
    return (
        f"passage_{representation}__{LANCE_MODEL_NAME}"
        f"__{start:04d}_{end:04d}"
    )


def make_schema() -> pa.Schema:
    return pa.schema([
        pa.field("passage_id", pa.string()),
        pa.field("corpus", pa.string()),
        pa.field("doc_id", pa.string()),
        pa.field("pub_year", pa.int32()),
        pa.field("word_start", pa.int32()),
        pa.field("word_end", pa.int32()),
        pa.field("token_start", pa.int32()),
        pa.field("token_end", pa.int32()),
        pa.field("embedding_model", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
    ])


class PassageWriter:
    """Idempotent writer for passage vectors, separate from event vectors."""

    def __init__(self, root: Path) -> None:
        self.db = lancedb.connect(str(root))
        self.schema = make_schema()
        self.tables = {}

    def get_table(self, name: str):
        if name in self.tables:
            return self.tables[name]

        existing = set(self.db.list_tables().tables)

        if name in existing:
            table = self.db.open_table(name)
        else:
            logger.info("[passages] creating Lance table %s", name)
            table = self.db.create_table(
                name,
                schema=self.schema,
            )

        self.tables[name] = table
        return table

    @staticmethod
    def existing_ids(table, ids: set[str]) -> set[str]:
        if not ids or table.count_rows() == 0:
            return set()

        found = set()
        ordered = sorted(ids)

        # passage_id contains only hexadecimal SHA-256 characters.
        for i in range(0, len(ordered), 500):
            chunk = ordered[i:i + 500]
            quoted = ",".join(f"'{value}'" for value in chunk)
            result = (
                table.search()
                .where(f"passage_id IN ({quoted})")
                .select(["passage_id"])
                .limit(len(chunk))
                .to_arrow()
            )
            found.update(
                result.column("passage_id").to_pylist()
            )

        return found

    def write(self, name: str, rows: list[dict]) -> int:
        if not rows:
            return 0

        table = self.get_table(name)

        ids = {row["passage_id"] for row in rows}
        existing = self.existing_ids(table, ids)

        # Also deduplicate within this incoming batch.
        seen = set(existing)
        new_rows = []

        for row in rows:
            pid = row["passage_id"]
            if pid in seen:
                continue
            seen.add(pid)
            new_rows.append(row)

        if new_rows:
            table.add(new_rows, mode="append")

        return len(new_rows)

    def build_indexes(self) -> None:
        for name, table in self.tables.items():
            count = table.count_rows()
            if count == 0:
                continue

            indexes = {
                index.name: index
                for index in table.list_indices()
            }

            vector_index_ready = (
                "vector_idx" in indexes
                and indexes["vector_idx"].num_unindexed_rows == 0
            )

            if not vector_index_ready:
                partitions = vector_index_partitions(count)
                logger.info(
                    "[passages] indexing %s: %d rows, %d partitions",
                    name, count, partitions,
                )
                table.create_index(
                    metric="cosine",
                    index_type="IVF_FLAT",
                    vector_column_name="vector",
                    num_partitions=partitions,
                    replace=True,
                )

            for column in ("passage_id", "pub_year"):
                index_name = f"{column}_idx"
                current = {
                    index.name: index
                    for index in table.list_indices()
                }
                if (
                    index_name not in current
                    or current[index_name].num_unindexed_rows > 0
                ):
                    table.create_scalar_index(
                        column,
                        index_type="BTREE",
                        replace=True,
                    )


def flush_buffers(
    writer: PassageWriter,
    buffers: dict,
) -> int:
    written = 0

    for name, rows in list(buffers.items()):
        if rows:
            written += writer.write(name, rows)
            buffers[name] = []

    return written


def import_shard(
    path: Path,
    writer: PassageWriter,
) -> tuple[int, int]:
    parquet = pq.ParquetFile(path)

    required = {
        "corpus", "doc_id", "pub_year",
        "word_start", "word_end",
        "token_start", "token_end",
        *(f"vec_{name}" for name in REPRESENTATIONS),
    }
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(
            f"{path.name}: missing columns: {sorted(missing)}"
        )

    buffers = defaultdict(list)
    passages_seen = 0
    rows_written = 0

    for batch in parquet.iter_batches(batch_size=READ_BATCH_SIZE):

        # Validate each representation in bulk before converting rows.
        for rep in REPRESENTATIONS:
            arr = batch.column(f"vec_{rep}")
            vec = arr.values.to_numpy(
                zero_copy_only=False
            ).reshape(-1, VECTOR_DIM)

            bad = (
                ~np.isfinite(vec).all(axis=1)
                | (np.abs(vec).sum(axis=1) == 0)
            )

            if bad.any():
                bad_rows = np.flatnonzero(bad)
                raise ValueError(
                    f"{path.name}: {len(bad_rows)} invalid "
                    f"{rep} vectors in batch; "
                    f"batch row offsets={bad_rows[:10].tolist()}"
                )

        for row in batch.to_pylist():
            passages_seen += 1

            year = row["pub_year"]
            if year is None:
                raise ValueError(
                    f"{path.name}: passage has no publication year; "
                    f"doc_id={row['doc_id']!r}"
                )

            start = int(row["word_start"])
            end = int(row["word_end"])
            if end <= start:
                raise ValueError(
                    f"{path.name}: invalid passage span {start}:{end}"
                )

            pid = passage_id(
                row["corpus"],
                row["doc_id"],
                start,
                end,
            )

            common = {
                "passage_id": pid,
                "corpus": row["corpus"],
                "doc_id": row["doc_id"],
                "pub_year": int(year),
                "word_start": start,
                "word_end": end,
                "token_start": int(row["token_start"]),
                "token_end": int(row["token_end"]),
                "embedding_model": LANCE_MODEL_NAME,
            }

            for representation in REPRESENTATIONS:
                vector = np.asarray(
                    row[f"vec_{representation}"],
                    dtype=np.float32,
                )

                if vector.shape != (VECTOR_DIM,):
                    raise ValueError(
                        f"{path.name}: {representation} vector has "
                        f"shape {vector.shape}, expected ({VECTOR_DIM},)"
                    )

                name = table_name(representation, int(year))
                buffers[name].append({
                    **common,
                    "vector": vector.tolist(),
                })

                if len(buffers[name]) >= WRITE_BATCH_SIZE:
                    rows_written += writer.write(
                        name, buffers[name]
                    )
                    buffers[name] = []

        if passages_seen % 10_000 == 0:
            logger.info(
                "[passages] %s: processed %d passages",
                path.name, passages_seen,
            )

    rows_written += flush_buffers(writer, buffers)

    logger.info(
        "[passages] %s: input passages=%d, new vector rows=%d",
        path.name, passages_seen, rows_written,
    )
    return passages_seen, rows_written


def main() -> None:
    input_dir = INPUT_DIR.resolve()

    if not input_dir.exists():
        raise SystemExit(
            f"Passage input directory does not exist: {input_dir}"
        )

    shards = sorted(input_dir.glob("passages_*.parquet"))
    if not shards:
        raise SystemExit(
            f"No passages_*.parquet files found in {input_dir}"
        )

    logger.info("[passages] input: %s", input_dir)
    logger.info("[passages] Lance root: %s", LANCE_INDEXES_DIR)
    logger.info("[passages] found %d shard(s)", len(shards))

    writer = PassageWriter(LANCE_INDEXES_DIR)

    total_passages = 0
    total_vectors = 0

    for shard in shards:
        passages, vectors = import_shard(shard, writer)
        total_passages += passages
        total_vectors += vectors

    if BUILD_INDEXES:
        writer.build_indexes()

    logger.info(
        "[passages] complete: passages read=%d, new vector rows=%d",
        total_passages, total_vectors,
    )

    for name, table in sorted(writer.tables.items()):
        logger.info(
            "[passages] table %s: %d rows",
            name, table.count_rows(),
        )


if __name__ == "__main__":
    main()