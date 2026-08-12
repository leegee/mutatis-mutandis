"""
parquet_observation_backend.py — Parquet + DuckDB backend for the observation API

Stage 2 of the storage refactor. Implements ObservationWriter,
ObservationStream and ObservationLookup against a hive-partitioned
Parquet layout, with DuckDB for predicate pushdown and batched reads.

Layout
------
    <root>/
        year=1640/
            part-000000.parquet
            part-000001.parquet
        year=1641/
            ...

Each part file contains the full observation schema (metadata + three
embedding scales). Partitioning by pub_year matches the year_filter path
used by FAISS builds and diachronic queries.

Embeddings are stored as Arrow fixed-size lists of float32 so DuckDB can
project them without decoding the entire row group when only metadata is
needed.

TODO
----
Enforce a clear separation of concerns.

Usage
-----
    from tier1.observation_store_api import (
        open_observation_writer,
        open_observation_stream,
        open_observation_lookup,
    )
    # import registers the backend
    import tier1.parquet_observation_backend  # noqa: F401

    writer = open_observation_writer("parquet", root, dim=768)
    stream = open_observation_stream("parquet", root)
    lookup = open_observation_lookup("parquet", root)
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tier1.observation_store_api import (
    DEFAULT_ENSEMBLE_WEIGHTS,
    NO_WINDOW_TOKEN_POS,
    register_backend,
)

# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _embedding_type(dim: int) -> pa.DataType:
    return pa.list_(pa.float32(), dim)


def observation_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            ("event_id", pa.int64()),
            ("corpus", pa.string()),
            ("doc_id", pa.string()),
            ("token", pa.string()),
            ("token_idx", pa.int64()),
            ("pub_year", pa.int16()),
            ("local_window_id", pa.int64()),
            ("local_window_token_pos", pa.int32()),
            ("medium_window_id", pa.int64()),
            ("medium_window_token_pos", pa.int32()),
            ("broad_window_id", pa.int64()),
            ("broad_window_token_pos", pa.int32()),
            ("emb_local", _embedding_type(dim)),
            ("emb_medium", _embedding_type(dim)),
            ("emb_broad", _embedding_type(dim)),
        ]
    )


def _glob(root: Path) -> str:
    """DuckDB-friendly recursive glob over all part files."""
    # hive-style year=*/part-*.parquet plus any loose *.parquet at root
    return str(root / "**" / "*.parquet")


# ---------------------------------------------------------------------------
# Parquet write tuning (Tier 1 is write-once, read-many)
# ---------------------------------------------------------------------------
#
# Workload: string metadata (high dict benefit) + three float32 embedding
# columns (modest zstd gain, dominate size). Prefer stronger compression
# and moderate row groups so DuckDB can skip unread row groups without
# decoding multi-hundred-MB blocks into RAM.
#
# Defaults chosen for EEBO-scale offline builds on workstation SSDs:
#   zstd level 6  — clearly smaller than level 3, still fast enough for
#                   Tier 1 (embedding time dominates, not Parquet I/O)
#   row_group_size 16_384 — ~16k events × 3 × 768 × 4 ≈ 150 MB uncompressed
#                   embeddings per group; comfortable for 64 GB hosts
#   data_page_size 1 MiB — fewer pages than the tiny default under large
#                   fixed-size lists, better sequential scan throughput
#   dictionary on corpus/doc_id/token only — embeddings stay plain

PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 6
PARQUET_ROW_GROUP_SIZE = 16_384
PARQUET_DATA_PAGE_SIZE = 1 << 20  # 1 MiB
PARQUET_DICT_COLUMNS = ("corpus", "doc_id", "token")


def write_observation_parquet(
    table: pa.Table,
    path: str | Path,
    *,
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
    data_page_size: int = PARQUET_DATA_PAGE_SIZE,
    use_dictionary: Sequence[str] | bool | None = None,
) -> None:
    """
    Write an observation table with project-standard Parquet encoding.

    Used by ParquetObservationWriter and compact_parquet_parts so online
    appends and offline compaction stay consistent.
    """
    if use_dictionary is None:
        use_dictionary = list(PARQUET_DICT_COLUMNS)
    path = Path(path)
    pq.write_table(
        table,
        path,
        compression=compression,
        compression_level=compression_level,
        row_group_size=row_group_size,
        data_page_size=data_page_size,
        use_dictionary=use_dictionary,
        write_statistics=True,
        # Column order matches schema; statistics enable year/token filters.
        coerce_timestamps="us",
    )


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    # Prefer threads for scan; callers may run many sequential queries.
    con.execute("SET threads TO 4")
    return con


def _has_parquet(root: Path) -> bool:
    if not root.exists():
        return False
    return any(root.rglob("*.parquet"))



# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

# Default part sizing: avoid one tiny file per document.
# ~100k rows × 3 × 768 float32 ≈ 0.9 GB uncompressed; zstd is much smaller.
DEFAULT_MIN_ROWS = 100_000
DEFAULT_MIN_BYTES = 256 * 1024 * 1024  # 256 MiB estimated uncompressed payload


class ParquetObservationWriter:
    """
    Append-only writer with per-year row buffers.

    Each row is one contextual observation identified by event_id.
    The three embeddings belong to that observation but may have
    different source windows, so window provenance is stored per scale.

    event_id is the only observation identity. There is deliberately no
    concept_id or vector_id: an observation is not a concept and there is
    no single vector identity for an ensemble of three vectors.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        dim: int,
        min_rows: int = DEFAULT_MIN_ROWS,
        min_bytes: int = DEFAULT_MIN_BYTES,
        compression: str = PARQUET_COMPRESSION,
        compression_level: int = PARQUET_COMPRESSION_LEVEL,
        row_group_size: int = PARQUET_ROW_GROUP_SIZE,
        data_page_size: int = PARQUET_DATA_PAGE_SIZE,
        **_kwargs: Any,
    ):
        self.root = Path(path)
        self.dim = int(dim)
        self.min_rows = max(1, int(min_rows))
        self.min_bytes = max(0, int(min_bytes))
        self.compression = compression
        self.compression_level = int(compression_level)
        self.row_group_size = max(1, int(row_group_size))
        self.data_page_size = max(64 * 1024, int(data_page_size))
        self._schema = observation_schema(self.dim)
        self._lock = threading.Lock()
        self.root.mkdir(parents=True, exist_ok=True)
        self._n_events: int | None = None
        self._con = _connect()
        self._buffers: dict[int, list[pa.Table]] = {}
        self._buffer_rows: dict[int, int] = {}
        self._buffer_bytes: dict[int, int] = {}
        self._closed = False

    def _estimate_bytes(self, n: int) -> int:
        emb = n * self.dim * 4 * 3
        meta = n * 96
        return emb + meta

    def append_events(
        self,
        *,
        event_id,
        corpus,
        doc_id,
        token,
        token_idx,
        pub_year,
        local_window_id,
        local_window_token_pos,
        medium_window_id,
        medium_window_token_pos,
        broad_window_id,
        broad_window_token_pos,
        emb_local,
        emb_medium,
        emb_broad,
    ) -> None:
        if self._closed:
            raise RuntimeError("ParquetObservationWriter is closed")

        event_id = np.asarray(event_id, dtype=np.int64)
        corpus = np.asarray(corpus).astype(str)
        doc_id = np.asarray(doc_id).astype(str)
        token = np.asarray(token).astype(str)
        token_idx = np.asarray(token_idx, dtype=np.int64)
        pub_year = np.asarray(pub_year, dtype=np.int16)

        local_window_id = np.asarray(local_window_id, dtype=np.int64)
        local_window_token_pos = np.asarray(
            local_window_token_pos, dtype=np.int32
        )
        medium_window_id = np.asarray(medium_window_id, dtype=np.int64)
        medium_window_token_pos = np.asarray(
            medium_window_token_pos, dtype=np.int32
        )
        broad_window_id = np.asarray(broad_window_id, dtype=np.int64)
        broad_window_token_pos = np.asarray(
            broad_window_token_pos, dtype=np.int32
        )

        emb_local = np.asarray(emb_local, dtype=np.float32)
        emb_medium = np.asarray(emb_medium, dtype=np.float32)
        emb_broad = np.asarray(emb_broad, dtype=np.float32)

        n = len(event_id)

        if n == 0:
            return

        metadata = (
            ("corpus", corpus),
            ("doc_id", doc_id),
            ("token", token),
            ("token_idx", token_idx),
            ("pub_year", pub_year),
            ("local_window_id", local_window_id),
            ("local_window_token_pos", local_window_token_pos),
            ("medium_window_id", medium_window_id),
            ("medium_window_token_pos", medium_window_token_pos),
            ("broad_window_id", broad_window_id),
            ("broad_window_token_pos", broad_window_token_pos),
        )

        for name, arr in metadata:
            if len(arr) != n:
                raise ValueError(
                    f"{name} length {len(arr)} != event_id length {n}"
                )

        for name, arr in (
            ("emb_local", emb_local),
            ("emb_medium", emb_medium),
            ("emb_broad", emb_broad),
        ):
            if arr.shape != (n, self.dim):
                raise ValueError(
                    f"{name} shape {arr.shape} != ({n}, {self.dim})"
                )

        years = np.unique(pub_year)

        with self._lock:
            for year in years:
                mask = pub_year == year

                table = self._build_table(
                    event_id[mask],
                    corpus[mask],
                    doc_id[mask],
                    token[mask],
                    token_idx[mask],
                    pub_year[mask],
                    local_window_id[mask],
                    local_window_token_pos[mask],
                    medium_window_id[mask],
                    medium_window_token_pos[mask],
                    broad_window_id[mask],
                    broad_window_token_pos[mask],
                    emb_local[mask],
                    emb_medium[mask],
                    emb_broad[mask],
                )

                y = int(year)
                nrows = table.num_rows

                self._buffers.setdefault(y, []).append(table)
                self._buffer_rows[y] = (
                    self._buffer_rows.get(y, 0) + nrows
                )
                self._buffer_bytes[y] = (
                    self._buffer_bytes.get(y, 0)
                    + self._estimate_bytes(nrows)
                )

                if (
                    self._buffer_rows[y] >= self.min_rows
                    or self._buffer_bytes[y] >= self.min_bytes
                ):
                    self._flush_year_unlocked(y)

            self._n_events = None

    def flush(self) -> None:
        with self._lock:
            for year in list(self._buffers):
                self._flush_year_unlocked(year)

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _flush_year_unlocked(self, year: int) -> None:
        tables = self._buffers.pop(year, None)
        self._buffer_rows.pop(year, None)
        self._buffer_bytes.pop(year, None)

        if not tables:
            return

        table = (
            pa.concat_tables(tables)
            if len(tables) > 1
            else tables[0]
        )

        self._write_part(year, table)
        self._n_events = None

    def _build_table(
        self,
        event_id,
        corpus,
        doc_id,
        token,
        token_idx,
        pub_year,
        local_window_id,
        local_window_token_pos,
        medium_window_id,
        medium_window_token_pos,
        broad_window_id,
        broad_window_token_pos,
        emb_local,
        emb_medium,
        emb_broad,
    ) -> pa.Table:
        def emb_col(mat: np.ndarray) -> pa.Array:
            flat = pa.array(
                mat.reshape(-1),
                type=pa.float32(),
            )
            return pa.FixedSizeListArray.from_arrays(
                flat,
                self.dim,
            )

        return pa.table(
            {
                "event_id": pa.array(event_id, type=pa.int64()),
                "corpus": pa.array(corpus, type=pa.string()),
                "doc_id": pa.array(doc_id, type=pa.string()),
                "token": pa.array(token, type=pa.string()),
                "token_idx": pa.array(token_idx, type=pa.int64()),
                "pub_year": pa.array(pub_year, type=pa.int16()),
                "local_window_id": pa.array(
                    local_window_id,
                    type=pa.int64(),
                ),
                "local_window_token_pos": pa.array(
                    local_window_token_pos,
                    type=pa.int32(),
                ),
                "medium_window_id": pa.array(
                    medium_window_id,
                    type=pa.int64(),
                ),
                "medium_window_token_pos": pa.array(
                    medium_window_token_pos,
                    type=pa.int32(),
                ),
                "broad_window_id": pa.array(
                    broad_window_id,
                    type=pa.int64(),
                ),
                "broad_window_token_pos": pa.array(
                    broad_window_token_pos,
                    type=pa.int32(),
                ),
                "emb_local": emb_col(emb_local),
                "emb_medium": emb_col(emb_medium),
                "emb_broad": emb_col(emb_broad),
            },
            schema=self._schema,
        )

    def _write_part(self, year: int, table: pa.Table) -> None:
        if table.num_rows == 0:
            return

        part_dir = self.root / f"year={year}"
        part_dir.mkdir(parents=True, exist_ok=True)

        part_name = f"part-{uuid.uuid4().hex[:12]}.parquet"
        dest = part_dir / part_name
        tmp = part_dir / f".{part_name}.tmp"

        write_observation_parquet(
            table,
            tmp,
            compression=self.compression,
            compression_level=self.compression_level,
            row_group_size=self.row_group_size,
            data_page_size=self.data_page_size,
        )

        tmp.rename(dest)

    def _count(self) -> int:
        if not _has_parquet(self.root):
            return 0

        glob = _glob(self.root)

        row = self._con.execute(
            f"""
            SELECT count(*)
            FROM read_parquet(
                '{glob}',
                hive_partitioning=true,
                union_by_name=true
            )
            """
        ).fetchone()

        return int(row[0]) if row else 0

    @property
    def n_events(self) -> int:
        if self._n_events is None:
            self._n_events = self._count()

        buffered = sum(self._buffer_rows.values())
        return self._n_events + buffered

    def get_doc_keys(self) -> set[tuple[str, str]]:
        if not _has_parquet(self.root):
            return set()

        glob = _glob(self.root)

        rows = self._con.execute(
            f"""
            SELECT DISTINCT corpus, doc_id
            FROM read_parquet(
                '{glob}',
                hive_partitioning=true,
                union_by_name=true
            )
            """
        ).fetchall()

        return {(str(c), str(d)) for c, d in rows}

    def get_event_ids(self) -> set[int]:
        if not _has_parquet(self.root):
            return set()

        glob = _glob(self.root)

        rows = self._con.execute(
            f"""
            SELECT event_id
            FROM read_parquet(
                '{glob}',
                hive_partitioning=true,
                union_by_name=true
            )
            """
        ).fetchall()

        return {int(r[0]) for r in rows}

    def embedding_dim(self) -> int:
        return self.dim

    def __len__(self) -> int:
        return self.n_events



# ---------------------------------------------------------------------------
# Stream
# ---------------------------------------------------------------------------

class ParquetObservationStream:
    """
    Batch streaming of multi-scale embeddings via DuckDB.

    Does not materialise the full corpus. year_filter is pushed into the
    SQL WHERE clause so unused year partitions are skipped when hive
    partitioning is present.
    """

    def __init__(self, root: str | Path, **_kwargs: Any):
        self.root = Path(root)
        self._con = _connect()
        self._dim: int | None = None

    def _ensure_dim(self) -> int:
        if self._dim is not None:
            return self._dim
        if not _has_parquet(self.root):
            self._dim = 0
            return 0
        # Infer dim from first non-empty emb_medium
        glob = _glob(self.root)
        row = self._con.execute(
            f"""
            SELECT emb_medium
            FROM read_parquet('{glob}', hive_partitioning=true, union_by_name=true)
            LIMIT 1
            """
        ).fetchone()
        if row is None or row[0] is None:
            self._dim = 0
        else:
            self._dim = len(row[0])
        return self._dim

    def iter_multi_scale_embeddings(
        self,
        batch_size: int = 8192,
        year_filter: Optional[set[int]] = None,
        year_manifest: Optional[Mapping[Any, np.ndarray]] = None,
    ) -> Iterator[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]:
        if not _has_parquet(self.root):
            return
            yield  # make this a generator  # noqa: unreachable

        dim = self._ensure_dim()
        glob = _glob(self.root)

        where = ""
        params: list[Any] = []
        if year_filter is not None and len(year_filter) > 0:
            # DuckDB IN list
            placeholders = ",".join("?" for _ in year_filter)
            # where = f"WHERE pub_year IN ({placeholders})"
            where = f"WHERE year IN ({placeholders})"
            params = list(year_filter)

        sql = f"""
            SELECT emb_local, emb_medium, emb_broad, event_id, pub_year
            FROM read_parquet('{glob}', hive_partitioning=true, union_by_name=true)
            {where}
        """
        # Use a fresh connection for the streaming cursor so concurrent
        # year_bounds / other queries on self._con stay safe.
        con = _connect()
        try:
            con.execute(sql, params)
            while True:
                rows = con.fetchmany(batch_size)
                if not rows:
                    break
                n = len(rows)
                emb_l = np.empty((n, dim), dtype=np.float32)
                emb_m = np.empty((n, dim), dtype=np.float32)
                emb_b = np.empty((n, dim), dtype=np.float32)
                eids = np.empty(n, dtype=np.int64)
                years = np.empty(n, dtype=np.int16)
                for i, (l, m, b, eid, y) in enumerate(rows):
                    emb_l[i] = l
                    emb_m[i] = m
                    emb_b[i] = b
                    eids[i] = eid
                    years[i] = y
                yield emb_l, emb_m, emb_b, eids, years
        finally:
            con.close()

    def year_bounds(self) -> tuple[int, int]:
        if not _has_parquet(self.root):
            raise RuntimeError("No parquet data found in store")
        glob = _glob(self.root)
        row = self._con.execute(
            f"""
            SELECT min(pub_year), max(pub_year)
            FROM read_parquet('{glob}', hive_partitioning=true, union_by_name=true)
            """
        ).fetchone()
        if row is None or row[0] is None:
            raise RuntimeError("No pub_year data found in any parquet file")
        return int(row[0]), int(row[1])




# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

class ParquetObservationLookup:
    """
    Selective observation access backed by Parquet + DuckDB.

    Metadata strategy
    -----------------
    On construction, metadata columns (no embeddings) are loaded into
    compact NumPy struct-of-arrays — the same memory trade-off as the
    current Zarr lookup after embeddings moved to FAISS. Peak RSS is
    dominated by metadata, not the ~16 GiB embedding matrices.

    Optional constructor kwargs:
        forms: iterable of tokens — only rows matching these tokens are
               loaded (case-insensitive). Use for single-concept runs.
        false_positives: tokens to exclude when forms is set.
        years: iterable of pub_year values to restrict the load.

    Embedding strategy
    ------------------
    1. If attach_index() was called, vectors are reconstructed from the
       attached per-year FAISS indices (identical path to Zarr).
    2. Otherwise vectors are fetched on demand from Parquet via DuckDB
       keyed by event_id (batched).

    This keeps the protocol surface identical while allowing a pure-Parquet
    path that does not require FAISS for embedding access.
    """

    _META_SQL_COLS = (
        "event_id",
        "corpus",
        "doc_id",
        "token",
        "token_idx",
        "pub_year",
        "local_window_id",
        "local_window_token_pos",
        "medium_window_id",
        "medium_window_token_pos",
        "broad_window_id",
        "broad_window_token_pos",
    )

    def __init__(
        self,
        root: str | Path,
        *,
        forms: Optional[Sequence[str]] = None,
        false_positives: Optional[Sequence[str]] = None,
        years: Optional[Sequence[int]] = None,
        **_kwargs: Any,
    ):
        self.root = Path(root)
        self._con = _connect()
        self._index = None  # optional FAISS attach
        self._dim: int | None = None
        self._emb_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._emb_cache_max = 50_000

        self._forms = {f.lower() for f in forms} if forms else None
        self._fps = {f.lower() for f in false_positives} if false_positives else set()
        self._years = set(int(y) for y in years) if years else None

        self._pos: dict[int, int] = {}
        self._pos_by_occurrence: dict[tuple[str, str, int], list[int]] = {}
        self._load_metadata()

    # --- loading -----------------------------------------------------------

    def _load_metadata(self) -> None:
        empty = {
            "event_id": np.empty(0, dtype=np.int64),
            "corpus": np.empty(0, dtype=object),
            "doc_id": np.empty(0, dtype=object),
            "token": np.empty(0, dtype=object),
            "token_idx": np.empty(0, dtype=np.int64),
            "pub_year": np.empty(0, dtype=np.int16),
            "local_window_id": np.empty(0, dtype=np.int64),
            "local_window_token_pos": np.empty(
                0,
                dtype=np.int32,
            ),
            "medium_window_id": np.empty(0, dtype=np.int64),
            "medium_window_token_pos": np.empty(
                0,
                dtype=np.int32,
            ),
            "broad_window_id": np.empty(0, dtype=np.int64),
            "broad_window_token_pos": np.empty(
                0,
                dtype=np.int32,
            ),
        }
        for k, v in empty.items():
            setattr(self, k, v)

        if not _has_parquet(self.root):
            return

        glob = _glob(self.root)
        clauses: list[str] = []
        params: list[Any] = []

        if self._forms is not None:
            placeholders = ",".join("?" for _ in self._forms)
            clauses.append(f"lower(token) IN ({placeholders})")
            params.extend(self._forms)
        if self._fps:
            placeholders = ",".join("?" for _ in self._fps)
            clauses.append(f"lower(token) NOT IN ({placeholders})")
            params.extend(self._fps)
        if self._years is not None:
            placeholders = ",".join("?" for _ in self._years)
            clauses.append(f"pub_year IN ({placeholders})")
            params.extend(self._years)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        cols = ", ".join(self._META_SQL_COLS)
        sql = f"""
            SELECT {cols}
            FROM read_parquet('{glob}', hive_partitioning=true, union_by_name=true)
            {where}
        """
        rel = self._con.execute(sql, params)
        # DuckDB may return a RecordBatchReader; materialise to a Table.
        arrow = rel.arrow()
        if hasattr(arrow, "read_all"):
            arrow = arrow.read_all()
        elif not isinstance(arrow, pa.Table):
            arrow = pa.Table.from_batches(list(arrow))

        n = arrow.num_rows
        if n == 0:
            return

        self.event_id = arrow.column("event_id").to_numpy().astype(
            np.int64,
            copy=False,
        )
        self.token_idx = arrow.column("token_idx").to_numpy().astype(
            np.int64,
            copy=False,
        )
        self.pub_year = arrow.column("pub_year").to_numpy().astype(
            np.int16,
            copy=False,
        )

        for name in (
            "local_window_id",
            "medium_window_id",
            "broad_window_id",
        ):
            setattr(
                self,
                name,
                arrow.column(name).to_numpy().astype(
                    np.int64,
                    copy=False,
                ),
            )

        for name in (
            "local_window_token_pos",
            "medium_window_token_pos",
            "broad_window_token_pos",
        ):
            setattr(
                self,
                name,
                arrow.column(name).to_numpy().astype(
                    np.int32,
                    copy=False,
                ),
            )

        self.corpus = np.asarray(
            arrow.column("corpus").to_pylist(),
            dtype=object,
        )
        self.doc_id = np.asarray(
            arrow.column("doc_id").to_pylist(),
            dtype=object,
        )
        self.token = np.asarray(
            arrow.column("token").to_pylist(),
            dtype=object,
        )

        self._pos = {int(eid): i for i, eid in enumerate(self.event_id.tolist())}
        self._build_position_index()

    def _build_position_index(self) -> None:
        self._pos_by_occurrence = {}
        for corpus, doc_id, token_idx, eid in zip(
            self.corpus.tolist(),
            self.doc_id.tolist(),
            self.token_idx.tolist(),
            self.event_id.tolist(),
        ):
            key = (str(corpus), str(doc_id), int(token_idx))
            bucket = self._pos_by_occurrence.get(key)
            if bucket is None:
                self._pos_by_occurrence[key] = [eid]
            else:
                bucket.append(eid)

    # --- size / schema -----------------------------------------------------

    def __len__(self) -> int:
        return len(self.event_id)

    @property
    def available_years(self) -> np.ndarray:
        if len(self.pub_year) == 0:
            return np.empty(0, dtype=np.int16)
        return np.unique(self.pub_year)

    # --- identity ----------------------------------------------------------

    def get_pos(self, event_id: int) -> int:
        return self._pos[int(event_id)]

    def _row_to_dict(self, pos: int) -> dict:
        def window_pos(value: int) -> int | None:
            return (
                None
                if value == NO_WINDOW_TOKEN_POS
                else int(value)
            )

        return {
            "event_id": int(self.event_id[pos]),
            "doc_id": str(self.doc_id[pos]),
            "corpus": str(self.corpus[pos]),
            "token": str(self.token[pos]),
            "token_idx": int(self.token_idx[pos]),
            "pub_year": int(self.pub_year[pos]),
            "local_window_id": int(self.local_window_id[pos]),
            "local_window_token_pos": window_pos(
                self.local_window_token_pos[pos]
            ),
            "medium_window_id": int(self.medium_window_id[pos]),
            "medium_window_token_pos": window_pos(
                self.medium_window_token_pos[pos]
            ),
            "broad_window_id": int(self.broad_window_id[pos]),
            "broad_window_token_pos": window_pos(
                self.broad_window_token_pos[pos]
            ),
        }


    def get_event_metadata(self, event_id: int) -> dict:
        return self._row_to_dict(self.get_pos(event_id))

    def get_event(self, event_id: int) -> dict:
        pos = self.get_pos(event_id)
        d = self._row_to_dict(pos)
        d["embedding"] = self.get_ensemble_embedding(pos)
        return d

    # --- form / position queries -------------------------------------------

    def iter_matching_event_ids(
        self,
        forms: Sequence[str],
        false_positives: Optional[Sequence[str]] = None,
    ) -> Iterator[int]:
        forms_set = {f.lower() for f in forms}
        fps = {f.lower() for f in (false_positives or [])}
        if len(self.token) == 0:
            return
        tokens_lower = np.char.lower(self.token.astype(str))
        mask = np.isin(tokens_lower, list(forms_set))
        if fps:
            mask &= ~np.isin(tokens_lower, list(fps))
        seen: set[int] = set()
        for eid in self.event_id[mask]:
            eid = int(eid)
            if eid in seen:
                continue
            seen.add(eid)
            yield eid

    def find_matching_event_ids(
        self,
        forms: Sequence[str],
        false_positives: Optional[Sequence[str]] = None,
    ) -> list[int]:
        return list(self.iter_matching_event_ids(forms, false_positives))

    def find_event_ids_by_positions(
        self,
        positions: Sequence[tuple[str, str, int]],
    ) -> dict[tuple[str, str, int], list[int]]:
        result: dict[tuple[str, str, int], list[int]] = {}
        for corpus, doc_id, token_idx in positions:
            key = (str(corpus), str(doc_id), int(token_idx))
            if key in result:
                continue
            result[key] = list(self._pos_by_occurrence.get(key, []))
        return result

    # --- embeddings --------------------------------------------------------

    def attach_index(self, index: Any) -> None:
        """
        Attach per-year FAISS indices for lazy reconstruction.

        Expected shape: dict[year][scale] -> object with
            .reconstruct(event_id) -> (dim,) ndarray
            .reconstruct_many(event_ids) -> (n, dim) ndarray
            .dim attribute
        Same contract as ZarrEventLookup.attach_index.
        """
        self._index = index
        if index:
            any_year = next(iter(index.values()))
            self._dim = next(iter(any_year.values())).dim

    def _ensure_dim(self) -> int:
        if self._dim is not None:
            return self._dim
        if not _has_parquet(self.root):
            self._dim = 0
            return 0
        glob = _glob(self.root)
        row = self._con.execute(
            f"""
            SELECT emb_medium
            FROM read_parquet('{glob}', hive_partitioning=true, union_by_name=true)
            LIMIT 1
            """
        ).fetchone()
        self._dim = len(row[0]) if row and row[0] is not None else 0
        return self._dim

    def _fetch_scales_from_parquet(
        self, event_ids: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (local, medium, broad) matrices aligned to event_ids order.
        Uses an in-process cache keyed by event_id.
        """
        dim = self._ensure_dim()
        n = len(event_ids)
        local = np.empty((n, dim), dtype=np.float32)
        medium = np.empty((n, dim), dtype=np.float32)
        broad = np.empty((n, dim), dtype=np.float32)

        missing: list[int] = []
        missing_idx: list[int] = []
        for i, eid in enumerate(event_ids):
            eid = int(eid)
            cached = self._emb_cache.get(eid)
            if cached is not None:
                local[i], medium[i], broad[i] = cached
            else:
                missing.append(eid)
                missing_idx.append(i)

        if missing:
            glob = _glob(self.root)
            # DuckDB list parameter
            placeholders = ",".join("?" for _ in missing)
            sql = f"""
                SELECT event_id, emb_local, emb_medium, emb_broad
                FROM read_parquet('{glob}', hive_partitioning=true, union_by_name=true)
                WHERE event_id IN ({placeholders})
            """
            rows = self._con.execute(sql, missing).fetchall()
            by_id = {int(r[0]): (r[1], r[2], r[3]) for r in rows}

            # Evict cache if oversized (simple FIFO-ish clear)
            if len(self._emb_cache) + len(by_id) > self._emb_cache_max:
                self._emb_cache.clear()

            for i, eid in zip(missing_idx, missing):
                trip = by_id.get(eid)
                if trip is None:
                    raise KeyError(f"event_id={eid} not found in parquet store")
                l, m, b = (
                    np.asarray(trip[0], dtype=np.float32),
                    np.asarray(trip[1], dtype=np.float32),
                    np.asarray(trip[2], dtype=np.float32),
                )
                local[i], medium[i], broad[i] = l, m, b
                self._emb_cache[eid] = (l, m, b)

        return local, medium, broad

    def _fetch_scales_from_faiss(
        self, event_ids: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self._index is not None
        dim = self._ensure_dim()
        n = len(event_ids)
        # Group by year for batched reconstruct_many
        positions = [self.get_pos(int(eid)) for eid in event_ids]
        years = self.pub_year[positions]

        local = np.empty((n, dim), dtype=np.float32)
        medium = np.empty((n, dim), dtype=np.float32)
        broad = np.empty((n, dim), dtype=np.float32)

        order = np.argsort(years, kind="stable")
        sorted_years = years[order]
        sorted_eids = np.asarray([int(event_ids[i]) for i in order], dtype=np.int64)

        start = 0
        n_ord = len(order)
        while start < n_ord:
            end = start + 1
            year = int(sorted_years[start])
            while end < n_ord and int(sorted_years[end]) == year:
                end += 1
            year_indices = self._index.get(year)
            if year_indices is None:
                raise KeyError(
                    f"No FAISS index for pub_year={year}. "
                    f"Available: {sorted(self._index.keys())}"
                )
            batch_ids = sorted_eids[start:end]
            local[order[start:end]] = year_indices["local"].reconstruct_many(batch_ids)
            medium[order[start:end]] = year_indices["medium"].reconstruct_many(batch_ids)
            broad[order[start:end]] = year_indices["broad"].reconstruct_many(batch_ids)
            start = end

        return local, medium, broad

    def _scales_for_ids(
        self, event_ids: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._index is not None:
            return self._fetch_scales_from_faiss(event_ids)
        return self._fetch_scales_from_parquet(event_ids)

    def get_ensemble_embedding(
        self,
        pos: int,
        weights: Sequence[float] = DEFAULT_ENSEMBLE_WEIGHTS,
    ) -> np.ndarray:
        eid = int(self.event_id[pos])
        local, medium, broad = self._scales_for_ids([eid])
        return (
            weights[0] * local[0]
            + weights[1] * medium[0]
            + weights[2] * broad[0]
        ).astype(np.float32)

    def get_embeddings(
        self,
        event_ids: Sequence[int],
        weights: Sequence[float] = DEFAULT_ENSEMBLE_WEIGHTS,
    ) -> np.ndarray:
        local, medium, broad = self._scales_for_ids(event_ids)
        return (
            weights[0] * local + weights[1] * medium + weights[2] * broad
        ).astype(np.float32)

    def get_concatenated_embeddings(
        self,
        event_ids: Sequence[int],
    ) -> np.ndarray:
        local, medium, broad = self._scales_for_ids(event_ids)

        def _norm_rows(M: np.ndarray) -> np.ndarray:
            norms = np.linalg.norm(M, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return M / norms

        return np.concatenate(
            [_norm_rows(local), _norm_rows(medium), _norm_rows(broad)],
            axis=1,
        ).astype(np.float32)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_backend(
    "parquet",
    writer=ParquetObservationWriter,
    stream=ParquetObservationStream,
    lookup=ParquetObservationLookup,
)

__all__ = [
    "write_observation_parquet",
    "ParquetObservationWriter",
    "ParquetObservationStream",
    "ParquetObservationLookup",
    "observation_schema",
]
