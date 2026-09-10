from __future__ import annotations

import os

import numpy as np

from lib.corpus_logging import logger
from .models import (
    BatchSearchResult,
    Float32Array,
    INVALID_EVENT_ID,
    SearchResult,
)
from .observation_index import ObservationIndex


# Whether to probe LanceDB's native multi-query batch dispatch at all.
#
# Confirmed broken on both lancedb 0.37.1 and 0.38.0 (schema error: no
# field named query_index), and this appears to be a longer-standing gap
# in the Python bindings specifically -- see lancedb/lancedb#1887
# ("Searching multiple vectors in one query will throw exception"), not
# something a routine version bump is likely to fix soon. Probing is
# therefore off by default so every process doesn't pay for one guaranteed
# -to-fail call before falling back.
#
# Set LANCE_PROBE_NATIVE_BATCH_SEARCH=1 to re-enable probing once a
# lancedb release is expected to support this, without needing a code
# change to find out.
_PROBE_NATIVE_BATCH_SEARCH = (
    os.environ.get(
        "LANCE_PROBE_NATIVE_BATCH_SEARCH",
        "0",
    )
    == "1"
)


class LanceObservationIndex(ObservationIndex):
    """LanceDB-backed immutable index over one or more chronological buckets."""

    RECONSTRUCT_BATCH_SIZE = 500

    # Bounds the number of live Lance result sets held in Python at once.
    # Independent of whether native multi-query dispatch is available.
    QUERY_CHUNK_SIZE = 256

    # Process-wide capability cache for native multi-query batch dispatch
    # (a single .search() call over multiple query vectors, correlated via
    # a query_index field on each result row).
    #
    # None  = probing enabled but not yet attempted this process
    # True  = native dispatch works, use it
    # False = native dispatch is not attempted/available; fall back to one
    #         .search() per query. This is the default -- see
    #         _PROBE_NATIVE_BATCH_SEARCH above for why.
    #
    # This is a class attribute (not per-instance) because the answer only
    # depends on the installed lancedb build, not on which table/bucket is
    # being searched, so resolving it once per process (when probing is
    # enabled at all) is sufficient.
    _native_batch_supported: bool | None = (
        None
        if _PROBE_NATIVE_BATCH_SEARCH
        else False
    )

    # Sentinel used to pad a batch_search() row when a query has fewer
    # than k genuine neighbours available (e.g. a seed in a sparse
    # chronological bucket). Padding is per-query: it never borrows width
    # from, or truncates, any other query in the same batch/chunk.
    #
    # This is retrieval.models.INVALID_EVENT_ID, the sentinel that
    # dataclass module already documents and types BatchSearchResult
    # (UInt64Array) around -- not an independently invented value. Do not
    # use -1 here: event_ids is unsigned, and -1 cannot be represented in
    # a uint64 array without either raising OverflowError on assignment
    # or silently wrapping to this exact same value.
    #
    # DISTANCE_SENTINEL (+inf) sorts last under the lower-is-better
    # convention used throughout this class, so a padded slot can never
    # displace a genuine candidate.
    EVENT_ID_SENTINEL = INVALID_EVENT_ID
    DISTANCE_SENTINEL = np.float32(np.inf)

    def __init__(
        self,
        table,
        *,
        dimensions: int = 768,
        year_start: int | None = None,
        year_end: int | None = None,
        model: str | None = None,
        nprobes: int = 20,
    ) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")

        if nprobes <= 0:
            raise ValueError("nprobes must be positive")

        if isinstance(table, (tuple, list)):
            tables = tuple(table)
        else:
            tables = (table,)

        if not tables:
            raise ValueError("at least one Lance table is required")

        self._tables = tables
        self._dimensions = dimensions
        self._year_start = year_start
        self._year_end = year_end
        self._model = model
        self._nprobes = nprobes

    def search(
        self,
        query: Float32Array,
        *,
        k: int,
    ) -> SearchResult:
        if k <= 0:
            raise ValueError("k must be positive")

        query_array = self._prepare_query(query)

        results = []

        for table in self._tables:
            request = (
                table
                .search(
                    query_array,
                    vector_column_name="vector",
                )
                .nprobes(self._nprobes)
                .limit(k)
                .select(["event_id", "_distance"])
            )

            request = self._apply_filter(
                request,
                prefilter=True,
            )

            rows = request.to_list()

            logger.debug(
                "[lance search] years=%s-%s model=%s table_candidates=%d k=%d",
                self._year_start,
                self._year_end,
                self._model,
                len(rows),
                k,
            )

            if rows:
                results.append(rows)

        if not results:
            return SearchResult(
                event_ids=np.empty(
                    0,
                    dtype=np.uint64,
                ),
                distances=np.empty(
                    0,
                    dtype=np.float32,
                ),
            )

        rows = [
            row
            for table_rows in results
            for row in table_rows
        ]

        converted = self._convert_rows(rows)

        if len(converted.event_ids) <= k:
            return converted

        # Lance distance is lower-is-better; negating it would select the worst
        # candidates when multiple tables are merged above.
        order = np.argsort(
            converted.distances,
            kind="stable",
        )[:k]

        return SearchResult(
            event_ids=converted.event_ids[order],
            distances=converted.distances[order],
        )

    def batch_search(
        self,
        queries: Float32Array,
        *,
        k: int,
        oversample: float | int = 1,
    ) -> BatchSearchResult:
        """
        Search all queries in each Lance table and merge results per query.

        Queries are processed in chunks so Python never holds the result
        dictionaries for the entire workset at once. Each query is issued
        as its own .search() call per table -- one round trip per
        (query, table) pair -- because LanceDB's native multi-query batch
        dispatch (a single .search() call over multiple query vectors,
        correlated via a query_index field on each result row) is not
        usable here: it fails with a schema error ("no field named
        query_index") on both lancedb 0.37.1 and 0.38.0, and the failure
        mode matches lancedb/lancedb#1887, a longer-standing Python
        bindings gap for multi-vector .search() calls rather than
        something a routine version bump resolves. See
        _PROBE_NATIVE_BATCH_SEARCH at module level: set
        LANCE_PROBE_NATIVE_BATCH_SEARCH=1 to re-enable a one-time
        per-process probe of the native path (see _native_batch_supported)
        once a lancedb release is expected to support it, without needing
        a code change to find out. Probing, when enabled, costs one cheap
        schema-validation failure per process rather than a wasted scan;
        it is off by default so that cost isn't paid on every run given
        the current, confirmed non-support.

        If a query has fewer than k genuine neighbours available (e.g. a
        seed in a sparse chronological bucket), its row is padded with
        EVENT_ID_SENTINEL (retrieval.models.INVALID_EVENT_ID) /
        DISTANCE_SENTINEL (+inf) entries rather than either truncating
        every other query in the same chunk to match it or raising.
        Padding is strictly per-query: other queries in the same batch
        always keep their own full, genuine results.

        INVALID_EVENT_ID is the sentinel retrieval.models already defines
        and documents for exactly this purpose, and BatchSearchResult
        types event_ids as UInt64Array to match it. Callers must compare
        against INVALID_EVENT_ID, not -1: the -1 checks currently in
        retrieval.lance_search.multiscale_search and
        reciprocal_rank_fusion predate this convention (or an unrelated
        one) and, since -1 cannot be represented in a uint64 array, have
        never matched a real padded row -- they need updating to check
        against INVALID_EVENT_ID for padding to actually be filtered.
        """
        if k <= 0:
            raise ValueError("k must be positive")

        if oversample <= 0:
            raise ValueError("oversample must be positive")

        query_array = self._prepare_queries(queries)
        query_count = query_array.shape[0]

        if query_count == 0:
            return BatchSearchResult(
                event_ids=np.empty(
                    (0, 0),
                    dtype=np.uint64,
                ),
                distances=np.empty(
                    (0, 0),
                    dtype=np.float32,
                ),
            )

        search_k = max(k, int(round(k * oversample)))

        logger.debug(
            "[lance batch_search] tables=%d queries=%d k=%d search_k=%d "
            "year_start=%s year_end=%s native_batch=%s",
            len(self._tables),
            query_count,
            k,
            search_k,
            self._year_start,
            self._year_end,
            type(self)._native_batch_supported,
        )

        result_event_ids = []
        result_distances = []

        for chunk_start in range(0, query_count, self.QUERY_CHUNK_SIZE):
            chunk_end = min(chunk_start + self.QUERY_CHUNK_SIZE, query_count)
            chunk_queries = query_array[chunk_start:chunk_end]

            # Now returns NumPy arrays directly
            chunk_ids, chunk_dists = self._search_chunk(
                chunk_queries, search_k=search_k
            )

            # Truncate / pad to exactly k (already padded with sentinels)
            result_event_ids.append(chunk_ids[:, :k])
            result_distances.append(chunk_dists[:, :k])

        return BatchSearchResult(
            event_ids=np.concatenate(result_event_ids, axis=0),
            distances=np.concatenate(result_distances, axis=0),
        )

    def _search_chunk(
        self,
        chunk_queries: Float32Array,
        *,
        search_k: int,
    ) -> list[list[tuple[int, float]]]:
        """
        Search one chunk of query vectors against every table, returning
        per-query candidate (event_id, distance) lists.

        Tries native multi-query dispatch first (see batch_search
        docstring) unless it is already known to be unsupported on this
        install; falls back to a per-query loop otherwise.
        """
        if type(self)._native_batch_supported is not False:
            try:
                return self._search_chunk_native(
                    chunk_queries,
                    search_k=search_k,
                )
            except Exception as exc:
                if "query_index" not in str(exc):
                    raise

                logger.warning(
                    "[lance batch_search] native multi-query dispatch "
                    "unsupported on this lancedb install (%s); falling "
                    "back to one .search() call per query. Upgrading "
                    "lancedb (>=0.38.0) restores native batching.",
                    exc,
                )

                type(self)._native_batch_supported = False

        return self._search_chunk_per_query(
            chunk_queries,
            search_k=search_k,
        )

    def _search_chunk_native(
        self,
        chunk_queries: Float32Array,
        *,
        search_k: int,
    ) -> list[list[tuple[int, float]]]:
        chunk_count = chunk_queries.shape[0]

        candidates_by_query: list[list[tuple[int, float]]] = [
            []
            for _ in range(chunk_count)
        ]

        for table in self._tables:
            request = (
                table
                .search(
                    chunk_queries,
                    vector_column_name="vector",
                )
                .nprobes(self._nprobes)
                .limit(search_k)
                .select(["event_id", "_distance", "query_index"])
            )

            request = self._apply_filter(request, prefilter=True)
            rows = request.to_list()

            for row in rows:
                query_index = int(row["query_index"])
                event_id = int(row["event_id"])
                distance = float(row["_distance"])

                if not 0 <= query_index < chunk_count:
                    raise RuntimeError(
                        f"Lance returned invalid query_index={query_index} "
                        f"for {chunk_count} queries in this chunk"
                    )

                candidates_by_query[query_index].append(
                    (event_id, distance)
                )

        type(self)._native_batch_supported = True

        return candidates_by_query


    def _search_chunk_per_query(
        self,
        chunk_queries: Float32Array,
        *,
        search_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns two arrays of shape (chunk_count, search_k):
            event_ids   uint64
            distances   float32
        Padded with EVENT_ID_SENTINEL / DISTANCE_SENTINEL when a query
        has fewer than search_k genuine hits.
        """
        chunk_count = chunk_queries.shape[0]

        # Pre-allocate the final result for the whole chunk
        all_event_ids = np.full(
            (chunk_count, search_k),
            self.EVENT_ID_SENTINEL,
            dtype=np.uint64,
        )
        all_distances = np.full(
            (chunk_count, search_k),
            self.DISTANCE_SENTINEL,
            dtype=np.float32,
        )

        for local_idx, query in enumerate(chunk_queries):
            # Collect candidates from every table that belongs to this index
            cand_ids = []
            cand_dists = []

            for table in self._tables:
                request = (
                    table
                    .search(query, vector_column_name="vector")
                    .nprobes(self._nprobes)
                    .limit(search_k)
                    .select(["event_id", "_distance"])
                )
                request = self._apply_filter(request, prefilter=True)

                # Prefer Arrow → NumPy instead of .to_list()
                # (lancedb ≥0.6 returns a pyarrow.Table from .to_arrow())
                arrow = request.to_arrow()
                if arrow.num_rows == 0:
                    continue

                ids = arrow.column("event_id").to_numpy(zero_copy_only=False).astype(np.uint64)
                dists = arrow.column("_distance").to_numpy(zero_copy_only=False).astype(np.float32)

                cand_ids.append(ids)
                cand_dists.append(dists)

            if not cand_ids:
                continue  # already filled with sentinels

            # Concatenate & keep the best search_k
            ids = np.concatenate(cand_ids)
            dists = np.concatenate(cand_dists)

            if len(ids) > search_k:
                order = np.argpartition(dists, search_k)[:search_k]
                # stable sort of the selected slice
                order = order[np.argsort(dists[order], kind="stable")]
                ids = ids[order]
                dists = dists[order]
            else:
                order = np.argsort(dists, kind="stable")
                ids = ids[order]
                dists = dists[order]

            n = len(ids)
            all_event_ids[local_idx, :n] = ids
            all_distances[local_idx, :n] = dists

        return all_event_ids, all_distances


    def reconstruct(
        self,
        event_id: int,
    ) -> Float32Array:
        """
        Retrieve the stored normalised vector for one event.

        The lookup is by stable semantic event ID, never by Lance row
        position. Tier 1 remains authoritative for the observation's
        metadata and provenance.
        """
        event_id = int(event_id)

        for table in self._tables:
            rows = (
                table
                .search()
                .where(
                    self._event_id_filter(event_id),
                    prefilter=True,
                )
                .select(["event_id", "vector"])
                .limit(1)
                .to_list()
            )

            if not rows:
                continue

            row = rows[0]

            if int(row["event_id"]) != event_id:
                raise RuntimeError(
                    f"Lance returned unexpected event_id="
                    f"{row['event_id']} for requested {event_id}"
                )

            return self._validate_vector(
                row["vector"],
                event_id,
            )

        raise KeyError(
            f"Lance index does not contain event_id={event_id}"
        )

    def reconstruct_many(
        self,
        event_ids,
    ) -> np.ndarray:
        """
        Retrieve stored vectors aligned with event_ids.

        Queries are chunked because an arbitrarily large OR expression is
        neither a useful nor a predictable bulk-retrieval mechanism.

        Duplicate requested IDs are permitted and are returned repeatedly
        in their original positions, matching FAISS reconstruct_many().
        """
        requested = [
            int(event_id)
            for event_id in event_ids
        ]

        if not requested:
            return np.empty(
                (0, self._dimensions),
                dtype=np.float32,
            )

        vectors = {}

        for start in range(
            0,
            len(requested),
            self.RECONSTRUCT_BATCH_SIZE,
        ):
            chunk = requested[
                start:start + self.RECONSTRUCT_BATCH_SIZE
            ]

            unique_ids = set(chunk)

            remaining_ids = set(unique_ids)

            for table in self._tables:
                if not remaining_ids:
                    break

                conditions = [
                    self._event_id_filter(event_id)
                    for event_id in remaining_ids
                ]

                rows = (
                    table
                    .search()
                    .where(
                        " OR ".join(conditions),
                        prefilter=True,
                    )
                    .select(["event_id", "vector"])
                    .limit(len(remaining_ids))
                    .to_list()
                )

                for row in rows:
                    event_id = int(row["event_id"])

                    if event_id not in unique_ids:
                        raise RuntimeError(
                            f"Lance returned unexpected event_id="
                            f"{event_id}"
                        )

                    vectors[event_id] = self._validate_vector(
                        row["vector"],
                        event_id,
                    )

                remaining_ids = unique_ids.difference(vectors)

            if remaining_ids:
                raise KeyError(
                    f"Lance index missing event_ids="
                    f"{sorted(remaining_ids)[:10]}"
                )

        return np.asarray(
            [
                vectors[event_id]
                for event_id in requested
            ],
            dtype=np.float32,
        )

    def _apply_filter(
        self,
        request,
        *,
        prefilter: bool,
    ):
        conditions = []

        if self._year_start is not None:
            conditions.append(
                f"year >= {int(self._year_start)}"
            )

        if self._year_end is not None:
            conditions.append(
                f"year <= {int(self._year_end)}"
            )

        if self._model is not None:
            escaped_model = self._model.replace(
                "'",
                "''",
            )
            conditions.append(
                f"embedding_model = '{escaped_model}'"
            )

        if conditions:
            request = request.where(
                " AND ".join(conditions),
                prefilter=prefilter,
            )

        return request

    @staticmethod
    def _event_id_filter(
        event_id: int,
    ) -> str:
        return f"event_id = {int(event_id)}"

    def _prepare_query(
        self,
        query: Float32Array,
    ) -> Float32Array:
        query_array = np.asarray(
            query,
            dtype=np.float32,
        )

        if query_array.ndim != 1:
            raise ValueError(
                "query must be one-dimensional"
            )

        if query_array.shape[0] != self._dimensions:
            raise ValueError(
                f"query dimension {query_array.shape[0]} "
                f"does not match index dimension "
                f"{self._dimensions}"
            )

        return self._normalise_query(query_array)

    def _prepare_queries(
        self,
        queries: Float32Array,
    ) -> Float32Array:
        query_array = np.asarray(
            queries,
            dtype=np.float32,
        )

        if query_array.ndim != 2:
            raise ValueError(
                "queries must be two-dimensional"
            )

        if query_array.shape[1] != self._dimensions:
            raise ValueError(
                f"query dimension {query_array.shape[1]} "
                f"does not match index dimension "
                f"{self._dimensions}"
            )

        return self._normalise_queries(query_array)

    @staticmethod
    def _normalise_query(
        query: Float32Array,
    ) -> Float32Array:
        norm = np.linalg.norm(query)

        if not np.isfinite(norm) or norm == 0:
            raise ValueError(
                "query vector must have a finite, non-zero norm"
            )

        return query / norm

    @staticmethod
    def _normalise_queries(
        queries: Float32Array,
    ) -> Float32Array:
        norms = np.linalg.norm(
            queries,
            axis=1,
            keepdims=True,
        )

        if np.any(~np.isfinite(norms)) or np.any(norms == 0):
            raise ValueError(
                "query vectors must have finite, non-zero norms"
            )

        return queries / norms

    def _validate_vector(
        self,
        vector,
        event_id: int,
    ) -> Float32Array:
        vector = np.asarray(
            vector,
            dtype=np.float32,
        )

        if vector.shape != (self._dimensions,):
            raise ValueError(
                f"Invalid reconstructed vector shape for "
                f"event_id={event_id}: {vector.shape}"
            )

        if not np.isfinite(vector).all():
            raise ValueError(
                f"Invalid reconstructed vector for event_id={event_id}"
            )

        return vector

    @staticmethod
    def _convert_rows(
        rows: list[dict],
    ) -> SearchResult:
        if not rows:
            return SearchResult(
                event_ids=np.empty(
                    0,
                    dtype=np.uint64,
                ),
                distances=np.empty(
                    0,
                    dtype=np.float32,
                ),
            )

        distances = np.asarray(
            [
                row["_distance"]
                if "_distance" in row
                else row["distance"]
                for row in rows
            ],
            dtype=np.float32,
        )

        event_ids = np.asarray(
            [row["event_id"] for row in rows],
            dtype=np.uint64,
        )

        order = np.argsort(
            distances,
            kind="stable",
        )

        return SearchResult(
            event_ids=event_ids[order],
            distances=distances[order],
        )