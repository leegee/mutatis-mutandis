from __future__ import annotations

import numpy as np

from lib.corpus_logging import logger
from .models import (
    BatchSearchResult,
    Float32Array,
    SearchResult,
)
from .observation_index import ObservationIndex


class LanceObservationIndex(ObservationIndex):
    """LanceDB-backed immutable index over one or more chronological buckets."""

    RECONSTRUCT_BATCH_SIZE = 500

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

    #
    def batch_search(
        self,
        queries: Float32Array,
        *,
        k: int,
        oversample: int = 1,
    ) -> BatchSearchResult:
        """
        Search each query independently and merge results across tables.

        The installed LanceDB version does not provide native multi-query
        result correlation, so each query is executed as a normal
        one-dimensional vector search. Queries are processed in chunks so
        that Python never holds the result dictionaries for the entire
        workset at once.

        Each query produces at most k * oversample candidates per physical
        table. Candidates are merged and reduced to the requested k results
        before moving on to the next query chunk.
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

        search_k = k * oversample

        logger.debug(
            "[lance batch_search] tables=%d queries=%d k=%d search_k=%d "
            "year_start=%s year_end=%s",
            len(self._tables),
            query_count,
            k,
            search_k,
            self._year_start,
            self._year_end,
        )

        # Keep the number of live Lance result sets bounded. This is
        # deliberately independent of Lance's native multi-query support.
        query_chunk_size = 32

        result_event_ids: list[np.ndarray] = []
        result_distances: list[np.ndarray] = []

        for chunk_start in range(0, query_count, query_chunk_size):
            chunk_end = min(
                chunk_start + query_chunk_size,
                query_count,
            )

            chunk_queries = query_array[chunk_start:chunk_end]
            chunk_count = chunk_queries.shape[0]

            chunk_event_ids = np.empty(
                (chunk_count, k),
                dtype=np.uint64,
            )
            chunk_distances = np.empty(
                (chunk_count, k),
                dtype=np.float32,
            )

            for local_query_index, single_query in enumerate(
                chunk_queries
            ):
                candidates: list[tuple[int, float]] = []

                for table in self._tables:
                    request = (
                        table
                        .search(
                            single_query,
                            vector_column_name="vector",
                        )
                        .nprobes(self._nprobes)
                        .limit(search_k)
                        .select(["event_id", "_distance"])
                    )

                    request = self._apply_filter(
                        request,
                        prefilter=True,
                    )

                    rows = request.to_list()

                    candidates.extend(
                        (
                            int(row["event_id"]),
                            float(row["_distance"]),
                        )
                        for row in rows
                    )

                    # The Lance result dictionaries are no longer needed
                    # after their scalar values have been extracted.

                candidates.sort(
                    key=lambda item: item[1],
                )

                selected = candidates[:k]

                if len(selected) != k:
                    raise RuntimeError(
                        "Lance returned insufficient neighbours: "
                        f"query_index={chunk_start + local_query_index} "
                        f"expected={k} "
                        f"got={len(selected)}"
                    )

                chunk_event_ids[local_query_index] = [
                    event_id
                    for event_id, _ in selected
                ]

                chunk_distances[local_query_index] = [
                    distance
                    for _, distance in selected
                ]

            result_event_ids.append(chunk_event_ids)
            result_distances.append(chunk_distances)

            logger.debug(
                "[lance batch_search] completed queries %d-%d/%d",
                chunk_start,
                chunk_end - 1,
                query_count,
            )

        return BatchSearchResult(
            event_ids=np.concatenate(
                result_event_ids,
                axis=0,
            ),
            distances=np.concatenate(
                result_distances,
                axis=0,
            ),
        )


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
