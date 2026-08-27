"""
retrieval/lance_observation_index.py
"""

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
    """LanceDB-backed immutable index over observation embeddings."""

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

        self._table = table
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

        request = (
            self._table
            .search(query_array)
            # LanceDB documentation recommends automatic nprobes tuning.
            # .nprobes(self._nprobes)
            .limit(k)
        )

        request = self._apply_filter(request)

        rows = request.to_list()

        return self._convert_rows(rows)

    def batch_search(
        self,
        queries: Float32Array,
        *,
        k: int,
        oversample = 1
    ) -> BatchSearchResult:
        """
        Use LanceDB's native batched vector search.

        The ObservationIndex contract requires results to be returned as
        query-major arrays. LanceDB returns a flat row collection containing
        a query_index for each result, so the rows are regrouped here.

        Failure mode:
            A filtered search can return fewer than k neighbours for an
            individual query. The returned arrays therefore use the actual
            common result width rather than fabricating invalid event IDs or
            distances.
        """
        if k <= 0:
            raise ValueError("k must be positive")

        query_array = self._prepare_queries(queries)
        query_count = query_array.shape[0]

        logger.info(f"[lance batch_search] k={k} oversample={oversample}")

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

        request = (
            self._table
            .search(query_array)
            # LanceDB documentation recommends automatic nprobes tuning.
            # .nprobes(self._nprobes)
            .limit(k * oversample)
        )

        request = self._apply_filter(request)

        rows = request.to_list()

        if not rows:
            return BatchSearchResult(
                event_ids=np.empty(
                    (query_count, 0),
                    dtype=np.uint64,
                ),
                distances=np.empty(
                    (query_count, 0),
                    dtype=np.float32,
                ),
            )

        results_by_query = [
            []
            for _ in range(query_count)
        ]

        for row in rows:
            query_index = row.get("query_index")

            if query_index is None:
                raise RuntimeError( "Lance batch search result is missing query_index" )

            query_index = int(query_index)

            if query_index < 0 or query_index >= query_count:
                raise RuntimeError( f"Lance batch search returned invalid query_index={query_index}" )

            distance = row.get( "_distance", row.get("distance") )

            if distance is None:
                raise RuntimeError( "Lance batch search result is missing distance" )

            results_by_query[query_index].append(
                (
                    int(row["event_id"]),
                    float(distance),
                )
            )

        widths = {
            len(results)
            for results in results_by_query
        }

        if len(widths) != 1:
            raise RuntimeError( "Lance batch search returned inconsistent result widths across queries" )

        width = widths.pop()

        if width == 0:
            return BatchSearchResult(
                event_ids=np.empty(
                    (query_count, 0),
                    dtype=np.uint64,
                ),
                distances=np.empty(
                    (query_count, 0),
                    dtype=np.float32,
                ),
            )

        event_ids = np.empty(
            (query_count, width),
            dtype=np.uint64,
        )

        distances = np.empty(
            (query_count, width),
            dtype=np.float32,
        )

        for query_index, results in enumerate( results_by_query ):
            event_ids[query_index] = [
                event_id
                for event_id, _ in results
            ]

            distances[query_index] = [
                distance
                for _, distance in results
            ]

        return BatchSearchResult(
            event_ids=event_ids,
            distances=distances,
        )


    def _apply_filter(
        self,
        request,
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
                " AND ".join(conditions)
            )

        return request

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

        return self._normalise_query(
            query_array
        )

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

        return self._normalise_queries(
            query_array
        )

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

        return SearchResult(
            event_ids=np.asarray(
                [
                    row["event_id"]
                    for row in rows
                ],
                dtype=np.uint64,
            ),
            distances=np.asarray(
                [
                    row.get(
                        "_distance",
                        row.get("distance"),
                    )
                    for row in rows
                ],
                dtype=np.float32,
            ),
        )
