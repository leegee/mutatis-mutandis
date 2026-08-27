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
        oversample: int,
    ) -> BatchSearchResult:
        """
        Search each query independently and return query-major results.

        The installed LanceDB version does not reliably expose query_index
        in the result of a multi-vector search. Using the single-query path
        therefore keeps the ObservationIndex contract independent of that
        backend-specific result shape.

        Failure mode:
            A filtered search can return fewer than k neighbours. All
            queries in a batch must nevertheless produce the same width
            because BatchSearchResult is rectangular. The width is therefore
            the minimum result count returned by any query.
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

        search_k = k

        logger.info(
            "[lance batch_search] queries=%d k=%d oversample=%d "
            "year_start=%s year_end=%s",
            query_count,
            k,
            oversample,
            self._year_start,
            self._year_end,
        )

        results = []

        for query in query_array:
            result = self.search(
                query,
                k=search_k,
            )

            results.append(result)

        width = min(
            len(result.event_ids)
            for result in results
        )

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

        for query_index, result in enumerate(results):
            event_ids[query_index] = result.event_ids[:width]
            distances[query_index] = result.distances[:width]

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
