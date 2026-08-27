from __future__ import annotations

import numpy as np

from .models import Float32Array, SearchResult, SearchSpace
from .observation_index_store import ObservationIndexStore
from .parquet_context import ObservationContext, ParquetContext
from .retriever import ObservationRetriever


class IndexedObservationRetriever(ObservationRetriever):
    """Resolve searches in selected observation spaces into corpus context."""

    def __init__(
        self,
        index_store: ObservationIndexStore,
        context: ParquetContext,
    ) -> None:
        self._index_store = index_store
        self._context = context


    def search(
        self,
        query: Float32Array,
        *,
        space: SearchSpace,
        k: int,
    ) -> list[ObservationContext]:
        if k <= 0:
            raise ValueError("k must be positive")

        indexes = self._index_store.get(space)

        if not indexes:
            return []

        results = [
            index.search(
                query,
                k=k,
            )
            for index in indexes
        ]

        event_ids = np.concatenate(
            [result.event_ids for result in results],
        )
        distances = np.concatenate(
            [result.distances for result in results],
        )

        if len(event_ids) > k:
            selected = np.argpartition(
                distances,
                k - 1,
            )[:k]

            selected = selected[
                np.argsort(distances[selected])
            ]
        else:
            selected = np.argsort(distances)

        merged = SearchResult(
            event_ids=event_ids[selected],
            distances=distances[selected],
        )

        return self._context.get_many(
            merged,
        )


    def batch_search(
        self,
        queries: Float32Array,
        *,
        space: SearchSpace,
        k: int,
        oversample: int,
    ) -> list[list[ObservationContext]]:
        if k <= 0:
            raise ValueError("k must be positive")

        indexes = self._index_store.get(space)

        if not indexes:
            return [
                []
                for _ in range(len(queries))
            ]

        batch_results = [
            index.batch_search(
                queries,
                k=k,
                oversample=oversample,
            )
            for index in indexes
        ]

        merged_results: list[SearchResult] = []

        for query_idx in range(len(queries)):
            event_ids = np.concatenate(
                [
                    results[query_idx].event_ids
                    for results in batch_results
                ],
            )

            distances = np.concatenate(
                [
                    results[query_idx].distances
                    for results in batch_results
                ],
            )

            if len(event_ids) > k:
                selected = np.argpartition(
                    distances,
                    k - 1,
                )[:k]

                selected = selected[
                    np.argsort(distances[selected])
                ]
            else:
                selected = np.argsort(distances)

            merged_results.append(
                SearchResult(
                    event_ids=event_ids[selected],
                    distances=distances[selected],
                )
            )

        return [
            self._context.get_many(result)
            for result in merged_results
        ]
