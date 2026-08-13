# retrieval/diskann_observation_index.py

from pathlib import Path

import diskannpy
import numpy as np

from .mapping import ObservationIdMapping
from .models import Float32Array, SearchResult, UInt64Array
from .observation_index import ObservationIndex


class DiskANNObservationIndex(ObservationIndex):
    """DiskANN-backed immutable index over observation embeddings."""

    def __init__(
        self,
        index_directory: str | Path,
        event_ids_path: str | Path,
        *,
        dimensions: int,
        num_threads: int = 0,
        search_complexity: int = 100,
        beam_width: int = 2,
        batch_num_threads: int = 0,
        num_nodes_to_cache: int = 0,
        index_prefix: str = "local",
    ) -> None:
        self._index_directory = Path(index_directory)
        self._event_ids = ObservationIdMapping(event_ids_path)

        self._dimensions = dimensions
        self._num_threads = num_threads
        self._batch_num_threads = batch_num_threads
        self._search_complexity = search_complexity
        self._beam_width = beam_width

        if dimensions <= 0:
            raise ValueError("dimensions must be positive")

        if search_complexity <= 0:
            raise ValueError("search_complexity must be positive")

        if beam_width <= 0:
            raise ValueError("beam_width must be positive")

        if num_threads < 0:
            raise ValueError("num_threads must be non-negative")

        if batch_num_threads < 0:
            raise ValueError("batch_num_threads must be non-negative")

        if num_nodes_to_cache < 0:
            raise ValueError("num_nodes_to_cache must be non-negative")

        self._index = diskannpy.StaticDiskIndex(
            index_directory=str(self._index_directory),
            num_threads=self._num_threads,
            num_nodes_to_cache=num_nodes_to_cache,
            cache_mechanism=0,
            distance_metric="l2",
            vector_dtype=np.float32,
            dimensions=self._dimensions,
            index_prefix=index_prefix,
        )

    def search(
        self,
        query: Float32Array,
        *,
        k: int,
    ) -> SearchResult:
        query_array = self._prepare_query(query)

        response = self._index.search(
            query_array,
            k_neighbors=k,
            complexity=self._search_complexity,
            beam_width=self._beam_width,
        )

        return self._convert_response(response)

    def batch_search(
        self,
        queries: Float32Array,
        *,
        k: int,
    ) -> SearchResult:
        query_array = self._prepare_queries(queries)

        response = self._index.batch_search(
            query_array,
            k_neighbors=k,
            complexity=self._search_complexity,
            num_threads=self._batch_num_threads,
            beam_width=self._beam_width,
        )

        return self._convert_batch_response(response)

    def _prepare_query(
        self,
        query: Float32Array,
    ) -> Float32Array:
        query_array = np.asarray(query, dtype=np.float32)

        if query_array.ndim != 1:
            raise ValueError("query must be one-dimensional")

        if query_array.shape[0] != self._dimensions:
            raise ValueError(
                f"query dimension {query_array.shape[0]} "
                f"does not match index dimension {self._dimensions}"
            )

        return self._normalise_query(query_array)

    def _prepare_queries(
        self,
        queries: Float32Array,
    ) -> Float32Array:
        query_array = np.asarray(queries, dtype=np.float32)

        if query_array.ndim != 2:
            raise ValueError("queries must be two-dimensional")

        if query_array.shape[1] != self._dimensions:
            raise ValueError(
                f"query dimension {query_array.shape[1]} "
                f"does not match index dimension {self._dimensions}"
            )

        return self._normalise_queries(query_array)

    def _normalise_query(
        self,
        query: Float32Array,
    ) -> Float32Array:
        norm = np.linalg.norm(query)

        if not np.isfinite(norm) or norm == 0:
            raise ValueError("query vector must have a finite, non-zero norm")

        return query / norm

    def _normalise_queries(
        self,
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

    def _convert_response(
        self,
        response: diskannpy.QueryResponse,
    ) -> SearchResult:
        local_ids = np.asarray(
            response.identifiers,
            dtype=np.int64,
        )

        return SearchResult(
            event_ids=self._map_local_ids(local_ids),
            distances=np.asarray(
                response.distances,
                dtype=np.float32,
            ),
        )

    def _convert_batch_response(
        self,
        response: diskannpy.QueryResponseBatch,
    ) -> SearchResult:
        local_ids = np.asarray(
            response.identifiers,
            dtype=np.int64,
        )

        return SearchResult(
            event_ids=self._map_local_ids(local_ids),
            distances=np.asarray(
                response.distances,
                dtype=np.float32,
            ),
        )

    def _map_local_ids(
        self,
        local_ids: np.ndarray,
    ) -> UInt64Array:
        return self._event_ids.event_ids(local_ids)
