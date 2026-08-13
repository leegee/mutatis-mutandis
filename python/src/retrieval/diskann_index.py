# retrieval/diskann_index.py

from pathlib import Path

import diskannpy
import numpy as np

from .mapping import ObservationIdMapping
from .models import Float32Array, SearchResult
from .observation_index import ObservationIndex


class DiskANNObservationIndex(ObservationIndex):
    """DiskANN-backed observation index."""

    def __init__(
        self,
        index_directory: str | Path,
        event_ids_path: str | Path,
        *,
        num_threads: int = 0,
        initial_search_complexity: int = 100,
        beam_width: int = 2,
        num_nodes_to_cache: int = 0,
        index_prefix: str = "local",
    ) -> None:
        self._index_directory = Path(index_directory)
        self._event_ids = np.load(event_ids_path, mmap_mode="r")
        self._beam_width = beam_width
        self._index = diskannpy.StaticDiskIndex(
            index_directory=str(self._index_directory),
            num_threads=num_threads,
            num_nodes_to_cache=num_nodes_to_cache,
            cache_mechanism=0,
            distance_metric="l2",
            vector_dtype=np.float32,
            dimensions=768,
            index_prefix=index_prefix,
        )

        self._search_complexity = initial_search_complexity


    def search(
        self,
        query: Float32Array,
        *,
        k: int,
    ) -> SearchResult:
        query = self._validate_queries(query, single=True)

        response = self._index.search(
            query_array,
            k_neighbors=k,
            complexity=self._search_complexity,
            beam_width=self._beam_width,
        )

        return SearchResult(
            event_ids=self._mapping.event_ids(response.identifiers),
            distances=response.distances,
        )


    def batch_search(
        self,
        queries: Float32Array,
        *,
        k: int,
    ) -> SearchResult:
        queries = self._validate_queries(queries, single=False)

        response = self._index.batch_search(
            query_array,
            k_neighbors=k,
            complexity=self._search_complexity,
            num_threads=num_threads,
            beam_width=self._beam_width,
        )

        return SearchResult(
            event_ids=self._mapping.event_ids(response.identifiers),
            distances=response.distances,
        )


    def _validate_queries(
        self,
        queries: Float32Array,
        *,
        single: bool,
    ) -> Float32Array:
        queries = np.asarray(queries)

        expected_ndim = 1 if single else 2

        if queries.ndim != expected_ndim:
            raise ValueError(
                f"expected {expected_ndim}-D query array, got {queries.ndim}-D"
            )

        if queries.dtype != np.float32:
            raise TypeError(
                f"queries must be float32, got {queries.dtype}"
            )

        if queries.shape[-1] != self._dimensions:
            raise ValueError(
                f"expected {self._dimensions} dimensions, got {queries.shape[-1]}"
            )

        if not np.isfinite(queries).all():
            raise ValueError("queries contain non-finite values")

        return queries
