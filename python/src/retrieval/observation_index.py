from abc import ABC, abstractmethod
from typing import Literal, Iterator


from .models import (
    BatchSearchResult,
    Float32Array,
    SearchResult,
    SearchSpace
)


class ObservationIndex(ABC):
    """Backend-independent interface to observation nearest-neighbour search."""

    @abstractmethod
    def search(
        self,
        query: Float32Array,
        *,
        k: int,
    ) -> SearchResult:
        """Return the k nearest observations for one query."""
        raise NotImplementedError

    @abstractmethod
    def batch_search(
        self,
        queries: Float32Array,
        *,
        k: int,
        oversample: int,
        query_years: tuple[int, ...] | None = None,
    ) -> BatchSearchResult:
        """
        Return the k nearest observations for each query.

        query_years, when supplied, restricts each query to observations
        from that query's publication year. It is query metadata rather
        than a property of the global SearchSpace.
        """
        raise NotImplementedError

