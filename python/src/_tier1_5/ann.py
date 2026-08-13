from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class RankedResults:
    """
    ANN candidates for a batch of queries.

    Row i of event_ids and scores describes the candidates returned for
    query i. Candidates within each row are ordered best-first.

    event_id is the stable semantic observation identifier. It is not an
    ANN-local row number and must remain meaningful outside the index.
    """

    event_ids: np.ndarray
    scores: np.ndarray

    def __post_init__(self) -> None:
        if self.event_ids.ndim != 2:
            raise ValueError(
                "event_ids must have shape (n_queries, k)"
            )

        if self.scores.ndim != 2:
            raise ValueError(
                "scores must have shape (n_queries, k)"
            )

        if self.event_ids.shape != self.scores.shape:
            raise ValueError(
                "event_ids and scores must have identical shape"
            )

        if self.event_ids.dtype.kind not in "iu":
            raise TypeError(
                "event_ids must contain integer event IDs"
            )


class ANNIndex(Protocol):
    """
    Geometric retrieval over stable semantic event IDs.

    The ANN index is a derived retrieval structure. It does not own
    provenance, metadata, semantic interpretation, or the canonical
    embedding archive.
    """

    @property
    def dimension(self) -> int:
        """Embedding dimensionality."""
        ...

    @property
    def size(self) -> int:
        """Number of indexed observations."""
        ...

    def add(
        self,
        event_ids: np.ndarray,
        vectors: np.ndarray,
    ) -> None:
        """
        Add embeddings associated with stable event IDs.

        Implementations must reject dimension mismatches and invalid
        event IDs rather than silently altering the semantic identity
        of observations.
        """
        ...

    def search(
        self,
        queries: np.ndarray,
        k: int,
    ) -> RankedResults:
        """
        Return the k nearest indexed observations for each query.

        Results must be ordered best-first within each query.
        """
        ...

    def reconstruct_many(
        self,
        event_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Return the indexed embedding for each stable event ID.

        Implementations may obtain these vectors from their own index
        representation rather than the canonical embedding archive.
        """
        ...

    def save(self, path: Path) -> None:
        """Persist the derived ANN index."""
        ...

    @classmethod
    def load(cls, path: Path) -> "ANNIndex":
        """Load a previously persisted ANN index."""
        ...
