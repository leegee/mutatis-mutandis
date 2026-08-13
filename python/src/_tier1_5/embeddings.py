from __future__ import annotations

from typing import Iterator, Protocol

import numpy as np


EmbeddingBatch = tuple[np.ndarray, np.ndarray]
# (event_ids, vectors)


class EmbeddingStore(Protocol):
    """
    Canonical access to Tier 1 embeddings.

    Implementations must support bounded-memory streaming and batched
    point lookup. Neither operation may require materialising the corpus.
    """

    def stream(
        self,
        scale: str,
        batch_size: int,
    ) -> Iterator[EmbeddingBatch]:
        ...

    def get(
        self,
        event_ids: np.ndarray,
        scale: str,
    ) -> np.ndarray:
        ...
