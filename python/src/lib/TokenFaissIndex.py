import faiss
import numpy as np
from typing import Tuple, Protocol, cast

class _FaissIndexProto(Protocol):
    def add(self, x: np.ndarray) -> None: ...
    def search(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]: ...

class TokenFaissIndex:
    """
    FAISS wrapper for token-level embeddings (mean vectors per token).
    No IDs are stored; DB lookup maps result indices → token metadata.
    """
    def __init__(self, dim: int):
        # Inner FAISS index (cosine similarity via inner product on normalized vectors)
        self._index = cast(_FaissIndexProto, faiss.IndexFlatIP(dim))

    def add(self, vectors: np.ndarray) -> None:
        """
        Add vectors to the index. Each row is a vector.
        Vectors should be normalized (unit length) for cosine similarity.
        """
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._index.add(vectors)

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the index for k nearest neighbors per query vector.
        Returns:
            distances: shape (n_queries, k)
            indices: shape (n_queries, k)
        """
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        distances, indices = self._index.search(queries, k)
        return distances, indices

    def save(self, path: str) -> None:
        faiss.write_index(cast(faiss.Index, self._index), path)

    @classmethod
    def load(cls, path: str) -> "TokenFaissIndex":
        obj = cls.__new__(cls)
        obj._index = cast(_FaissIndexProto, faiss.read_index(path))
        return obj
