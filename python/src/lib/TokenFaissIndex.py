import faiss
import numpy as np
from typing import Tuple, Protocol, cast

class _FaissIndexProto(Protocol):
    """
    Minimal protocol for FAISS index surface used by TokenFaissIndex.
    """
    def add(self, x: np.ndarray) -> None: ...
    def search(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]: ...

    @property
    def ntotal(self) -> int: ...


class TokenFaissIndex:
    """
    FAISS wrapper for token-level embeddings (mean vectors per token).
    No IDs are stored; DB lookup maps result indices → token metadata.
    """

    def __init__(self, dim: int):
        """
        Initialize a new FAISS index of given dimensionality.

        Args:
            dim: dimension of embedding vectors
        """
        # internal FAISS index implementing cosine similarity via inner product on normalized vectors.
        self._index = cast(_FaissIndexProto, faiss.IndexFlatIP(dim))

    @property
    def ntotal(self) -> int:
        """
        Number of vectors currently stored in the index.
        """
        return self._index.ntotal

    def add(self, vectors: np.ndarray) -> None:
        """
        Add vectors to the index. Each row is a vector.
        Vectors should be normalized (unit length) for cosine similarity.

        Args:
            vectors: np.ndarray of shape (n_vectors, dim)
        """
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._index.add(vectors)

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the index for k nearest neighbors per query vector.

        Args:
            queries: np.ndarray of shape (n_queries, dim)
            k: number of nearest neighbors to retrieve

        Returns:
            Tuple of (distances, indices)
            distances: shape (n_queries, k)
            indices: shape (n_queries, k)
        """
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        distances, indices = self._index.search(queries, k)
        return distances, indices

    def save(self, path: str) -> None:
        """
        Persist the FAISS index to disk.

        Args:
            path: file path to write the index
        """
        faiss.write_index(cast(faiss.Index, self._index), path)

    @classmethod
    def load(cls, path: str) -> "TokenFaissIndex":
        """
        Load a FAISS index from disk.

        Args:
            path: file path of a saved index

        Returns:
            TokenFaissIndex instance wrapping the loaded FAISS index
        """
        obj = cls.__new__(cls)
        obj._index = cast(_FaissIndexProto, faiss.read_index(path))
        return obj
