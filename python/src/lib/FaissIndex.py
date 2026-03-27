from pathlib import Path
import faiss
import numpy as np
from typing import Sequence, Tuple, cast, Protocol


class _FaissIndexProto(Protocol):
    def add_with_ids(self, x: np.ndarray, xids: np.ndarray) -> None: ...
    def search(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]: ...


class FaissIndex:
    """
    FAISS wrapper that ALWAYS uses explicit numeric IDs.

    IDs must correspond to `pamphlet_tokens.token_occurrence_id`.

    There is no non-ID mode: this enforces consistency across the pipeline.
    """

    def __init__(self, dim: int):
        base_index = faiss.IndexFlatIP(dim)
        self._index = cast(
            _FaissIndexProto,
            faiss.IndexIDMap(base_index)
        )

    def add(self, vectors: np.ndarray, ids: Sequence[int]) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        ids_arr = np.ascontiguousarray(ids, dtype=np.int64)

        if vectors.shape[0] != ids_arr.shape[0]:
            raise ValueError("ids must match number of vectors")

        self._index.add_with_ids(vectors, ids_arr)

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        distances, indices = self._index.search(queries, k)
        return distances, indices

    def save(self, path: str) -> None:
        faiss.write_index(cast(faiss.Index, self._index), path)

    @classmethod
    def load(cls, path: str) -> "FaissIndex":
        if not Path(path).is_file():
            raise FileNotFoundError(f"Index file not found: {path}")

        obj = cls.__new__(cls)
        obj._index = cast(_FaissIndexProto, faiss.read_index(path))

        # Hard assertion: must be IDMap
        if not isinstance(obj._index, faiss.IndexIDMap):
            raise TypeError("Loaded FAISS index is not IndexIDMap (ID mode required)")

        return obj
