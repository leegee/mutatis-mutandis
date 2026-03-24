import faiss
import numpy as np
from typing import Optional, Sequence, Tuple, cast


class FaissIndex:
    """
    Typed façade over FAISS.

    Invariants:
    - Always stores float32, contiguous vectors
    - If IDs are used once, index is permanently ID-mapped
    - Caller never interacts with raw faiss.Index
    """

    def __init__(self, dim: int) -> None:
        base = faiss.IndexFlatIP(dim)
        self._index: faiss.Index = base
        self._id_mode = False

    def add(self, vectors: np.ndarray, ids: Optional[Sequence[int]] = None) -> None:
        if ids is None and self._id_mode:
            raise ValueError("Cannot add vectors without IDs after ID mode is enabled")

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        if ids is not None:
            ids_arr = np.asarray(ids, dtype=np.int64)
            if ids_arr.shape[0] != vectors.shape[0]:
                raise ValueError("ids must match number of vectors")

            if not self._id_mode:
                # one-way transition
                self._index = faiss.IndexIDMap(self._index)
                self._id_mode = True

            idx = cast(faiss.IndexIDMap, self._index)
            idx.add_with_ids(vectors, ids_arr)  # type: ignore

        else:
            base = cast(faiss.Index, self._index)
            base.add(vectors)  # type: ignore

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        idx = cast(faiss.Index, self._index)
        return idx.search(queries, k)  # type: ignore

    def save(self, path: str) -> None:
        faiss.write_index(self._index, path)

    @classmethod
    def load(cls, path: str) -> "FaissIndex":
        obj = cls.__new__(cls)
        obj._index = faiss.read_index(path)
        obj._id_mode = isinstance(obj._index, faiss.IndexIDMap)
        return obj
