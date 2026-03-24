import faiss
import numpy as np
from typing import Optional, Sequence, Tuple, cast


class FaissIndex:
    def __init__(self, dim: int) -> None:
        self._index: faiss.Index = faiss.IndexFlatIP(dim)
        self._id_mode: bool = False

    def add(self, vectors: np.ndarray, ids: Optional[Sequence[int]] = None) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        if ids is not None:
            ids_arr = np.ascontiguousarray(ids, dtype=np.int64)
            if ids_arr.shape[0] != vectors.shape[0]:
                raise ValueError("ids must match number of vectors")

            if not self._id_mode:
                self._index = faiss.IndexIDMap(self._index)
                self._id_mode = True

            idx = cast(faiss.IndexIDMap, self._index)
            idx.add_with_ids(x=vectors, xids=ids_arr, n=vectors.shape[0])

        else:
            base = cast(faiss.Index, self._index)
            base.add(x=vectors, n=vectors.shape[0])

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        idx = cast(faiss.Index, self._index)

        distances = np.empty((queries.shape[0], k), dtype=np.float32)
        labels = np.empty((queries.shape[0], k), dtype=np.int64)
        idx.search(x=queries, k=k, n=k, distances=distances, labels=labels)
        return labels, distances

    def save(self, path: str) -> None:
        faiss.write_index(self._index, path)

    @classmethod
    def load(cls, path: str) -> "FaissIndex":
        obj = cls.__new__(cls)
        obj._index = faiss.read_index(path)
        obj._id_mode = isinstance(obj._index, faiss.IndexIDMap)
        return obj

