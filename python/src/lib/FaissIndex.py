import faiss
import numpy as np
from typing import Optional, Sequence, Tuple, cast, Protocol


class _FaissIndexProto(Protocol):
    def add(self, x: np.ndarray) -> None: ...
    def add_with_ids(self, x: np.ndarray, xids: np.ndarray) -> None: ...
    def search(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]: ...


class FaissIndex:
    def __init__(self, dim: int) -> None:
        self._index = cast(_FaissIndexProto, faiss.IndexFlatIP(dim))
        self._id_mode: bool = False

    def add(self, vectors: np.ndarray, ids: Optional[Sequence[int]] = None) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        if ids is not None:
            ids_arr = np.ascontiguousarray(ids, dtype=np.int64)
            if ids_arr.shape[0] != vectors.shape[0]:
                raise ValueError("ids must match number of vectors")

            if not self._id_mode:
                self._index = cast(
                    _FaissIndexProto,
                    faiss.IndexIDMap(cast(faiss.Index, self._index)),
                )
                self._id_mode = True

            self._index.add_with_ids(vectors, ids_arr)

        else:
            self._index.add(vectors)

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        labels, distances = self._index.search(queries, k)
        return labels, distances

    def save(self, path: str) -> None:
        faiss.write_index(cast(faiss.Index, self._index), path)

    @classmethod
    def load(cls, path: str) -> "FaissIndex":
        obj = cls.__new__(cls)
        obj._index = cast(_FaissIndexProto, faiss.read_index(path))
        obj._id_mode = isinstance(obj._index, faiss.IndexIDMap)
        return obj
