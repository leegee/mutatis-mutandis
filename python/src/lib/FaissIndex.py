from pathlib import Path
import faiss
import numpy as np
from typing import Sequence, Tuple, cast, Protocol
from lib.eebo_logging import logger


class _FaissIndexProto(Protocol):
    def add_with_ids(self, x: np.ndarray, xids: np.ndarray) -> None: ...
    def search(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]: ...


class FaissIndex:
    """
    FAISS wrapper that ALWAYS uses explicit numeric IDs.

    IDs must correspond to `pamphlet_tokens.token_occurrence_id`.

    invariant: vectors are unit-normalized so inner product == cosine similarity
    """

    def __init__(self, dim: int):
        base_index = faiss.IndexFlatIP(dim)
        self._index = cast(
            _FaissIndexProto,
            faiss.IndexIDMap(base_index)
        )

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        # invariant: no zero vectors enter the index
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Zero vector encountered during normalization")
        return x / norms

    def add(self, vectors: np.ndarray, ids: Sequence[int]) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        ids_arr = np.ascontiguousarray(ids, dtype=np.int64)

        if vectors.shape[0] != ids_arr.shape[0]:
            raise ValueError("ids must match number of vectors")

        # enforce cosine/IP equivalence
        vectors = self._normalize(vectors)

        self._index.add_with_ids(vectors, ids_arr)

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        queries = np.ascontiguousarray(queries, dtype=np.float32)

        # enforce same geometry at query time
        queries = self._normalize(queries)

        distances, indices = self._index.search(queries, k)
        return distances, indices

    def save(self, path: Path) -> None:
        # invariant: index must remain IndexIDMap over IndexFlatIP
        faiss.write_index(cast(faiss.Index, self._index), str(path))

    @classmethod
    def load(cls, path: Path) -> "FaissIndex":
        if not Path(path).is_file():
            raise FileNotFoundError(f"Index file not found: {path}")

        obj = cls.__new__(cls)
        obj._index = cast(_FaissIndexProto, faiss.read_index(str(path)))

        # invariant: ID mapping must be preserved across persistence
        if not isinstance(obj._index, faiss.IndexIDMap):
            raise TypeError("Loaded FAISS index is not IndexIDMap (ID mode required)")

        # invariant: base index must be inner-product for cosine equivalence
        def _unwrap_index(index):
            seen = set()
            while True:
                if id(index) in seen:
                    break
                seen.add(id(index))

                if hasattr(index, "index"):
                    index = index.index
                    continue
                if hasattr(index, "base_index"):
                    index = index.base_index
                    continue

                break
            return index

        base = _unwrap_index(obj._index)
        # logger.info("metric_type: " + str(base.metric_type)) // 0

        if not hasattr(base, "metric_type"):
            raise TypeError(f"Cannot determine metric type for index: {type(base)}")

        if base.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise TypeError(
                f"FAISS index must use INNER_PRODUCT (cosine), got metric_type={base.metric_type}"
            )

        return obj
