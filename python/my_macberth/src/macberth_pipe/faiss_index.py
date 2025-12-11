import faiss
import numpy as np
from pathlib import Path
import pickle

class FaissIndex:
    def __init__(self, dim: int, path: Path):
        self.path = path

        if path.exists():
            self.index = faiss.read_index(str(path))
        else:
            # Flat L2; can be replaced with HNSW easily
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))

    def add(self, vectors: np.ndarray, ids: np.ndarray):
        """
        vectors: (n, dim)
        ids:     (n,) int64 custom IDs (doc or chunk IDs)
        """
        assert vectors.shape[0] == ids.shape[0]
        self.index.add_with_ids(vectors, ids)

    def search(self, query: np.ndarray, k: int = 10):
        """
        query: (dim,)
        """
        query = query.reshape(1, -1)
        distances, ids = self.index.search(query, k)
        return distances[0], ids[0]

    def save(self):
        faiss.write_index(self.index, str(self.path))
