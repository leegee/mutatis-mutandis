# faiss_store.py

from pathlib import Path
import faiss
import numpy as np
import json
from typing import Optional

from macberth_pipe.embedding import Embeddings
from macberth_pipe.embedding_store import EmbeddingsStore
from macberth_pipe.semantic import ChunkMeta


class FaissStore:
    """
    Persistent FAISS index that stays in sync with EmbeddingsStore.
    Stores:
      - index.faiss : FAISS binary index
      - mapping.json : mapping of FAISS row -> (doc_id, chunk_idx)
    """

    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.directory / "index.faiss"
        self.map_path = self.directory / "mapping.json"

        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.map_path) as f:
                self.mapping = json.load(f)
        else:
            self.index = None
            self.mapping = []

    def build(self, emb: Embeddings):
        dim = emb.vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(emb.vectors.astype("float32"))
        self.index = index

        # Build mapping
        self.mapping = [
            {
                "doc_id": m.doc_id,
                "chunk_idx": m.chunk_idx
            }
            for m in emb.metas
        ]

        self._save()

    def append(self, new_emb: Embeddings):
        if self.index is None:
            return self.build(new_emb)

        self.index.add(new_emb.vectors.astype("float32"))

        # Append mapping
        new_map = [
            {
                "doc_id": m.doc_id,
                "chunk_idx": m.chunk_idx
            }
            for m in new_emb.metas
        ]
        self.mapping.extend(new_map)

        self._save()

    def _save(self):
        faiss.write_index(self.index, str(self.index_path))
        with open(self.map_path, "w") as f:
            json.dump(self.mapping, f)

    def search(self, query_vec: np.ndarray, top_k: int = 5):
        q = np.asarray(query_vec, dtype="float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)

        scores, idxs = self.index.search(q, top_k)

        results = []
        for rank, idx in enumerate(idxs[0]):
            m = self.mapping[idx]
            results.append({
                "rank": rank,
                "score": float(scores[0, rank]),
                "doc_id": m["doc_id"],
                "chunk_idx": m["chunk_idx"],
            })

        return results
