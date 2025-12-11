from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .types import Embeddings, ChunkMeta

FAISS_FILE_PATH = Path("faiss-cache/faiss-index")

class SemanticIndex:
    """
    Semantic search abstraction. Uses FAISS internally for persistent or in-memory search.
    """
    def __init__(self, emb: Embeddings, store_dir: Path = FAISS_FILE_PATH):
        self.emb = emb
        self.store_dir = store_dir

        if store_dir:
            # Use persistent FAISS store
            self.faiss_store = FaissStore(store_dir)
            if self.faiss_store.index is None:
                self.faiss_store.build(emb)
            else:
                self.faiss_store.append(emb)
        else:
            # In-memory FAISS index
            dim = emb.vectors.shape[1]
            import faiss
            self.faiss_store = None
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(emb.vectors.astype("float32"))

    def search(self, query_vec: np.ndarray, top_k: int = 5):
        q = np.asarray(query_vec, dtype="float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)

        results = []
        if self.faiss_store:
            # Persistent FAISS
            idx_results = self.faiss_store.search(q, top_k=top_k)
            for r in idx_results:
                # retrieve full ChunkMeta from embeddings
                meta = next((m for m in self.emb.metas if m.doc_id == r['doc_id'] and m.chunk_idx == r['chunk_idx']), None)
                if meta:
                    results.append({
                        'rank': r['rank'],
                        'score': r['score'],
                        'doc_id': meta.doc_id,
                        'chunk_idx': meta.chunk_idx,
                        'text': meta.text,
                        'start_char': meta.start_char,
                        'end_char': meta.end_char,
                        'title': meta.title,
                        'author': meta.author,
                        'year': meta.year,
                        'permalink': meta.permalink
                    })
        else:
            # In-memory search
            scores, idxs = self.index.search(q, top_k)
            for rank, idx in enumerate(idxs[0]):
                meta = self.emb.metas[idx]
                results.append({
                    'rank': rank,
                    'score': float(scores[0, rank]),
                    'doc_id': meta.doc_id,
                    'chunk_idx': meta.chunk_idx,
                    'text': meta.text,
                    'start_char': meta.start_char,
                    'end_char': meta.end_char,
                    'title': meta.title,
                    'author': meta.author,
                    'year': meta.year,
                    'permalink': meta.permalink
                })
        return results
        