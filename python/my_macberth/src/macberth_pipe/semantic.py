# semantic.py

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import sqlite3
from typing import Optional

from .faiss_store import FaissStore
from .types import Embeddings, ChunkMeta

FAISS_FILE_PATH = Path("../../../faiss-cache/faiss-index")
SQLITE_DB_PATH = Path("../../eebo-data/eebo-tcp_metadata.sqlite").resolve()


class SemanticIndex:
    """
    Semantic search abstraction.
    Uses FAISS internally for persistent or in-memory search.
    Stores FAISS IDs in SQLite for reproducibility and deduplication.
    """

    def __init__(
        self,
        emb: Embeddings,
        store_dir: Path = FAISS_FILE_PATH,
        sqlite_db: Path = SQLITE_DB_PATH,
    ):
        self.emb = emb
        self.store_dir = store_dir
        self.sqlite_db = sqlite_db

        # Ensure SQLite table exists
        self._ensure_sqlite_table()

        # Initialize FAISS
        if store_dir:
            self.faiss_store = FaissStore(store_dir, sqlite_db=sqlite_db)
            if self.faiss_store.index is None:
                self.faiss_store.build(emb.vectors)
            else:
                self.faiss_store.append(emb.vectors)
            self.faiss_store.register_embeddings(emb.metas)
        else:
            # In-memory FAISS index
            import faiss
            dim = emb.vectors.shape[1]
            self.faiss_store = None
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(emb.vectors.astype("float32"))

    def _ensure_sqlite_table(self):
        """Create table for FAISS ID mapping if it doesn't exist."""
        conn = sqlite3.connect(self.sqlite_db)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS faiss_embeddings (
                faiss_id INTEGER PRIMARY KEY,
                doc_id TEXT NOT NULL,
                chunk_idx INTEGER NOT NULL,
                UNIQUE(doc_id, chunk_idx)
            )
        """)
        conn.commit()
        conn.close()

    def search(self, query_vec: np.ndarray, top_k: int = 5):
        q = np.asarray(query_vec, dtype="float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)

        results = []

        if self.faiss_store:
            # FAISS search returns faiss_id, doc_id, chunk_idx directly
            idx_results = self.faiss_store.search(q, top_k=top_k)
            for r in idx_results:
                # Lookup meta from doc_id and chunk_idx
                meta = next(
                    (m for m in self.emb.metas if m.doc_id == r["doc_id"] and m.chunk_idx == r["chunk_idx"]),
                    None
                )
                if meta:
                    results.append({
                        "rank": r["rank"],
                        "score": r["score"],
                        "doc_id": meta.doc_id,
                        "chunk_idx": meta.chunk_idx,
                        "text": meta.text,
                        "start_char": meta.start_char,
                        "end_char": meta.end_char,
                        "title": meta.title,
                        "author": meta.author,
                        "year": meta.year,
                        "permalink": meta.permalink
                    })
        else:
            # In-memory search
            scores, idxs = self.index.search(q, top_k)
            for rank, idx in enumerate(idxs[0]):
                meta = self.emb.metas[idx]
                results.append({
                    "rank": rank,
                    "score": float(scores[0, rank]),
                    "doc_id": meta.doc_id,
                    "chunk_idx": meta.chunk_idx,
                    "text": meta.text,
                    "start_char": meta.start_char,
                    "end_char": meta.end_char,
                    "title": meta.title,
                    "author": meta.author,
                    "year": meta.year,
                    "permalink": meta.permalink
                })

        return results
