import sqlite3
from pathlib import Path
import numpy as np
import faiss
from typing import Optional
from macberth_pipe.types import Embeddings, ChunkMeta # noqa F401


class FaissStore:
    def __init__(self, store_dir: Path, sqlite_db: Optional[Path] = None):
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.store_dir / "index.faiss"
        self.sqlite_db = sqlite_db

        self.index: Optional[faiss.IndexFlatL2] = None
        self.dim: Optional[int] = None

        # Initialize SQLite table if provided
        if self.sqlite_db:
            self._init_sqlite()

        # Load FAISS index if exists
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))

    def _init_sqlite(self):
        with sqlite3.connect(self.sqlite_db) as conn:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT,
                    chunk_idx INTEGER,
                    title TEXT,
                    author TEXT,
                    year TEXT,
                    permalink TEXT
                )
            """)
            conn.commit()

    def build(self, emb: Embeddings):
        self.dim = emb.vectors.shape[1]
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(emb.vectors.astype("float32"))
        self._save_metadata(emb)
        self._save_index()

    def append(self, emb: Embeddings):
        if self.index is None:
            self.build(emb)
        else:
            self.index.add(emb.vectors.astype("float32"))
            self._save_metadata(emb)
            self._save_index()

    def _save_index(self):
        faiss.write_index(self.index, str(self.index_path))

    def _save_metadata(self, emb: Embeddings):
        if not self.sqlite_db:
            return
        with sqlite3.connect(self.sqlite_db) as conn:
            c = conn.cursor()
            rows = [(
                m.doc_id, m.chunk_idx, m.title, m.author, m.year, m.permalink
            )
                    for m in emb.metas]
            c.executemany("""
                INSERT INTO embeddings
                (doc_id, chunk_idx, title, author, year, permalink)
                VALUES (?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()

    def search(self, query_vec: np.ndarray, top_k: int = 5):
        q = np.asarray(query_vec, dtype="float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if self.index is None:
            raise RuntimeError("FAISS index not initialized")
        scores, idxs = self.index.search(q, top_k)
        results = [
            {'rank': r, 'idx': int(id_), 'score': float(scores[0, r])}
            for r, id_ in enumerate(idxs[0])
        ]
        return results
