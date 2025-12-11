# faiss_store.py

from pathlib import Path
import numpy as np
import sqlite3
from typing import Optional, List
import faiss
import logging

logger = logging.getLogger(__name__)


class FaissStore:
    """
    Persistent FAISS index with SQLite-backed ID mapping.
    Each embedding chunk gets a unique faiss_id stored in SQLite.
    """

    def __init__(self, store_dir: Path, sqlite_db: Optional[Path] = None):
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self.index_file = store_dir / "index.faiss"
        self.sqlite_db = sqlite_db
        self.index: Optional[faiss.Index] = None
        self.dim: Optional[int] = None

        if self.sqlite_db:
            self._ensure_sqlite_table()

        self._load_index()

    def _ensure_sqlite_table(self):
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

    def _load_index(self):
        if self.index_file.exists():
            self.index = faiss.read_index(str(self.index_file))
            self.dim = self.index.d
            logger.debug(f"Loaded FAISS index from {self.index_file}")
        else:
            self.index = None

    def build(self, emb_vectors: np.ndarray):
        self.dim = emb_vectors.shape[1]
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(emb_vectors.astype("float32"))
        self._save_index()
        logger.debug(f"Built new FAISS index with {emb_vectors.shape[0]} vectors")

    def append(self, emb_vectors: np.ndarray):
        if self.index is None:
            self.build(emb_vectors)
        else:
            self.index.add(emb_vectors.astype("float32"))
            self._save_index()
            logger.debug(f"Appended {emb_vectors.shape[0]} vectors to FAISS index")

    def _save_index(self):
        faiss.write_index(self.index, str(self.index_file))

    def register_embeddings(self, metas: list):
        """Register chunk metadata in SQLite to assign persistent faiss_ids."""
        if self.sqlite_db is None:
            return

        conn = sqlite3.connect(self.sqlite_db)
        c = conn.cursor()

        start_id = self._get_next_faiss_id()
        for i, meta in enumerate(metas):
            faiss_id = start_id + i
            try:
                c.execute(
                    "INSERT OR IGNORE INTO faiss_embeddings (faiss_id, doc_id, chunk_idx) VALUES (?, ?, ?)",
                    (faiss_id, meta.doc_id, meta.chunk_idx)
                )
            except sqlite3.IntegrityError:
                pass
        conn.commit()
        conn.close()
        logger.debug(f"Registered {len(metas)} embeddings in SQLite")

    def _get_next_faiss_id(self) -> int:
        if self.sqlite_db is None:
            return 0
        conn = sqlite3.connect(self.sqlite_db)
        c = conn.cursor()
        c.execute("SELECT MAX(faiss_id) FROM faiss_embeddings")
        row = c.fetchone()
        conn.close()
        return (row[0] + 1) if row[0] is not None else 0

    def search(self, query_vectors: np.ndarray, top_k: int = 5) -> List[dict]:
        """
        Returns list of dicts containing: query_idx, rank, score, faiss_id, doc_id, chunk_idx
        """
        q = np.asarray(query_vectors, dtype="float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if self.index is None:
            raise RuntimeError("FAISS index is not built")

        scores, idxs = self.index.search(q, top_k)
        results = []

        # Bulk lookup for faiss_id -> doc_id, chunk_idx
        if self.sqlite_db:
            conn = sqlite3.connect(self.sqlite_db)
            c = conn.cursor()
            faiss_ids = set(id for row in idxs for id in row)
            placeholders = ",".join("?" for _ in faiss_ids)
            c.execute(f"SELECT faiss_id, doc_id, chunk_idx FROM faiss_embeddings WHERE faiss_id IN ({placeholders})", tuple(faiss_ids))
            id_map = {row[0]: (row[1], row[2]) for row in c.fetchall()}
            conn.close()
        else:
            id_map = {}

        for qi, (score_row, idx_row) in enumerate(zip(scores, idxs)):
            for rank, (idx, score) in enumerate(zip(idx_row, score_row)):
                doc_id, chunk_idx = id_map.get(idx, (None, None))
                results.append({
                    "query_idx": qi,
                    "rank": rank,
                    "faiss_id": int(idx),
                    "score": float(score),
                    "doc_id": doc_id,
                    "chunk_idx": chunk_idx
                })

        return results
