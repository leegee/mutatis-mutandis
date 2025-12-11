import sqlite3
from pathlib import Path
from typing import Optional, Iterable, Tuple

SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    faiss_id INTEGER UNIQUE,
    eebo_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    hash TEXT UNIQUE,
    text TEXT
);
"""

class EmbeddingStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute(SCHEMA)
        self.conn.commit()

    # --- lookup ---
    def has_hash(self, h: str) -> bool:
        cur = self.conn.execute(
            "SELECT 1 FROM embeddings WHERE hash = ?", (h,)
        )
        return cur.fetchone() is not None

    def get_by_faiss_id(self, fid: int) -> Optional[Tuple[int,int,int,str]]:
        """
        Returns (id, eebo_id, chunk_index, text)
        """
        cur = self.conn.execute(
            "SELECT id, eebo_id, chunk_index, text FROM embeddings WHERE faiss_id = ?",
            (fid,)
        )
        return cur.fetchone()

    def get_by_hash(self, h: str) -> Optional[Tuple[int, int, int]]:
        """
        Returns (faiss_id, eebo_id, chunk_index)
        """
        cur = self.conn.execute(
            "SELECT faiss_id, eebo_id, chunk_index FROM embeddings WHERE hash = ?",
            (h,)
        )
        return cur.fetchone()

    # --- insert ---
    def add_embedding(
        self,
        faiss_id: int,
        eebo_id: int,
        chunk_index: int,
        h: str,
        text: str
    ):
        self.conn.execute(
            "INSERT INTO embeddings (faiss_id, eebo_id, chunk_index, hash, text) "
            "VALUES (?, ?, ?, ?, ?)",
            (faiss_id, eebo_id, chunk_index, h, text)
        )
        self.conn.commit()

    # --- bulk insert (faster with batching) ---
    def add_many(self, rows: Iterable[Tuple[int,int,int,str,str]]):
        """
        rows = [(faiss_id, eebo_id, chunk_index, hash, text), ...]
        """
        self.conn.executemany(
            "INSERT INTO embeddings (faiss_id, eebo_id, chunk_index, hash, text) "
            "VALUES (?, ?, ?, ?, ?)",
            rows
        )
        self.conn.commit()

    def close(self):
        self.conn.close()
