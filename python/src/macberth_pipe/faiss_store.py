# faiss_store.py
from pathlib import Path
import numpy as np
import sqlite3
from typing import Optional, List, Tuple
import faiss
import logging

from .types import Embeddings

logger = logging.getLogger(__name__)


class FaissStore:
    """
    Persistent FAISS index with SQLite-backed ID mapping.

    Usage:
      store = FaissStore(store_dir=Path("..."), sqlite_db=Path("..."))
      store.build(emb)        # emb is Embeddings -> creates index and mapping
      store.append(emb)       # emb is Embeddings -> appends new vectors (skipping duplicates)
      results = store.search(qvec, top_k=5)  # returns list with faiss_id, score, doc_id, chunk_idx
    """

    def __init__(self, store_dir: Path, sqlite_db: Optional[Path] = None):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self.index_file = self.store_dir / "index.faiss"
        self.sqlite_db = Path(sqlite_db) if sqlite_db is not None else None

        self.index: Optional[faiss.Index] = None
        self.dim: Optional[int] = None

        if self.sqlite_db:
            self._ensure_sqlite_table()

        self._load_index()

    # ---------- SQLite helpers ----------
    def _ensure_sqlite_table(self):
        conn = sqlite3.connect(self.sqlite_db)
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS faiss_embeddings (
                faiss_id INTEGER PRIMARY KEY,
                doc_id TEXT NOT NULL,
                chunk_idx INTEGER NOT NULL,
                UNIQUE(doc_id, chunk_idx)
            )
            """
        )
        conn.commit()
        conn.close()

    def _get_next_faiss_id(self) -> int:
        if self.sqlite_db is None:
            return 0
        conn = sqlite3.connect(self.sqlite_db)
        c = conn.cursor()
        c.execute("SELECT MAX(faiss_id) FROM faiss_embeddings")
        row = c.fetchone()
        conn.close()
        return (row[0] + 1) if (row and row[0] is not None) else 0

    def _existing_doc_chunk_pairs(self) -> set:
        """Return set of (doc_id, chunk_idx) already registered."""
        if self.sqlite_db is None:
            return set()
        conn = sqlite3.connect(self.sqlite_db)
        c = conn.cursor()
        c.execute("SELECT doc_id, chunk_idx FROM faiss_embeddings")
        rows = c.fetchall()
        conn.close()
        return {(r[0], int(r[1])) for r in rows}

    def _register_many(self, rows: List[Tuple[int, str, int]]):
        """
        Insert rows = [(faiss_id, doc_id, chunk_idx), ...] using INSERT OR IGNORE.
        """
        if self.sqlite_db is None or not rows:
            return
        conn = sqlite3.connect(self.sqlite_db)
        c = conn.cursor()
        c.executemany(
            "INSERT OR IGNORE INTO faiss_embeddings (faiss_id, doc_id, chunk_idx) VALUES (?, ?, ?)",
            rows,
        )
        conn.commit()
        conn.close()

    # ---------- FAISS index persistence ----------
    def _load_index(self):
        if self.index_file.exists():
            try:
                self.index = faiss.read_index(str(self.index_file))
                # if IndexIDMap, we can read dimension; for others, attempt to get d
                try:
                    self.dim = self.index.d
                except Exception:
                    # fallback: if wrapped index, try to get ntotal>0 reconstruct
                    self.dim = None
                logger.debug("Loaded FAISS index from %s", self.index_file)
            except Exception as e:
                logger.warning("Failed to load FAISS index: %s", e)
                self.index = None
        else:
            self.index = None

    def _save_index(self):
        if self.index is None:
            raise RuntimeError("No FAISS index to save")
        faiss.write_index(self.index, str(self.index_file))
        logger.debug("Saved FAISS index to %s", self.index_file)

    def build(self, emb: Embeddings):
        """Build FAISS index from Embeddings object"""
        if emb.vectors.size == 0:
            self.index = None
            self.dim = None
            return
        self.dim = emb.vectors.shape[1]
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(emb.vectors.astype("float32"))
        self._save_index()
        logger.debug(f"Built new FAISS index with {emb.vectors.shape[0]} vectors")

    def append(self, emb: Embeddings):
        """Append embeddings to existing FAISS index"""
        if emb.vectors.size == 0:
            return
        if self.index is None:
            self.build(emb)
        else:
            self.index.add(emb.vectors.astype("float32"))
            self._save_index()
            logger.debug(f"Appended {emb.vectors.shape[0]} vectors to FAISS index")

        """
        Append new embeddings to an existing index. Skips metas already registered (doc_id,chunk_idx).
        If no index exists, behaves like build().
        """
        if emb.vectors.size == 0:
            logger.debug("No vectors to append (empty Embeddings)")
            return

        # If no index yet, just build
        if self.index is None:
            logger.debug("No existing index, building new one")
            self.build(emb)
            return

        # ensure index is an IndexIDMap; if not, wrap it (best effort)
        if not isinstance(self.index, faiss.IndexIDMap):
            logger.debug("Existing FAISS index is not IndexIDMap — wrapping it in IndexIDMap")
            # wrap: create new IndexIDMap and re-add existing vectors with implicit IDs 0..ntotal-1
            try:
                base = faiss.IndexFlatL2(self.index.d)
                new_index = faiss.IndexIDMap(base)
                # reconstruct existing vectors (may be expensive)
                ntotal = self.index.ntotal
                existing_vecs = np.stack([self.index.reconstruct(i) for i in range(ntotal)]).astype("float32")
                existing_ids = np.arange(0, ntotal, dtype="int64")
                new_index.add_with_ids(existing_vecs, existing_ids)
                self.index = new_index
                logger.debug("Wrapped existing index into IndexIDMap with %d vectors", ntotal)
            except Exception as e:
                logger.warning("Could not wrap existing index: %s", e)
                # fallback — still try to add without IDs (will increase ntotal but IDs unknown)
                try:
                    self.index.add(emb.vectors.astype("float32"))
                    self._save_index()
                    logger.info("Appended vectors without explicit IDs (no sqlite mapping updated)")
                    return
                except Exception as ex:
                    raise RuntimeError("Unable to append to existing FAISS index") from ex

        # Now index is IndexIDMap and we can add_with_ids
        existing_pairs = self._existing_doc_chunk_pairs()
        to_add_vecs = []
        to_add_rows = []  # (faiss_id, doc_id, chunk_idx)
        next_id = self._get_next_faiss_id()

        # iterate through metas + vectors in lockstep; skip already-present metas
        vec_idx = 0
        for meta in emb.metas:
            pair = (meta.doc_id, int(meta.chunk_idx))
            if pair in existing_pairs:
                # skip this chunk - already indexed
                vec_idx += 1
                continue
            # take corresponding vector from emb.vectors; but careful: emb may contain multiple vectors per doc
            vec = emb.vectors[vec_idx]
            to_add_vecs.append(vec)
            to_add_rows.append((next_id, meta.doc_id, int(meta.chunk_idx)))
            next_id += 1
            vec_idx += 1

        if not to_add_vecs:
            logger.debug("No new chunks to add (all were already indexed)")
            return

        vecs_arr = np.vstack(to_add_vecs).astype("float32")
        ids_arr = np.array([r[0] for r in to_add_rows], dtype="int64")
        # add with explicit IDs
        self.index.add_with_ids(vecs_arr, ids_arr)
        self._save_index()
        # register in sqlite
        self._register_many(to_add_rows)
        logger.info("Appended %d new vectors to FAISS and registered %d rows", len(to_add_vecs), len(to_add_rows))

    # ---------- Search ----------
    def search(self, query_vectors: np.ndarray, top_k: int = 5) -> List[dict]:
        """
        Returns list of dicts containing:
            query_idx, rank, faiss_id (int), score (float), doc_id (str or None), chunk_idx (int or None)
        """
        q = np.asarray(query_vectors, dtype="float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)

        if self.index is None:
            raise RuntimeError("FAISS index is not built")

        scores, idxs = self.index.search(q, top_k)
        results = []

        # Collect distinct faiss_ids (ignore -1 which some indexes use for empty slots)
        faiss_id_set = {int(x) for row in idxs for x in row if int(x) >= 0}

        id_map = {}
        if self.sqlite_db and faiss_id_set:
            conn = sqlite3.connect(self.sqlite_db)
            c = conn.cursor()
            placeholders = ",".join("?" for _ in faiss_id_set)
            c.execute(
                f"SELECT faiss_id, doc_id, chunk_idx FROM faiss_embeddings WHERE faiss_id IN ({placeholders})",
                tuple(faiss_id_set),
            )
            for fid, doc_id, chunk_idx in c.fetchall():
                id_map[int(fid)] = (doc_id, int(chunk_idx))
            conn.close()

        for qi, (score_row, idx_row) in enumerate(zip(scores, idxs)):
            for rank, (fid, score) in enumerate(zip(idx_row, score_row)):
                fid_int = int(fid)
                doc_id, chunk_idx = id_map.get(fid_int, (None, None))
                results.append(
                    {
                        "query_idx": qi,
                        "rank": rank,
                        "faiss_id": fid_int,
                        "score": float(score),
                        "doc_id": doc_id,
                        "chunk_idx": chunk_idx,
                    }
                )

        return results

    # ---------- Utilities ----------
    def get_total_vectors(self) -> int:
        return 0 if self.index is None else int(self.index.ntotal)

    def close(self):
        """No persistent connection to close, but keep method for API symmetry."""
        return
