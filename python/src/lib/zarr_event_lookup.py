import zarr
import numpy as np

from lib.eebo_logging import logger
from lib.zarr_store_dirs import store_dirs

BATCH_SIZE = 8192
_NO_WPOS = -1

class ZarrEventLookup:
    """
    In-memory index of Tier 1 observation events, stored as parallel numpy
    arrays (struct-of-arrays) keyed by row position, plus a single
    event_id -> row position dict.

    When forms is provided, only events whose token matches one of the
    supplied forms are loaded. This is the normal path for single-concept
    runs and keeps memory use proportional to the concept, not the corpus.

    When forms is None, all events are loaded. This is required when
    querying across multiple concepts in a single run.

    vector_id is stored as metadata only — NOT used as a lookup key.

    Embeddings are loaded alongside metadata so that FAISS queries can be
    issued using the canonical Zarr vector rather than relying on FAISS
    internal vector storage. See module docstring for trade-offs and the
    deferred migration path to EeboFaissIndex.reconstruct().

    Loads three embedding scales and provides ensemble vectors for downstream use.
    """

    _FIELDS = {
        "event_id":         np.int64,
        "vector_id":        np.int64,
        "doc_id":           object,
        "token":            object,
        "token_idx":        np.int64,
        "window_id":        np.int64,
        "window_token_pos": np.int64,
    }


    def __init__(self, root, forms: set[str] | None = None, false_positives: set[str] | None = None):
        self.root            = root
        self.forms           = {f.lower() for f in forms} if forms else None
        self.false_positives = {f.lower() for f in false_positives} if false_positives else set()

        self._pos: dict[int, int] = {}
        self._chunks: dict[str, list] = {field: [] for field in self._FIELDS}

        self._emb_local_chunks  = []
        self._emb_medium_chunks = []
        self._emb_broad_chunks  = []

        self._build()


    def _build(self):
        logger.info("[tier2] building event lookup with multi-scale embeddings")
        if self.forms:
            logger.info(f"[tier2] filtering to forms={self.forms}")
        if self.false_positives:
            logger.info(f"[tier2] excluding false_positives={self.false_positives}")

        for store_dir in store_dirs(self.root):
            g = zarr.open_group(str(store_dir), mode="r")
            if "events" not in g:
                continue
            self._load_store(g["events"], store_dir)

        self._finalize()
        logger.info(f"[tier2] events={len(self._pos)}")


    def _load_store(self, e, store_dir):
        """Load events + multi-scale embeddings."""
        if "event_id" not in e:
            raise KeyError(f"Missing event_id in {store_dir} - rebuild Tier 1")

        wpos = e.get("window_token_pos")
        n = e["event_id"].shape[0]

        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)

            b_eids = e["event_id"][start:end]
            b_vids = e["vector_id"][start:end]
            b_docs = e["doc_id"][start:end]
            b_toks = e["token"][start:end]
            b_idxs = e["token_idx"][start:end]
            b_wins = e["window_id"][start:end]
            b_wpos = wpos[start:end] if wpos is not None else None

            b_local  = e["emb_local"][start:end]
            b_medium = e["emb_medium"][start:end]
            b_broad  = e["emb_broad"][start:end]

            b_toks = b_toks.astype(str)
            b_docs = b_docs.astype(str)
            b_toks_lower = np.char.lower(b_toks)

            if self.forms is not None:
                keep = np.isin(b_toks_lower, list(self.forms))
            else:
                keep = np.ones(end - start, dtype=bool)

            if self.false_positives:
                keep &= ~np.isin(b_toks_lower, list(self.false_positives))

            if not keep.any():
                continue

            self._chunks["event_id"].append(np.asarray(b_eids, dtype=np.int64)[keep])
            self._chunks["vector_id"].append(np.asarray(b_vids, dtype=np.int64)[keep])
            self._chunks["doc_id"].append(b_docs[keep])
            self._chunks["token"].append(b_toks[keep])
            self._chunks["token_idx"].append(np.asarray(b_idxs, dtype=np.int64)[keep])
            self._chunks["window_id"].append(np.asarray(b_wins, dtype=np.int64)[keep])

            if b_wpos is not None:
                wpos_col = np.asarray(b_wpos, dtype=np.int64)[keep]
            else:
                wpos_col = np.full(int(keep.sum()), _NO_WPOS, dtype=np.int64)
            self._chunks["window_token_pos"].append(wpos_col)

            self._emb_local_chunks.append(np.asarray(b_local, dtype=np.float32)[keep])
            self._emb_medium_chunks.append(np.asarray(b_medium, dtype=np.float32)[keep])
            self._emb_broad_chunks.append(np.asarray(b_broad, dtype=np.float32)[keep])


    def _finalize(self):
        n_total = sum(arr.shape[0] for arr in self._chunks["event_id"])

        if n_total == 0:
            for field, dtype in self._FIELDS.items():
                setattr(self, field, np.empty(0, dtype=dtype))
            self.emb_local = self.emb_medium = self.emb_broad = np.empty((0, 768), dtype=np.float32)
            return

        for field, dtype in self._FIELDS.items():
            setattr(self, field, np.concatenate(self._chunks[field]).astype(dtype, copy=False))

        self.emb_local  = np.concatenate(self._emb_local_chunks, axis=0)
        self.emb_medium = np.concatenate(self._emb_medium_chunks, axis=0)
        self.emb_broad  = np.concatenate(self._emb_broad_chunks, axis=0)

        self._pos = {int(eid): pos for pos, eid in enumerate(self.event_id)}

        self._chunks.clear()
        self._emb_local_chunks.clear()
        self._emb_medium_chunks.clear()
        self._emb_broad_chunks.clear()

        logger.info(f"[tier2] loaded {n_total:,} events with multi-scale embeddings")


    def get_ensemble_embedding(self, pos: int, weights=[0.25, 0.50, 0.25]):
        return (
            weights[0] * self.emb_local[pos] +
            weights[1] * self.emb_medium[pos] +
            weights[2] * self.emb_broad[pos]
        )


    def get_event(self, event_id: int) -> dict:
        pos = self._pos[int(event_id)]
        d = self._row_to_dict(pos)
        d["embedding"] = self.get_ensemble_embedding(pos)
        return d


    def _row_to_dict(self, pos: int) -> dict:
        wpos = int(self.window_token_pos[pos])
        return {
            "event_id": int(self.event_id[pos]),
            "vector_id": int(self.vector_id[pos]),
            "doc_id": str(self.doc_id[pos]),
            "token": str(self.token[pos]),
            "token_idx": int(self.token_idx[pos]),
            "window_id": int(self.window_id[pos]),
            "window_token_pos": None if wpos == _NO_WPOS else wpos,
        }


    def iter_matching_event_ids(self, forms, false_positives=None):
        """
        Yield event_ids whose token matches `forms` and is not in
        `false_positives`.
        """
        forms = {f.lower() for f in forms}
        false_positives = {f.lower() for f in (false_positives or [])}

        if len(self.token) == 0:
            return

        tokens_lower = np.char.lower(self.token.astype(str))
        mask = np.isin(tokens_lower, list(forms))

        if false_positives:
            mask &= ~np.isin(tokens_lower, list(false_positives))

        for eid in self.event_id[mask]:
            yield int(eid)


    def get_pos(self, event_id: int) -> int:
        """event_id -> row position. Raises KeyError if not present."""
        return self._pos[int(event_id)]


    def get_embeddings(self, event_ids):
        """
        Return (n, d) embedding matrix aligned to event_ids.
        Uses ensemble embedding.
        """
        return np.vstack([
            self.get_ensemble_embedding(self.get_pos(int(eid)))
            for eid in event_ids
        ]).astype(np.float32)
