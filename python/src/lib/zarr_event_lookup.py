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
        """
        Load events + multi-scale embeddings.

        PERFORMANCE NOTE: token/doc_id/etc are cheap fields (int64, short
        strings) - reading and filtering them per batch is fast. The
        embedding arrays (emb_local/medium/broad) are the expensive part:
        768-dim float32 per row, and Zarr has to decompress each chunk it
        touches. The keep-mask (derived from token matches against
        self.forms) is now computed FIRST, from the cheap fields only, and
        batches with zero matches skip the embedding reads entirely via
        `continue` before ever touching e["emb_local"]/["emb_medium"]/
        ["emb_broad"]. Previously the embedding slices were read
        unconditionally for every batch regardless of whether anything in
        that batch matched - meaning a single-concept forms-filtered load
        still paid the FULL corpus's embedding-decompression cost, even
        though most of the decompressed data was immediately discarded.
        With a concept's occurrences spread across many documents (and
        therefore likely touching most batches at least once), this alone
        won't eliminate embedding reads entirely for a typical concept -
        but any batch that genuinely contains zero matches (increasingly
        likely for rarer concepts, or smaller shards) now costs almost
        nothing instead of a full embedding decompression.
        """
        if "event_id" not in e:
            raise KeyError(f"Missing event_id in {store_dir} - rebuild Tier 1")

        wpos = e.get("window_token_pos")
        n = e["event_id"].shape[0]

        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)

            # --- cheap fields first ---
            b_eids = e["event_id"][start:end]
            b_vids = e["vector_id"][start:end]
            b_docs = e["doc_id"][start:end]
            b_toks = e["token"][start:end]
            b_idxs = e["token_idx"][start:end]
            b_wins = e["window_id"][start:end]
            b_wpos = wpos[start:end] if wpos is not None else None

            b_toks = b_toks.astype(str)
            b_docs = b_docs.astype(str)
            b_toks_lower = np.char.lower(b_toks)

            if self.forms is not None:
                keep = np.isin(b_toks_lower, list(self.forms))
            else:
                keep = np.ones(end - start, dtype=bool)

            if self.false_positives:
                keep &= ~np.isin(b_toks_lower, list(self.false_positives))

            # Skip the expensive embedding reads entirely for batches with
            # no matches - this is the actual fix. Previously b_local/
            # b_medium/b_broad were read unconditionally above this point,
            # so every batch paid full decompression cost regardless of
            # whether `keep` had any True values.
            if not keep.any():
                continue

            # --- expensive fields only for batches with at least one match ---
            b_local  = e["emb_local"][start:end]
            b_medium = e["emb_medium"][start:end]
            b_broad  = e["emb_broad"][start:end]

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
        `false_positives`. Deduplicated — see note below.
        """
        forms = {f.lower() for f in forms}
        false_positives = {f.lower() for f in (false_positives or [])}

        if len(self.token) == 0:
            return

        tokens_lower = np.char.lower(self.token.astype(str))
        mask = np.isin(tokens_lower, list(forms))

        if false_positives:
            mask &= ~np.isin(tokens_lower, list(false_positives))

        # Defensive dedup: event_id should be unique per row by construction,
        # but if the same event_id ever appears twice in the underlying arrays
        # then yielding it twice from here would propagate duplicates into every
        # consumer.
        seen: set[int] = set()
        for eid in self.event_id[mask]:
            eid = int(eid)
            if eid in seen:
                continue
            seen.add(eid)
            yield eid


    def get_pos(self, event_id: int) -> int:
        """event_id -> row position. Raises KeyError if not present."""
        return self._pos[int(event_id)]


    def get_embeddings(self, event_ids, weights=(0.25, 0.50, 0.25)):
        """
        Return (n, d) embedding matrix aligned to event_ids.
        Uses ensemble embedding.

        Vectorized: previously this built the (n, d) matrix via a Python
        loop calling get_ensemble_embedding() (itself a per-row weighted
        sum) once per event_id, then np.vstack-ed the results - n separate
        small numpy allocations plus n dict lookups in a Python loop. This
        version does one array of position lookups, then three whole-array
        fancy-index reads and a single vectorized weighted sum - one set
        of allocations regardless of n, instead of n of them.
        """
        positions = np.array([self.get_pos(int(eid)) for eid in event_ids], dtype=np.int64)

        return (
            weights[0] * self.emb_local[positions] +
            weights[1] * self.emb_medium[positions] +
            weights[2] * self.emb_broad[positions]
        ).astype(np.float32)


    def get_concatenated_embedding(self, pos: int) -> np.ndarray:
        """
        Return local/medium/broad embeddings, each L2-normalized, concatenated
        into one (3*d,) vector — preserves per-scale structure for clustering
        rather than collapsing it via weighted average.
        """
        def _norm(v):
            n = np.linalg.norm(v)
            return v / n if n > 0 else v

        return np.concatenate([
            _norm(self.emb_local[pos]),
            _norm(self.emb_medium[pos]),
            _norm(self.emb_broad[pos]),
        ]).astype(np.float32)


    def get_concatenated_embeddings(self, event_ids) -> np.ndarray:
        """
        Vectorized (n, 3*d) concatenated embedding matrix aligned to event_ids.
        """
        positions = np.array([self.get_pos(int(eid)) for eid in event_ids], dtype=np.int64)

        def _norm_rows(M):
            norms = np.linalg.norm(M, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return M / norms

        local  = _norm_rows(self.emb_local[positions])
        medium = _norm_rows(self.emb_medium[positions])
        broad  = _norm_rows(self.emb_broad[positions])

        return np.concatenate([local, medium, broad], axis=1).astype(np.float32)
