import zarr
import numpy as np

from lib.corpus_logging import logger
from lib.zarr_store_dirs import store_dirs

BATCH_SIZE = 8192
_NO_WPOS = -1


class _LazyScaleEmbeddings:
    """
    Array-like accessor for one embedding scale (local/medium/broad),
    backed by CorpusFaissIndex.reconstruct() instead of an eagerly loaded
    (N, 768) matrix for the whole corpus. Full-corpus eager loading of
    all three scales does not fit in memory (~15.7 GiB for ~1.8M events);
    this replaces it with on-demand reconstruction from the per-year
    FAISS indices, which store vectors verbatim (IndexFlatIP).

    Supports both scalar indexing (emb_local[pos] -> (dim,) array) and
    array indexing (emb_local[positions] -> (n, dim) array) — the same
    contract the eager array previously provided, so multiscale_search
    and every other caller needs no changes.

    PERF: array indexing groups requested positions by pub_year before
    reconstructing, and issues one batched CorpusFaissIndex.reconstruct_many()
    call per year instead of one CorpusFaissIndex.reconstruct() call per
    position. This is called on every concept-year (via load_vectors ->
    get_concatenated_embeddings) as well as once for the corpus-wide
    global projection, so avoiding a Python-level FAISS call per single
    event -- multiplied by 3 scales -- matters across the whole pipeline,
    not just for the one-time global fit. Output order always matches
    the order of the input positions, regardless of grouping.

    No result caching here by design — this is a thin on-demand layer.
    Callers that repeatedly touch the same event_ids across multiple
    phases (UMAP, clustering, BFS) should go through EmbeddingCache,
    which caches on top of this and needs no changes either.
    """

    def __init__(self, lookup: "ZarrEventLookup", index, scale: str, dim: int):
        self._lookup = lookup
        self._index = index   # dict[year][scale] -> CorpusFaissIndex
        self._scale = scale
        self._dim = dim

    def _reconstruct_one(self, pos: int) -> np.ndarray:
        eid = int(self._lookup.event_id[pos])
        year = int(self._lookup.pub_year[pos])
        year_indices = self._index.get(year)
        if year_indices is None:
            raise KeyError(
                f"No FAISS index for pub_year={year} (event_id={eid}, "
                f"scale={self._scale}). Available years: {sorted(self._index.keys())}"
            )
        return year_indices[self._scale].reconstruct(eid)

    def _reconstruct_batch(self, positions: np.ndarray) -> np.ndarray:
        """
        Reconstruct multiple positions, batched per pub_year so each
        FAISS index is called once with all the ids it owns rather than
        once per id. `positions` may span any mix of years in any order;
        output rows are returned in the same order as `positions`.
        """
        years = self._lookup.pub_year[positions]
        eids = self._lookup.event_id[positions]

        # Stable sort so equal-year runs are contiguous; we scatter back
        # to original order at the end via `order`.
        order = np.argsort(years, kind="stable")
        sorted_years = years[order]
        sorted_eids = eids[order]

        out_sorted = np.empty((len(positions), self._dim), dtype=np.float32)

        n = len(order)
        start = 0
        while start < n:
            end = start + 1
            year = sorted_years[start]
            while end < n and sorted_years[end] == year:
                end += 1

            year_int = int(year)
            year_indices = self._index.get(year_int)
            if year_indices is None:
                bad_eid = int(sorted_eids[start])
                raise KeyError(
                    f"No FAISS index for pub_year={year_int} (event_id={bad_eid}, "
                    f"scale={self._scale}). Available years: {sorted(self._index.keys())}"
                )

            for scale in ("local", "medium", "broad"):
                idx = year_indices[scale]
                logger.info(
                    "[zarr-event-lookup-check] year=%d scale=%s contains=%s ntotal=%d",
                    year_int,
                    scale,
                    int(sorted_eids[start]) in idx.ids(),
                    idx.ntotal,
                )

            batch_ids = sorted_eids[start:end]
            try:
                missing = [
                    int(eid)
                    for eid in batch_ids
                    if int(eid) not in year_indices[self._scale].ids()
                ]

                if missing:
                    logger.error(
                        "[lookup] FAISS mismatch year=%d scale=%s requested=%d missing=%d",
                        year_int,
                        self._scale,
                        len(batch_ids),
                        len(missing),
                    )
                    logger.error(
                        "[lookup] missing sample=%s",
                        missing[:10],
                    )
                out_sorted[start:end] = year_indices[self._scale].reconstruct_many(batch_ids)
            except KeyError:
                logger.error(
                    "[lookup] year=%d scale=%s batch=%d",
                    year_int,
                    self._scale,
                    len(batch_ids),
                )

                logger.error(
                    "[lookup] first ids=%s",
                    batch_ids[:10].tolist(),
                )

                raise

            start = end

        out = np.empty_like(out_sorted)
        out[order] = out_sorted
        return out

    def __getitem__(self, idx):
        is_scalar = isinstance(idx, (int, np.integer))

        if is_scalar:
            return self._reconstruct_one(int(idx))

        positions = np.atleast_1d(idx).astype(np.int64)
        return self._reconstruct_batch(positions)


class ZarrEventLookup:
    """
    In-memory index of Tier 1 observation events, stored as parallel numpy
    arrays (struct-of-arrays) keyed by row position, plus a single
    event_id -> row position dict.

    vector_id is stored as metadata only — NOT used as a lookup key.

    Metadata is loaded eagerly into compact NumPy arrays. Embeddings remain in
    the per-year FAISS indices and are reconstructed lazily on demand via
    CorpusFaissIndex.reconstruct(). This avoids loading approximately 16 GiB of
    embedding matrices into memory while preserving the existing array-like API.

    Exposes three embedding scales (local, medium, broad) through lazy array-like
    accessors. Downstream code continues to index emb_local, emb_medium, and emb_broad
    exactly as before, while vectors are reconstructed on demand.
    """

    _FIELDS = {
        "event_id":         np.int64,
        "vector_id":        np.int64,
        "corpus":           object,
        "doc_id":           object,
        "token":            object,
        "token_idx":        np.int64,
        "window_id":        np.int64,
        "window_token_pos": np.int64,
        "pub_year":         np.int16,
    }


    def __init__(self, root):
        self.root            = root
        self._pos: dict[int, int] = {}
        self._chunks: dict[str, list] = {field: [] for field in self._FIELDS}
        self._index = None
        self.emb_local = self.emb_medium = self.emb_broad = None   # set via attach_index()
        self._build()


    @property
    def shape(self):
        self._require_index()
        return (len(self.event_id), self.emb_local._dim)


    @property
    def dtype(self):
        return np.float32


    @property
    def available_years(self) -> np.ndarray:
        return np.unique(self.pub_year)


    def __len__(self):
        return len(self.event_id)


    def _build(self):
        logger.info("[tier2 zarr-stream] building event lookup with multi-scale embeddings")
        for store_dir in store_dirs(self.root):
            logger.info("[tier2 zarr-stream] reading store %s", store_dir)
            g = zarr.open_group(str(store_dir), mode="r")
            if "events" not in g:
                continue
            self._load_store(g["events"], store_dir)
        self._finalize()
        logger.info(f"[tier2 zarr-stream] events={len(self._pos)}")


    def _load_store(self, e, store_dir):
        """
        Load event metadata only. Embeddings remain resident in the per-year
        FAISS indices and are reconstructed lazily when accessed.
        """
        # print(e.tree())
        # print(e["event_id"].shape)
        # print(e["corpus"].shape)

        required = {"event_id", "corpus", "doc_id", "vector_id"}
        missing = required - set(e.keys())
        if missing:
            raise KeyError( f"Missing Tier1 event fields {sorted(missing)} in {store_dir}" )

        wpos = e.get("window_token_pos")
        n = e["event_id"].shape[0]

        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)

            # --- cheap fields first ---
            b_eids   = e["event_id"][start:end]
            b_vids   = e["vector_id"][start:end]
            b_docs   = e["doc_id"][start:end]
            b_corpus = e["corpus"][start:end]
            b_toks   = e["token"][start:end]
            b_idxs   = e["token_idx"][start:end]
            b_wins   = e["window_id"][start:end]
            b_years  = e["pub_year"][start:end]

            b_wpos       = wpos[start:end] if wpos is not None else None
            b_toks       = b_toks.astype(str)
            b_docs       = b_docs.astype(str)
            b_corpus     = b_corpus.astype(str)
            b_toks_lower = np.char.lower(b_toks)

            keep = np.ones(end - start, dtype=bool)

            self._chunks["event_id"].append(np.asarray(b_eids, dtype=np.int64)[keep])
            self._chunks["vector_id"].append(np.asarray(b_vids, dtype=np.int64)[keep])
            self._chunks["doc_id"].append(b_docs[keep])
            self._chunks["corpus"].append(b_corpus[keep])
            self._chunks["token"].append(b_toks[keep])
            self._chunks["token_idx"].append(np.asarray(b_idxs, dtype=np.int64)[keep])
            self._chunks["window_id"].append(np.asarray(b_wins, dtype=np.int64)[keep])
            self._chunks["pub_year"].append(np.asarray(b_years, dtype=np.int16)[keep])   # NEW

            if b_wpos is not None:
                wpos_col = np.asarray(b_wpos, dtype=np.int64)[keep]
            else:
                wpos_col = np.full(int(keep.sum()), _NO_WPOS, dtype=np.int64)
            self._chunks["window_token_pos"].append(wpos_col)


    def attach_index(self, index: dict[int, dict[str, "CorpusFaissIndex"]]) -> None:
        """
        Attach the per-year, per-scale FAISS indices used for lazy embedding reconstruction.
        After attachment, emb_local, emb_medium, and emb_broad behave like NumPy arrays,
        but each vector is reconstructed from the appropriate FAISS index on demand.
        This preserves the previous indexing API while avoiding eager loading of the full
        embedding matrices.
        """
        self._index = index
        any_year = next(iter(index.values()))
        dim = next(iter(any_year.values())).dim
        self.emb_local  = _LazyScaleEmbeddings(self, index, "local", dim)
        self.emb_medium = _LazyScaleEmbeddings(self, index, "medium", dim)
        self.emb_broad  = _LazyScaleEmbeddings(self, index, "broad", dim)


    def _require_index(self):
        if self.emb_local is None:
            raise RuntimeError( "attach_index() must be called before accessing embeddings." )


    def _finalize(self):
        n_total = sum(arr.shape[0] for arr in self._chunks["event_id"])

        if n_total == 0:
            for field, dtype in self._FIELDS.items():
                setattr(self, field, np.empty(0, dtype=dtype))
            self.emb_local = self.emb_medium = self.emb_broad = np.empty((0, 768), dtype=np.float32)
            self._pos_by_occurrence = {}
            return

        for field, dtype in self._FIELDS.items():
            setattr(self, field, np.concatenate(self._chunks[field]).astype(dtype, copy=False))

        self._pos = {int(eid): pos for pos, eid in enumerate(self.event_id)}
        self._build_position_index()
        self._chunks.clear()
        logger.info(f"[tier2] loaded {n_total:,} events with multi-scale embeddings")


    def _build_position_index(self):
        """
        Build a (corpus, doc_id, token_idx) -> [event_id, ...] index for
        find_event_ids_by_positions(), so repeated corpus-occurrence lookups
        don't have to linear-scan the entire event table on every call.

        A single corpus occurrence can correspond to multiple Tier 1 events
        (observed under multiple contextual windows), hence the list value.

        Built once, eagerly, alongside self._pos -- same tradeoff as
        event_id -> pos: O(n) to build once at load time, O(1) amortized per
        lookup after that, instead of O(n) per call.
        """
        self._pos_by_occurrence: dict[tuple[str, str, int], list[int]] = {}

        # .tolist() up front so the loop body is pure-Python scalars, not
        # repeated numpy scalar boxing/unboxing per element.
        corpus_list    = self.corpus.tolist()
        doc_id_list    = self.doc_id.tolist()
        token_idx_list = self.token_idx.tolist()
        event_id_list  = self.event_id.tolist()

        for corpus, doc_id, token_idx, eid in zip(
            corpus_list, doc_id_list, token_idx_list, event_id_list
        ):
            key = (corpus, doc_id, token_idx)
            bucket = self._pos_by_occurrence.get(key)
            if bucket is None:
                self._pos_by_occurrence[key] = [eid]
            else:
                bucket.append(eid)


    def get_ensemble_embedding(self, pos: int, weights=[0.25, 0.50, 0.25]):
        self._require_index()
        return (
            weights[0] * self.emb_local[pos] +
            weights[1] * self.emb_medium[pos] +
            weights[2] * self.emb_broad[pos]
        )


    def get_event_metadata(self, event_id: int) -> dict:
        """
        Return event provenance without reconstructing embeddings.

        Embedding reconstruction is deliberately excluded because it requires
        attached FAISS indices and is expensive. Most consumers only need the
        corpus coordinates of an event.
        """
        pos = self._pos[int(event_id)]
        return self._row_to_dict(pos)


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
            "corpus": str(self.corpus[pos]),
            "token": str(self.token[pos]),
            "token_idx": int(self.token_idx[pos]),
            "window_id": int(self.window_id[pos]),
            "window_token_pos": None if wpos == _NO_WPOS else wpos,
            "pub_year": int(self.pub_year[pos]),
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


    def find_matching_event_ids( self, forms, false_positives=None, ):
        return list(
            self.iter_matching_event_ids(
                forms,
                false_positives,
            )
        )


    def find_event_ids_by_positions(self, positions):
        """
        Resolve corpus occurrence positions to Tier 1 observation events.

        A single corpus occurrence may have multiple Tier 1 events because it can
        be observed under multiple contextual windows.

        Corpus is part of the key because doc_id is not globally unique.

        O(1) dict lookups against the (corpus, doc_id, token_idx) index built
        once in _finalize() / _build_position_index(), rather than a linear
        scan of every event in the lookup per call.
        """
        result = {}

        for corpus, doc_id, token_idx in positions:
            key = (
                str(corpus),
                str(doc_id),
                int(token_idx),
            )

            if key in result:
                continue

            result[key] = list(
                self._pos_by_occurrence.get(key, [])
            )

        return result


    def get_pos(self, event_id: int) -> int:
        """event_id -> row position. Raises KeyError if not present."""
        return self._pos[int(event_id)]


    def get_embeddings(self, event_ids, weights=(0.25, 0.50, 0.25)):
        """
        Returns a vectorized weighted ensemble by reconstructing the required local, medium and broad
        embeddings for the requested events. Reconstruction is performed lazily via the attached FAISS
        indices, while the weighted sum is computed in a single vectorized NumPy operation.
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
        self._require_index()

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
        Embeddings are reconstructed lazily and normalized per scale before concatenation.
        """
        self._require_index()
        positions = np.array([self.get_pos(int(eid)) for eid in event_ids], dtype=np.int64)

        def _norm_rows(M):
            norms = np.linalg.norm(M, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return M / norms

        local  = _norm_rows(self.emb_local[positions])
        medium = _norm_rows(self.emb_medium[positions])
        broad  = _norm_rows(self.emb_broad[positions])

        return np.concatenate([local, medium, broad], axis=1).astype(np.float32)


    def get_window_events( self, doc_id, window_id, ):
        mask = (
            (self.doc_id == doc_id)
            &
            (self.window_id == window_id)
        )

        positions = np.where(mask)[0]

        return [
            self._row_to_dict(int(pos))
            for pos in positions
        ]


    def get_window_text( self, doc_id, window_id, ):
        mask = (
            (self.doc_id == doc_id)
            &
            (self.window_id == window_id)
        )

        positions = np.where(mask)[0]

        events = sorted(
            (
                self._row_to_dict(int(pos))
                for pos in positions
            ),
            key=lambda x: x["token_idx"],
        )

        return " ".join(
            e["token"]
            for e in events
        )
