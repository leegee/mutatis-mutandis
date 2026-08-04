#!/usr/bin/env python
"""
eebo_faiss.py

FAISS retrieval layer for corpus semantic event embeddings.

This module intentionally treats FAISS as a *derived geometric index*,
not as a canonical data store.

Architectural role
------------------

The corpus embedding pipeline now operates as:

    Postgres (identity + text provenance)
        ↓
    Zarr event log (canonical semantic events)
        ↓
    FAISS index (approximate geometric retrieval)

FAISS therefore owns ONLY:
    - vector geometry
    - event-id lookup
    - similarity search

It does NOT own:
    - metadata
    - provenance
    - semantic interpretation

Vector reconstruction
---------------------

EeboFaissIndex exposes a reconstruct() method that retrieves the
unit-normalised vector stored for a given event_id directly from the
FAISS index, without reaching back into the Zarr store.

This is currently supported because the pipeline uses IndexFlatIP, which
stores vectors verbatim. It is NOT supported by IndexHNSWFlat, which does
not retain vectors after index construction.

The intended future use is in ZarrEventLookup (tier2_concept_neighbours.py):
once the index type is confirmed stable, the "embedding" field can be
dropped from by_event_id and replaced with reconstruct() calls, eliminating
the in-memory copy of the full corpus embedding matrix. Until that migration
is made, Zarr remains the authoritative source for query vectors and
reconstruct() is provided but not called.

Core invariant
--------------

Every indexed vector corresponds to a stable semantic event ID.

An event ID should uniquely identify:
    - a token occurrence
    - in a document
    - at a specific token position

All vectors are unit-normalised before insertion so that:

    inner product == cosine similarity

This guarantees stable geometric interpretation across:
    - ingestion
    - querying
    - persistence
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import faiss
import numpy as np

from lib.corpus_logging import logger
from lib.corpus_config import faiss_index_paths, discover_index_years, FAISS_INDEX_DIR

class EeboFaissIndex:
    """
    Thin wrapper around FAISS IndexIDMap2.

    Design goals:
        - explicit semantic event IDs
        - cosine similarity semantics
        - persistence validation
        - future-compatible with HNSW migration
    """

    def __init__(self, dim: int, exact: bool = True):
        self.dim = dim

        if exact:
            self.base = faiss.IndexFlatIP(dim)
        else:
            # future-friendly approximate mode
            # M=32 is a good general-purpose HNSW default
            self.base = faiss.IndexHNSWFlat(dim, 32)
            self.base.metric_type = faiss.METRIC_INNER_PRODUCT

        self._index = faiss.IndexIDMap2(self.base)
        self._ids = set()

    @staticmethod
    def wipe_faiss_index() -> None:
        """
        Deletes persisted FAISS index from disk.

        Failure mode:
            - if file is in use, OS will raise
            - if path is wrong, silent mismatch risk upstream

        This must always be followed by rebuild_from_tier1().
        """

        shutil.rmtree( FAISS_INDEX_DIR )
        logger.info(f"[faiss] deleted index={FAISS_INDEX_DIR}")

    @staticmethod
    def _normalize(
        x: np.ndarray,
        event_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Enforce cosine/IP equivalence.

        Failure mode:
            zero vectors imply invalid embedding generation upstream.
            event_ids, if provided, are included in the error to identify
            the offending observations.
        """

        x = np.asarray(x, dtype=np.float32)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        zero_mask = (norms == 0).ravel()

        if np.any(zero_mask):
            zero_positions = np.where(zero_mask)[0].tolist()
            if event_ids is not None:
                offending = [
                    int(np.asarray(event_ids)[i]) for i in zero_positions
                ]
                raise ValueError(
                    f"Zero vector encountered during normalisation at batch "
                    f"positions {zero_positions}, event_ids={offending}. "
                    f"This indicates invalid embedding generation upstream."
                )
            else:
                raise ValueError(
                    f"Zero vector encountered during normalisation at batch "
                    f"positions {zero_positions}. "
                    f"This indicates invalid embedding generation upstream."
                )
        return x / norms


    def add(self, vectors: np.ndarray, event_ids: Sequence[int]) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        ids     = np.ascontiguousarray(event_ids, dtype=np.int64)

        if vectors.ndim != 2:
            raise ValueError("vectors must have shape (n, dim)")

        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"vector dim mismatch: expected {self.dim}, got {vectors.shape[1]}"
            )

        if vectors.shape[0] != ids.shape[0]:
            raise ValueError("number of vectors must match number of event IDs")

        if np.any(ids == -1):
            raise ValueError("Invalid FAISS ids (-1) detected")

        # Guard against within-batch duplicates
        seen = set()
        for eid in ids:
            eid = int(eid)
            if eid in seen:
                raise ValueError(f"Duplicate event_id in batch: {eid}. Did you mean to run with the argument --clear?")
            seen.add(eid)

        # Guard against cross-call duplicates
        if self._index.ntotal > 0:
            cross_dupes = [int(eid) for eid in ids if int(eid) in self._ids]
            if cross_dupes:
                raise ValueError(
                    f"event_ids already present in index: {cross_dupes[:10]}"
                    f"{'...' if len(cross_dupes) > 10 else ''}"
                )

        vectors = self._normalize(vectors, event_ids=ids)
        self._index.add_with_ids(vectors, ids)
        self._ids.update(int(i) for i in ids)


    def ids(self) -> set[int]:
        return self._ids


    def search(
        self,
        queries: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search nearest semantic neighbours.

        Returns:
            similarities, event_ids

        similarities:
            cosine similarity scores in descending order

        event_ids:
            semantic event identifiers corresponding to neighbours
        """

        queries = np.asarray(queries, dtype=np.float32)

        if queries.ndim == 1:
            queries = queries[None, :]

        if queries.shape[1] != self.dim:
            raise ValueError(
                f"query dim mismatch: expected {self.dim}, got {queries.shape[1]}"
            )

        queries = self._normalize(queries)

        scores, ids = self._index.search(queries, k)

        return scores, ids

    @property
    def ntotal(self) -> int:
        """
        Number of indexed semantic events.
        """
        return self._index.ntotal

    def reconstruct(self, event_id: int) -> np.ndarray:
        """
        Retrieve the unit-normalised vector stored for event_id.

        Returns:
            (dim,) float32 array — the L2-normalised vector as it was
            inserted, i.e. suitable for direct inner-product comparison.

        Index-type constraint:
            This method is only supported when the underlying base index is
            IndexFlatIP, which stores vectors verbatim. It will raise a
            RuntimeError for IndexHNSWFlat, which discards vectors after
            construction and does not support reconstruction.

            Before calling this method, callers should confirm the index
            type via isinstance(self._index.index, faiss.IndexFlatIP).

        Intended use (deferred migration):
            ZarrEventLookup in tier2_concept_neighbours.py currently stores
            a copy of every embedding in its by_event_id dict, holding the
            full corpus embedding matrix in memory. Once the index type is
            confirmed stable as IndexFlatIP, that "embedding" field can be
            dropped and replaced with calls to this method, eliminating the
            duplicate copy. The migration is deferred because switching to
            IndexHNSWFlat would silently break any caller relying on
            reconstruct().

        Note:
            Reconstructed vectors are the normalised form stored in FAISS,
            not the raw pre-normalisation vectors from Zarr. For cosine
            similarity purposes these are equivalent, but for any use case
            that requires the original unnormalised embedding, Zarr remains
            the authoritative source.
        """

        base = self._index.index
        if not hasattr(self._index, "reconstruct"):
            raise RuntimeError(
                f"FAISS index {type(self._index).__name__} does not support "
                f"reconstruction."
            )
        vec = np.zeros(self.dim, dtype=np.float32)
        self._index.reconstruct(int(event_id), vec)
        if not np.isfinite(vec).all():
            raise ValueError(f"Invalid reconstructed vector for {event_id}")
        return vec


    def save(self, path: Path) -> None:
        """
        Persist FAISS index.

        Persistence invariant:
            metric geometry must survive round-trip.
        """
        path = Path(path)
        # logger.debug(f"[faiss] saving index={path} ntotal={self._index.ntotal}")
        faiss.write_index(self._index, str(path))


    @classmethod
    def _load_paths(
        cls,
        paths_by_year,
        workers=6,
    ):
        jobs = [
            (year, scale, path)
            for year, scales in paths_by_year.items()
            for scale, path in scales.items()
        ]

        logger.info(
            f"[faiss] loading {len(jobs)} indices with {workers} workers"
        )

        indexes = {
            year: {}
            for year in paths_by_year
        }

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(cls.load, path):
                (year, scale)
                for year, scale, path in jobs
            }

            for future in as_completed(futures):
                year, scale = futures[future]
                indexes[year][scale] = future.result()

        for year, scales in indexes.items():
            for scale, index in scales.items():
                if index.ntotal == 0:
                    raise RuntimeError(
                        f"Empty FAISS index: {year}/{scale}"
                    )

        return indexes


    @classmethod
    def load(cls, path: Path) -> "EeboFaissIndex":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"FAISS index not found: {path}")
        # logger.debug(f"[faiss] loading index={path}")

        obj = cls(dim=1)
        obj._index = faiss.read_index(str(path))

        # logger.debug(
        #     f"[faiss] loaded wrapper={type(obj._index).__name__} "
        #     f"base={type(obj._index.index).__name__}"
        # )

        # logger.debug(
        #     f"[faiss] reconstruct capability: "
        #     f"wrapper={hasattr(obj._index, 'reconstruct')} "
        #     f"base={hasattr(obj._index.index, 'reconstruct')}"
        # )

        if not isinstance(obj._index, faiss.IndexIDMap2):
            raise TypeError("Loaded FAISS index must be IndexIDMap2 (semantic IDs are required)")

        base = obj._index.index
        if not hasattr(base, "metric_type"):
            raise TypeError(f"Cannot determine metric type for index: {type(base)}")
        if base.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise TypeError("FAISS index must use INNER_PRODUCT (cosine similarity invariant)")

        if hasattr(base, "d"):
            obj.dim = base.d
        else:
            raise TypeError(
                f"Cannot infer embedding dimension from FAISS index of type "
                f"{type(base).__name__}."
            )

        obj._ids = set(int(i) for i in faiss.vector_to_array(obj._index.id_map).tolist())  # NEW

        # logger.debug(f"[faiss] loaded ntotal={obj._index.ntotal} dim={obj.dim}")
        return obj


    @classmethod
    def load_range( cls, years, masked=False, workers=6 ):
        paths_by_year = {
            year: faiss_index_paths( masked=masked, year=year )
            for year in years
        }
        return cls._load_paths( paths_by_year, workers=workers, )


    @classmethod
    def load_existing_range( cls, start_year, end_year, masked=False, workers=6, ):
        years = [
            year
            for year in discover_index_years(masked)
            if start_year <= year <= end_year
        ]
        if not years:
            raise RuntimeError( f"No FAISS indices found between {start_year}-{end_year}" )
        paths_by_year = {
            year: faiss_index_paths( masked=masked, year=year, )
            for year in years
        }
        return cls._load_paths( paths_by_year, workers=workers, )


    @classmethod
    def load_all( cls, masked=False, workers=6, ):
        paths_by_year = {
            year: faiss_index_paths(
                masked=masked,
                year=year,
            )
            for year in discover_index_years(masked)
        }
        if not paths_by_year:
            raise RuntimeError("No FAISS indices found")
        return cls._load_paths( paths_by_year, workers=workers, )


    def reconstruct_many(
        self,
        event_ids: Sequence[int],
    ) -> np.ndarray:
        """
        Reconstruct multiple vectors from the index.

        Returns
        -------
        (N,D) float32 array aligned with event_ids.

        Failure modes
        -------------
        Raises if any requested event_id is absent from the index.

        This method exists so downstream algorithms never need to know
        anything about FAISS reconstruction semantics.
        """

        X = np.empty((len(event_ids), self.dim), dtype=np.float32)

        for i, eid in enumerate(event_ids):
            if int(eid) not in self._ids:
                logger.error(
                    "[faiss] index size=%d",
                    len(self._ids),
                )

                logger.error(
                    "[faiss] missing ids sample=%s",
                    list(map(int, event_ids[:10])),
                )

                logger.error(
                    "[faiss] index sample ids=%s",
                    list(sorted(self._ids))[:10],
                )

                logger.error(
                    "[faiss] index size=%d min=%d max=%d",
                    len(self._ids),
                    min(self._ids),
                    max(self._ids),
                )

                raise KeyError(
                    f"FAISS missing event_id={eid}. "
                    f"Index contains {len(self._ids)} ids."
                )

            self._index.reconstruct(int(eid), X[i])

        return X


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k: int = 60,
    top_n: int | None = None,
) -> list[tuple[int, float]]:
    """
    Fuse multiple ranked neighbour-id lists (e.g. local/medium/broad FAISS
    results for one query) via Reciprocal Rank Fusion:
        score(id) = sum over lists containing id of 1 / (k + rank)
    rank is 1-indexed. -1 (FAISS's "no result") entries are ignored.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, eid in enumerate(ranked, start=1):
            if eid == -1:
                continue
            scores[eid] = scores.get(eid, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return fused[:top_n] if top_n else fused


def _merge_topk_across_years(
    year_results: list[tuple[np.ndarray, np.ndarray]],
    search_k: int,
    n_queries: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge per-year (scores, ids) search results into a single top-search_k
    per query, for one scale.

    Each entry in year_results is (scores, ids) as returned by
    EeboFaissIndex.search() for one year's index at this scale, shape
    (n_queries, search_k). Since event_ids are globally unique (stable-hashed
    from doc_id/token_idx/window position, not scoped per year), there's no
    cross-year collision risk here — merging is a straightforward top-k
    re-rank over the concatenated candidates.
    """
    merged_scores = np.full((n_queries, search_k), -np.inf, dtype=np.float32)
    merged_ids = np.full((n_queries, search_k), -1, dtype=np.int64)

    for i in range(n_queries):
        candidates = []
        for scores, ids in year_results:
            for s, eid in zip(scores[i], ids[i]):
                eid = int(eid)
                if eid != -1:
                    candidates.append((float(s), eid))

        candidates.sort(key=lambda x: x[0], reverse=True)
        top = candidates[:search_k]

        for j, (s, eid) in enumerate(top):
            merged_scores[i, j] = s
            merged_ids[i, j] = eid

    return merged_scores, merged_ids


def multiscale_search(
    index: dict[int, dict[str, "EeboFaissIndex"]],
    lookup,
    positions,
    top_n: int,
    pub_year: int | None = None,
    rrf_k: int = 60,
    oversample: int = 3,
) -> list[list[dict]]:
    """
    Search local/medium/broad FAISS indices for the queries at `positions`,
    fuse the three ranked lists per query via RRF.

    `index` is keyed as index[year][scale] -> EeboFaissIndex, matching the
    per-year, per-scale layout produced by the old build_indices.py.

    pub_year:
        If given, search only that year's three scale indices.
        If None (default), search every year's indices per scale and merge
        each scale's results into a single top-`search_k` list per query
        before fusion — i.e. an unscoped, corpus-wide search, same behaviour
        as before per-year partitioning was introduced.

        NOTE: unscoped search issues 3 * len(index) FAISS calls per batch of
        queries (one per scale per year). Fine for the current year count;
        if the corpus grows to cover many centuries this may want a
        `year_range` param instead of scanning every year — deferred until
        that's actually needed.

    Returns a list aligned with `positions`; each entry is a list of dicts:
        {
            "event_id":     int,
            "rrf_score":    float,
            "score_local":  float | None,
            "score_medium": float | None,
            "score_broad":  float | None,
        }
    truncated to top_n, ordered by rrf_score descending.
    """
    search_k = top_n * oversample
    scales = ("local", "medium", "broad")
    n_queries = len(positions)

    if pub_year is not None:
        if pub_year not in index:
            raise KeyError(
                f"No index found for pub_year={pub_year}. "
                f"Available years: {sorted(index.keys())}"
            )
        years_to_search = [pub_year]
    else:
        years_to_search = sorted(index.keys())

    per_scale = {}
    for scale in scales:
        queries = getattr(lookup, f"emb_{scale}")[positions]

        if len(years_to_search) == 1:
            per_scale[scale] = index[years_to_search[0]][scale].search(queries, search_k)
        else:
            year_results = [
                index[year][scale].search(queries, search_k)
                for year in years_to_search
                if scale in index[year]
            ]
            per_scale[scale] = _merge_topk_across_years(year_results, search_k, n_queries)

    fused = []
    for i in range(n_queries):
        # id -> raw cosine score, in rank order, per scale
        scale_scores = {
            scale: {
                int(nid): float(score)
                for nid, score in zip(per_scale[scale][1][i], per_scale[scale][0][i])
                if int(nid) != -1
            }
            for scale in scales
        }
        ranked_lists = [list(scale_scores[scale].keys()) for scale in scales]
        fused_ids = reciprocal_rank_fusion(ranked_lists, k=rrf_k, top_n=top_n)

        fused.append([
            {
                "event_id":     eid,
                "rrf_score":    rrf_score,
                "score_local":  scale_scores["local"].get(eid),
                "score_medium": scale_scores["medium"].get(eid),
                "score_broad":  scale_scores["broad"].get(eid),
            }
            for eid, rrf_score in fused_ids
        ])
    return fused

