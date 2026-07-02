#!/usr/bin/env python
"""
tier2_0_concept_events.py - Tier 2: Concept Neighbourhood Analysis

This module performs the first semantic analysis over the Tier 1 observation
space. It uses the FAISS retrieval index to identify contextual
neighbourhoods around lexical concepts while preserving complete provenance
back to the original corpus observations.

Architecture
------------

Tier 1

    corpus
        →
    semantic observations
        ├── metadata
        ├── emb_local
        ├── emb_medium
        └── emb_broad

            │

Tier 1.5

    weighted ensemble embeddings
            │
            ▼
        FAISS index

            │

Tier 2

    concept forms
            │
            ▼
    matching observations
            │
            ▼
    neighbourhood retrieval
            │
            ▼
    contextual concept analysis
            │
            ▼
    SQLite analysis database

Tier 2 performs no embedding generation. It analyses the observation geometry
constructed by earlier tiers.

Core invariants
---------------

1. Tier 1 is the semantic source of truth
   - All metadata and embeddings originate from the Tier 1 observation store.

2. FAISS is a retrieval layer only
   - Neighbourhoods are determined geometrically without assigning semantic
     interpretation.

3. Observations remain atomic
   - Every result corresponds to a single corpus-grounded contextual
     observation identified by a stable event_id.

4. Lexical identity is independent of observation identity
   - Multiple observations may share the same vector_id while representing
     different contextual occurrences.

5. Provenance is never lost
   - Every neighbour can be traced back to its document, token position and
     contextual window.

6. Multi-scale representations are analysed through an ensemble embedding
   - Queries use the weighted combination of local, medium and broad
     embeddings generated in Tier 1.

Performance model
-----------------

Tier 1 observations are streamed from Zarr into an in-memory
struct-of-arrays lookup optimised for neighbourhood analysis.

When analysing a single concept, only observations matching the requested
forms are loaded, making memory consumption proportional to the concept
rather than the corpus.

Storage model
-------------

Metadata are stored as parallel NumPy arrays indexed by row position,
together with an event_id → row lookup.

The lookup contains:

- observation metadata
- aligned local embeddings
- aligned medium embeddings
- aligned broad embeddings

Ensemble embeddings are computed on demand rather than materialised
separately.

This design provides good cache locality while avoiding millions of small
Python objects.

Neighbourhood model
-------------------

Each query observation is searched against the global FAISS index using its
ensemble embedding.

Neighbourhoods may be expanded to two levels:

- depth 1: direct semantic neighbours
- depth 2: neighbours-of-neighbours

Both levels retain full provenance and explicitly record the path through
which secondary neighbours were discovered.

Outputs
-------

Results are written to a normalised SQLite database containing:

- concepts
- concept_forms
- query observations
- neighbourhood relationships
- aggregate statistics
- document metadata

The database is intended as the persistent semantic substrate for later
visualisation, clustering and diachronic analysis.

Design intent
-------------

Tier 2 intentionally performs neighbourhood analysis rather than concept
modelling. It establishes the local semantic geometry surrounding lexical
concepts while leaving higher-level interpretation—such as clustering,
semantic field induction, temporal comparison and semantic drift—to later
tiers.

This separation keeps retrieval, neighbourhood construction and semantic
interpretation as distinct stages of the pipeline, allowing each to evolve
independently without compromising provenance or reproducibility.
"""

from __future__ import annotations

import os
import argparse
import sqlite3
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import zarr

from lib.eebo_config import CONCEPT_SETS, INDEXES_DIR, FAISS_TIER1_INDEX, ZARR_ROOT, OUT_DIR, CORPUS_TIER2_DB_PATH
from lib.eebo_faiss import EeboFaissIndex
from lib.eebo_logging import logger, setEmit
from lib.concept_resolve import resolve_concepts
from lib.eebo_db import get_connection
from lib.zarr_store_dirs import store_dirs
from lib.tier2_diagnostics import (
    audit_embedding_diversity,
    audit_embedding_isotropy,
    audit_hubness,
    audit_neighbour_identity,
    audit_knn_stability,
    knn_diagnostics,
)

K           = 25
BATCH_SIZE  = 8192

# Sentinel for absent window_token_pos in the int64 column. -1 is never a
# valid token position, so it is unambiguous as "not present".
_NO_WPOS = -1


# Event lookup
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

    # Field name -> numpy dtype for the metadata columns (excludes
    # embeddings, which are handled separately as a 2D float32 array).
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
        # Per-batch staging lists, concatenated into arrays in _finalize.
        self._chunks: dict[str, list[np.ndarray]] = {field: [] for field in self._FIELDS}
        # Multi-scale embedding chunks
        self._emb_local_chunks  = []
        self._emb_medium_chunks = []
        self._emb_broad_chunks  = []
        self._build()


    def _build(self):
        logger.info("[tier2] building event lookup")
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
        Load events from one Zarr store into staging columns.
        Supports multi-scale embeddings
        """
        if "event_id" not in e:
            raise KeyError(f"Missing event_id in {store_dir} - rebuild Tier 1")

        wpos = e["window_token_pos"] if "window_token_pos" in e else None
        n = e["event_id"].shape[0]

        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)

            # Load metadata
            b_eids = e["event_id"][start:end]
            b_vids = e["vector_id"][start:end]
            b_docs = e["doc_id"][start:end]
            b_toks = e["token"][start:end]
            b_idxs = e["token_idx"][start:end]
            b_wins = e["window_id"][start:end]
            b_wpos = wpos[start:end] if wpos is not None else None

            # Multi-scale embeddings
            b_local  = e["emb_local"][start:end]
            b_medium = e["emb_medium"][start:end]
            b_broad  = e["emb_broad"][start:end]

            b_toks = b_toks.astype(str)
            b_docs = b_docs.astype(str)
            b_toks_lower = np.char.lower(b_toks)

            # Filtering
            if self.forms is not None:
                keep = np.isin(b_toks_lower, list(self.forms))
            else:
                keep = np.ones(end - start, dtype=bool)

            if self.false_positives:
                keep &= ~np.isin(b_toks_lower, list(self.false_positives))

            if not keep.any():
                continue

            keep_count = int(keep.sum())

            # Append metadata
            self._chunks["event_id"].append(np.asarray(b_eids, dtype=np.int64)[keep])
            self._chunks["vector_id"].append(np.asarray(b_vids, dtype=np.int64)[keep])
            self._chunks["doc_id"].append(b_docs[keep])
            self._chunks["token"].append(b_toks[keep])
            self._chunks["token_idx"].append(np.asarray(b_idxs, dtype=np.int64)[keep])
            self._chunks["window_id"].append(np.asarray(b_wins, dtype=np.int64)[keep])

            if b_wpos is not None:
                wpos_col = np.asarray(b_wpos, dtype=np.int64)[keep]
            else:
                wpos_col = np.full(keep_count, _NO_WPOS, dtype=np.int64)
            self._chunks["window_token_pos"].append(wpos_col)

            # Multi-scale embeddings
            self._emb_local_chunks.append(np.asarray(b_local, dtype=np.float32)[keep])
            self._emb_medium_chunks.append(np.asarray(b_medium, dtype=np.float32)[keep])
            self._emb_broad_chunks.append(np.asarray(b_broad, dtype=np.float32)[keep])


    def _finalize(self):
        """
        Concatenate all staged chunks.
        """
        n_total = sum(arr.shape[0] for arr in self._chunks["event_id"])

        if n_total == 0:
            for field, dtype in self._FIELDS.items():
                setattr(self, field, np.empty(0, dtype=dtype))
            self.emb_local  = np.empty((0, 768), dtype=np.float32)  # adjust dim if needed
            self.emb_medium = np.empty((0, 768), dtype=np.float32)
            self.emb_broad  = np.empty((0, 768), dtype=np.float32)
            return

        # Metadata
        for field, dtype in self._FIELDS.items():
            setattr(self, field, np.concatenate(self._chunks[field]).astype(dtype, copy=False))

        # Embeddings
        self.emb_local  = np.concatenate(self._emb_local_chunks, axis=0)
        self.emb_medium = np.concatenate(self._emb_medium_chunks, axis=0)
        self.emb_broad  = np.concatenate(self._emb_broad_chunks, axis=0)

        self._pos = {int(eid): pos for pos, eid in enumerate(self.event_id)}

        # Cleanup
        self._chunks.clear()
        self._emb_local_chunks.clear()
        self._emb_medium_chunks.clear()
        self._emb_broad_chunks.clear()

        logger.info(f"[tier2] loaded {n_total:,} events with multi-scale embeddings")


    def get_ensemble_embedding(self, pos: int, weights=None) -> np.ndarray:
        """Return weighted ensemble embedding for a given event position."""
        if weights is None:
            weights = [0.25, 0.50, 0.25]  # local, medium, broad

        return (
            weights[0] * self.emb_local[pos] +
            weights[1] * self.emb_medium[pos] +
            weights[2] * self.emb_broad[pos]
        )

    def get_event(self, event_id: int) -> dict:
        """
        Return a dict for one event, in the same shape as the previous
        per-event dict representation (event_id, vector_id, doc_id, token,
        token_idx, window_id, window_token_pos, embedding).

        Kept for compatibility with code that wants a single event as a
        dict (e.g. logging, one-off lookups). Hot loops should prefer
        get_pos() plus direct array access to avoid per-call dict
        allocation — see analyse_concept.
        """
        pos = self._pos[int(event_id)]
        event = self._row_to_dict(pos)
        event["embedding"] = self.get_ensemble_embedding(pos)   # default to ensemble
        return event


    def get_pos(self, event_id: int) -> int:
        """event_id -> row position. Raises KeyError if not present."""
        return self._pos[int(event_id)]


    def _row_to_dict(self, pos: int) -> dict:
        """Convert a row position into the legacy dict format."""
        wpos = int(self.window_token_pos[pos])

        return {
            "event_id":         int(self.event_id[pos]),
            "vector_id":        int(self.vector_id[pos]),
            "doc_id":           str(self.doc_id[pos]),
            "token":            str(self.token[pos]),
            "token_idx":        int(self.token_idx[pos]),
            "window_id":        int(self.window_id[pos]),
            "window_token_pos": None if wpos == _NO_WPOS else wpos,
            # "embedding" is now added in get_event() using ensemble
        }


    def iter_matching_event_ids(self, forms, false_positives=None):
        """
        Yield event_ids matching forms).
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


# Document metadata
def load_doc_metadata(conn) -> dict:
    """
    Build doc_id -> metadata mapping from documents + place_normalization.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT ON (d.doc_id)
            d.doc_id,
            d.pub_year,
            d.title,
            pn.normalized_places,
            pn.geom,
            ST_Y(pn.geom::geometry) AS lat,
            ST_X(pn.geom::geometry) AS lng
        FROM documents d
        LEFT JOIN place_normalization pn ON d.pub_place = pn.raw_place
        ORDER BY d.doc_id
    """)

    out = {}
    for doc_id, year, title, places, geom, lat, lng in cur.fetchall():
        out[doc_id] = {
            "pub_year": year,
            "title": title,
            "places": places,
            "geom": geom,   # ST_Point
            "lat": lat,
            "lng": lng,
        }

    return out


# Concept analysis
def analyse_concept(
    doc_meta, index, lookup, concept_name, concept,
    *,
    diagnostics     = False,
    false_positives = None,
    top_n: int      = K,
    depth: int      = 1,
):
    """
    Compute neighbourhood structure for all events matching a concept.

    Supports both legacy "window" events and new "clause_complex" events.

    Parameters
    ----------
    depth : int
        1 = direct FAISS neighbours only (original behaviour)
        2 = also include neighbours-of-neighbours (depth-2 expansion)

    Returns
    -------
    dict with keys:
        - concept, forms, n_events
        - aggregate (top tokens/docs/windows)
        - events: list of dicts containing:
            * event metadata
            * neighbours: list of neighbour dicts (same structure)
    """
    # Column references from lookup (struct-of-arrays)
    L_event_id      = lookup.event_id
    L_vector_id     = lookup.vector_id
    L_doc_id        = lookup.doc_id
    L_token         = lookup.token
    L_token_idx     = lookup.token_idx
    L_window_id     = lookup.window_id
    L_wpos          = lookup.window_token_pos

    # CONCEPT_SET union with any args supplied to this routine
    forms           = set(concept["forms"])
    forms           = {
        f.lower()
        for f in (
            list(forms or [])
            + list(concept.get("forms", []))
        )
    }

    false_positives = {
        f.lower()
        for f in (
            list(false_positives or [])
            + list(concept.get("false_positives", []))
        )
    }

    event_ids = list(lookup.iter_matching_event_ids(forms, false_positives))

    logger.info(f"[tier2] concept={concept_name}")
    if false_positives:
        logger.info(f"[tier2] excluding false_positives={false_positives}")
    if not event_ids:
        return {"concept": concept_name, "empty": True}

    event_pos = np.fromiter(
        (lookup.get_pos(eid) for eid in event_ids),
        dtype=np.int64,
        count=len(event_ids),
    )

    query_vecs = np.array([lookup.get_ensemble_embedding(p) for p in event_pos])

    if diagnostics:
        logger.debug(f"[tier2] query_events={len(event_ids)}")
        logger.debug(f"[tier2] sample_event_id={event_ids[0]}")
        logger.debug(f"[tier2] sample_embedding_shape={query_vecs.shape}")

    all_scores, all_neigh_ids = index.search(query_vecs, K)

    if diagnostics:
        audit_embedding_diversity(concept_name, query_vecs)
        audit_embedding_isotropy(query_vecs)
        audit_hubness(index, query_vecs, k=K)
        audit_neighbour_identity(all_neigh_ids)
        audit_knn_stability(index, lookup, event_ids, k=K)

    token_counter  = Counter()
    doc_counter    = Counter()
    window_counter = Counter()
    results        = []

    # Set of original query event_ids — used to exclude self-matches at
    # depth 2 without repeated list membership tests.
    query_event_id_set = set(event_ids)

    for i, eid in enumerate(event_ids):
        q_pos = int(event_pos[i])
        q_doc_id = str(L_doc_id[q_pos])
        q_pub_year = doc_meta.get(q_doc_id, {}).get("pub_year")

        q_meta  = doc_meta.get(q_doc_id, {})
        q_geom  = q_meta.get("geom")
        q_lat   = q_meta.get("lat")
        q_lng   = q_meta.get("lng")
        q_wpos  = int(L_wpos[q_pos])

        neighbours = []

        # Depth 1 neighbours
        for nid, score in zip(all_neigh_ids[i], all_scores[i]):
            nid_int = int(nid)
            if nid_int == -1 or nid_int == eid:
                continue
            n_pos = lookup.get_pos(nid_int)
            n_token = str(L_token[n_pos])
            if n_token.lower() in false_positives:
                continue

            n_doc_id = str(L_doc_id[n_pos])
            n_geom = doc_meta.get(n_doc_id, {}).get("geom")
            n_lat = doc_meta.get(n_doc_id, {}).get("lat")
            n_lng = doc_meta.get(n_doc_id, {}).get("lng")
            n_window_id = int(L_window_id[n_pos])
            n_wpos = int(L_wpos[n_pos])

            token_counter[n_token] += 1
            doc_counter[n_doc_id] += 1
            window_counter[(n_doc_id, n_window_id)] += 1

            neighbours.append({
                "event_id":         int(L_event_id[n_pos]),
                "vector_id":        int(L_vector_id[n_pos]),
                "token":            n_token,
                "doc_id":           n_doc_id,
                "pub_year":         doc_meta.get(n_doc_id, {}).get("pub_year"),
                "token_idx":        int(L_token_idx[n_pos]),
                "window_id":        n_window_id,
                "window_token_pos": None if n_wpos == _NO_WPOS else n_wpos,
                "score":            float(score),
                "depth":            1,
                "geom":             n_geom,
                "lat":              n_lat,
                "lng":              n_lng,
                "via_event_id":     None,
            })

        # Main query event
        results.append({
            "event_id":         int(L_event_id[q_pos]),
            "vector_id":        int(L_vector_id[q_pos]),
            "token":            str(L_token[q_pos]),
            "doc_id":           q_doc_id,
            "pub_year":         q_pub_year,
            "token_idx":        int(L_token_idx[q_pos]),
            "window_id":        int(L_window_id[q_pos]),
            "window_token_pos": None if q_wpos == _NO_WPOS else q_wpos,
            "geom":             q_geom,
            "lat":              q_lat,
            "lng":              q_lng,
            "neighbours":       neighbours,
        })

        q_pos = int(event_pos[i])
        q_doc_id = str(L_doc_id[q_pos])
        q_pub_year = doc_meta.get(q_doc_id, {}).get("pub_year")

        q_meta = doc_meta.get(q_doc_id, {})
        q_geom = q_meta.get("geom")
        q_lat = q_meta.get("lat")
        q_lng = q_meta.get("lng")

        neighbours = []

        for nid, score in zip(all_neigh_ids[i], all_scores[i]):
            # Skip no match or self-matches:
            nid_int = int(nid)
            if nid_int == -1 or nid_int == eid:
                continue
            n_pos                                   = lookup.get_pos(int(nid))
            n_token                                 = str(L_token[n_pos])
            if n_token.lower() in false_positives:
                continue
            n_doc_id                                = str(L_doc_id[n_pos])
            n_geom                                  = doc_meta.get(n_doc_id, {}).get("geom")
            n_lat                                   = doc_meta.get(n_doc_id, {}).get("lat")
            n_lng                                   = doc_meta.get(n_doc_id, {}).get("lng")
            n_window_id                             = int(L_window_id[n_pos])
            token_counter[n_token]                  += 1
            doc_counter[n_doc_id]                   += 1
            window_counter[(n_doc_id, n_window_id)] += 1
            n_wpos                                  = int(L_wpos[n_pos])
            neighbours.append({
                "event_id":         int(L_event_id[n_pos]),
                "vector_id":        int(L_vector_id[n_pos]),
                "token":            n_token,
                "doc_id":           n_doc_id,
                "pub_year":         doc_meta.get(n_doc_id, {}).get("pub_year"),
                "token_idx":        int(L_token_idx[n_pos]),
                "window_id":        n_window_id,
                "window_token_pos": None if n_wpos == _NO_WPOS else n_wpos,
                "score":            float(score),
                "depth":            1,
                "geom":             n_geom,
                "lat":              n_lat,
                "lng":              n_lng,
                "via_event_id":     None,
            })

        q_wpos = int(L_wpos[q_pos])

        results.append({
            "event_id":         int(L_event_id[q_pos]),
            "vector_id":        int(L_vector_id[q_pos]),
            "token":            str(L_token[q_pos]),
            "doc_id":           q_doc_id,
            "pub_year":         q_pub_year,
            "token_idx":        int(L_token_idx[q_pos]),
            "window_id":        int(L_window_id[q_pos]),
            "window_token_pos": None if q_wpos == _NO_WPOS else q_wpos,
            "geom":             q_geom,
            "lat":              q_lat,
            "lng":              q_lng,
            "neighbours":       neighbours,
        })

    # Depth-2: neighbours-of-neighbours
    if depth >= 2:
        # Collect the unique set of depth-1 neighbour event_ids across all
        # query events, excluding the original query events themselves.
        d1_event_id_set: set[int] = set()
        for res in results:
            for n in res["neighbours"]:
                d1_event_id_set.add(n["event_id"])
        d1_event_id_set -= query_event_id_set

        # Build a reverse map: depth-1 event_id -> list of query event_ids
        # that hold it as a direct neighbour. Used to fan depth-2 results
        # back out to the originating query events.
        via_to_queries: dict[int, list[int]] = {}
        for res in results:
            for n in res["neighbours"]:
                via_eid = n["event_id"]
                if via_eid in d1_event_id_set:
                    via_to_queries.setdefault(via_eid, []).append(res["event_id"])

        # Also build a per-query-event set of its depth-1 neighbour ids so
        # we can skip events already seen at depth 1 for that query event.
        query_d1_neighbours: dict[int, set[int]] = {
            res["event_id"]: {n["event_id"] for n in res["neighbours"]}
            for res in results
        }

        # Index results by query event_id for O(1) append access.
        results_by_query: dict[int, dict] = {res["event_id"]: res for res in results}

        if d1_event_id_set:
            d1_ids_list  = list(d1_event_id_set)
            d1_positions = np.array(
                [lookup.get_pos(eid) for eid in d1_ids_list if eid in lookup._pos],
                dtype=np.int64,
            )
            # Filter d1_ids_list to only those present in the lookup
            # (get_pos above would raise for missing ids; guard here).
            d1_ids_list = [eid for eid in d1_ids_list if eid in lookup._pos]

            if d1_positions.size > 0:
                d1_vecs = np.array([lookup.get_ensemble_embedding(p) for p in d1_positions])
                d2_scores, d2_neigh_ids = index.search(d1_vecs, K)

                logger.info(
                    f"[tier2] depth-2 search: via_events={len(d1_ids_list)}"
                )

                for i, via_eid in enumerate(d1_ids_list):
                    query_eids_for_via = via_to_queries.get(via_eid, [])
                    if not query_eids_for_via:
                        continue

                    for nid, score in zip(d2_neigh_ids[i], d2_scores[i]):
                        nid_int = int(nid)
                        if nid_int == -1:
                            continue
                        # Exclude original query events and depth-1 neighbours
                        # (checked per query event below) and unknown ids.
                        if nid_int in query_event_id_set:
                            continue
                        if nid_int not in lookup._pos:
                            continue

                        n_pos    = lookup.get_pos(nid_int)
                        n_token  = str(L_token[n_pos])
                        if n_token.lower() in false_positives:
                            continue

                        n_doc_id    = str(L_doc_id[n_pos])
                        n_window_id = int(L_window_id[n_pos])
                        n_wpos      = int(L_wpos[n_pos])

                        neighbour_record = {
                            "event_id":         int(L_event_id[n_pos]),
                            "vector_id":        int(L_vector_id[n_pos]),
                            "token":            n_token,
                            "doc_id":           n_doc_id,
                            "pub_year":         doc_meta.get(n_doc_id, {}).get("pub_year"),
                            "token_idx":        int(L_token_idx[n_pos]),
                            "window_id":        n_window_id,
                            "window_token_pos": None if n_wpos == _NO_WPOS else n_wpos,
                            "score":            float(score),
                            "depth":            2,
                            "via_event_id":     via_eid,
                        }

                        # Fan out to every query event that had via_eid as
                        # a depth-1 neighbour, skipping if this event_id is
                        # already a depth-1 neighbour of that query event.
                        for q_eid in query_eids_for_via:
                            if nid_int in query_d1_neighbours.get(q_eid, set()):
                                continue
                            results_by_query[q_eid]["neighbours"].append(neighbour_record)

    return {
        "concept":   concept_name,
        "forms":     forms,
        "n_events":  len(event_ids),
        "aggregate": {
            "top_tokens":  token_counter.most_common(top_n),
            "top_docs":    doc_counter.most_common(top_n),
            "top_windows": window_counter.most_common(top_n),
        },
        "events": results,
    }



_SCHEMA_INIT = """
CREATE TABLE IF NOT EXISTS concepts (
    concept  TEXT PRIMARY KEY,
    n_events INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS concept_forms (
    concept           TEXT NOT NULL,
    form              TEXT NOT NULL,
    is_false_positive INTEGER DEFAULT 0,
    PRIMARY KEY (concept, form),
    FOREIGN KEY (concept) REFERENCES concepts(concept)
);

CREATE TABLE IF NOT EXISTS events (
    event_id         INTEGER PRIMARY KEY,
    concept          TEXT    NOT NULL,
    vector_id        INTEGER,
    token            TEXT,
    doc_id           TEXT,
    pub_year         INTEGER,
    token_idx        INTEGER,
    window_id        INTEGER,
    window_token_pos INTEGER,
    geom             TEXT,
    lat              NUMBER,
    lng              NUMBER,
    nx               REAL,
    ny               REAL,
    gnx              REAL,
    gny              REAL,
    cluster_id       INTEGER,
    cluster_label    TEXT,
    FOREIGN KEY (concept) REFERENCES concepts(concept)
);

CREATE TABLE IF NOT EXISTS neighbours (
    event_id            INTEGER NOT NULL,
    neighbour_event_id  INTEGER NOT NULL,
    depth               INTEGER NOT NULL DEFAULT 1,
    via_event_id        INTEGER,
    vector_id           INTEGER,
    token               TEXT,
    doc_id              TEXT,
    pub_year            INTEGER,
    token_idx           INTEGER,
    window_id           INTEGER,
    window_token_pos    INTEGER,
    score               REAL,
    geom                TEXT,
    lat                 NUMBER,
    lng                 NUMBER,
    nx                  REAL,
    ny                  REAL,
    gnx                 REAL,
    gny                 REAL,
    cluster_id          INTEGER,
    cluster_label       TEXT,
    PRIMARY KEY (event_id, neighbour_event_id, depth),
    FOREIGN KEY (event_id) REFERENCES events(event_id)
);

CREATE INDEX IF NOT EXISTS events_geom_idx ON events(geom);
CREATE INDEX IF NOT EXISTS neighbours_geom_idx ON neighbours(geom);

CREATE INDEX IF NOT EXISTS events_lat_idx ON events(lat);
CREATE INDEX IF NOT EXISTS events_lng_idx ON events(lng);
CREATE INDEX IF NOT EXISTS neighbours_lat_idx ON neighbours(lat);
CREATE INDEX IF NOT EXISTS neighbours_lng_idx ON neighbours(lng);
CREATE INDEX IF NOT EXISTS idx_events_concept_pubyear_nx ON events(concept, pub_year, nx, ny, gnx, gny);

CREATE TABLE concept_projection_bounds (
    concept   TEXT NOT NULL,
    local_min_x  REAL, local_max_x  REAL,
    local_min_y  REAL, local_max_y  REAL,
    global_min_x REAL, global_max_x REAL,
    global_min_y REAL, global_max_y REAL,
    PRIMARY KEY (concept)
);

-- Flattened aggregate rows for top_tokens, top_docs, top_windows.
-- kind    = 'token' | 'doc' | 'window'
-- token/doc rows : value = token string or doc_id; window columns NULL
-- window rows    : window_doc_id + window_id set; value NULL
CREATE TABLE IF NOT EXISTS concept_aggregate (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    concept       TEXT    NOT NULL,
    kind          TEXT    NOT NULL,
    rank          INTEGER NOT NULL,
    value         TEXT,
    window_doc_id TEXT,
    window_id     INTEGER,
    count         INTEGER NOT NULL,
    cluster_id    INTEGER,
    FOREIGN KEY (concept) REFERENCES concepts(concept)
);

CREATE INDEX IF NOT EXISTS idx_events_concept       ON events(concept);
CREATE INDEX IF NOT EXISTS idx_events_token         ON events(token);
CREATE INDEX IF NOT EXISTS idx_events_event_id      ON events(event_id);
CREATE INDEX IF NOT EXISTS idx_events_doc_id        ON events(doc_id);
CREATE INDEX IF NOT EXISTS idx_events_concept_year  ON events(concept, pub_year);
CREATE INDEX IF NOT EXISTS idx_neighbours_event_id  ON neighbours(event_id);
CREATE INDEX IF NOT EXISTS idx_neighbours_token     ON neighbours(token);
CREATE INDEX IF NOT EXISTS idx_neighbours_depth     ON neighbours(event_id, depth);
CREATE INDEX IF NOT EXISTS idx_aggregate_concept    ON concept_aggregate(concept, kind);

CREATE TABLE IF NOT EXISTS concept_cluster_info (
    concept        TEXT    NOT NULL,
    cluster_id     INTEGER NOT NULL,
    cluster_label  TEXT,
    centroid_nx    REAL,
    centroid_ny    REAL,
    centroid_gnx   REAL,
    centroid_gny   REAL,
    point_count    INTEGER,
    PRIMARY KEY (concept, cluster_id)
);
"""

_SCHEMA_CLEAR = """
    DROP TABLE IF EXISTS concept_cluster_info;
    DROP TABLE IF EXISTS concept_aggregate;
    DROP TABLE IF EXISTS neighbours;
    DROP TABLE IF EXISTS events;
    DROP TABLE IF EXISTS concepts;
"""

_DELETE_CONCEPT = [
    "DELETE FROM concept_cluster_info WHERE concept = ?",
    "DELETE FROM concept_aggregate WHERE concept = ?",
    "DELETE FROM neighbours WHERE event_id IN (SELECT event_id FROM events WHERE concept = ?)",
    "DELETE FROM events WHERE concept = ?",
    "DELETE FROM concepts WHERE concept = ?",
]


def sqlite3_connection(db_path):
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA busy_timeout=5000;")  # optional but very relevant in FastAPI context
    return con


def load_concept_forms(conn, concept):
    cur = conn.execute(
        "SELECT form, is_false_positive FROM concept_forms WHERE concept = ?",
        (concept,)
    )
    rows = cur.fetchall()

    forms = {r[0].lower() for r in rows if not r[1]}
    fps   = {r[0].lower() for r in rows if r[1]}

    return forms, fps


def write_sqlite(output: dict, db_path, *, clear: bool = False, doc_meta: dict = None):
    """
    Write analyse_concept output to a normalised SQLite database.

    If clear=True, all existing tables are dropped and recreated before
    writing. Use this when rebuilding the full corpus analysis from scratch.

    Otherwise, existing rows for each concept in output are deleted and
    rewritten, leaving all other concepts intact. This is the correct path
    for single-concept runs from the UI.
    """
    logger.debug(f"[tier2] writing sqlite -> {db_path}")

    con = sqlite3_connection(db_path)

    if clear:
        logger.info("[tier2] clearing sqlite database")
        con.executescript(_SCHEMA_CLEAR)

    con.executescript(_SCHEMA_INIT)

    for concept_name, data in output.items():
        if data.get("empty"):
            continue

        con.execute("BEGIN")

        # Remove existing rows for this concept
        for stmt in _DELETE_CONCEPT:
            con.execute(stmt, (concept_name,))

        con.execute(
            "INSERT OR IGNORE INTO concepts (concept, n_events) VALUES (?, ?)",
            (concept_name, data["n_events"]),
        )

        forms = set(data.get("forms", []))
        false_positives = set(data.get("false_positives", []))

        # normal forms
        for form in forms:
            con.execute(
                """
                INSERT INTO concept_forms (concept, form, is_false_positive)
                VALUES (?, ?, 0)
                ON CONFLICT(concept, form)
                DO UPDATE SET is_false_positive = 0
                """,
                (concept_name, form),
            )

        # false positives override
        for form in false_positives:
            con.execute(
                """
                INSERT INTO concept_forms (concept, form, is_false_positive)
                VALUES (?, ?, 1)
                ON CONFLICT(concept, form)
                DO UPDATE SET is_false_positive = 1
                """,
                (concept_name, form),
            )

        con.executemany( """
            INSERT OR IGNORE INTO events
            (event_id, concept, vector_id, token, doc_id, pub_year,
            token_idx, window_id, window_token_pos, geom, lat, lng )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    e["event_id"],
                    concept_name,
                    e["vector_id"],
                    e["token"],
                    e["doc_id"],
                    e["pub_year"],
                    e["token_idx"],
                    e["window_id"],
                    e["window_token_pos"],
                    e.get("geom"),
                    e.get("lat"),
                    e.get("lng"),
                )
                for e in data["events"]
            ],
        )

        # neighbours
        con.executemany("""
            INSERT OR IGNORE INTO neighbours
            (event_id, neighbour_event_id, depth, via_event_id, vector_id,
            token, doc_id, pub_year, token_idx, window_id, window_token_pos,
            score, geom, lat, lng )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                (
                    e["event_id"],
                    n["event_id"],
                    n.get("depth", 1),
                    n.get("via_event_id"),
                    n["vector_id"],
                    n["token"],
                    n["doc_id"],
                    n["pub_year"],
                    n["token_idx"],
                    n["window_id"],
                    n["window_token_pos"],
                    n["score"],
                    n.get("geom"),
                    n.get("lat"),
                    n.get("lng"),
                )
                for e in data["events"]
                for n in e["neighbours"]
            ],
        )

        con.executemany(
            """INSERT INTO concept_aggregate
               (concept, kind, rank, value, window_doc_id, window_id, count)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            list(_aggregate_rows(concept_name, data["aggregate"])),
        )

        con.executemany(
            """UPDATE events
            SET geom = ?, lat = ?, lng = ?
            WHERE doc_id = ? AND (lat IS NULL OR lng IS NULL)""",
            [
                (meta["geom"], meta["lat"], meta["lng"], doc_id)
                for doc_id, meta in doc_meta.items()
                if meta.get("lat") is not None and meta.get("lng") is not None
            ]
        )

        con.commit()

    populate_documents_table(con, doc_meta)

    con.close()
    logger.info(f"[tier2] sqlite write complete: {db_path}")


def populate_documents_table(con, doc_meta):
    """Populate a lightweight documents table from doc_meta."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id    TEXT PRIMARY KEY,
            title     TEXT,
            author    TEXT,
            pub_year  INTEGER,
            publisher TEXT,
            pub_place TEXT
        )
    """)

    con.execute("DELETE FROM documents")

    data = [
        (doc_id, meta.get("title"), None, meta.get("pub_year"), None, None)
        for doc_id, meta in doc_meta.items()
    ]

    con.executemany(
        "INSERT INTO documents (doc_id, title, author, pub_year, publisher, pub_place) VALUES (?,?,?,?,?,?)",
        data
    )


def _aggregate_rows(concept_name, aggregate):
    """Yield concept_aggregate row tuples from an analyse_concept aggregate dict."""
    for rank, (token, count) in enumerate(aggregate["top_tokens"]):
        yield (concept_name, "token", rank, token, None, None, count)

    for rank, (doc_id, count) in enumerate(aggregate["top_docs"]):
        yield (concept_name, "doc", rank, doc_id, None, None, count)

    for rank, ((doc_id, window_id), count) in enumerate(aggregate["top_windows"]):
        yield (concept_name, "window", rank, None, doc_id, window_id, count)


def get_processed_concepts(db_path) -> set[str]:
    if not Path(db_path).is_file():
        return set()
    try:
        con = sqlite3_connection(db_path)
        cur = con.execute("SELECT concept FROM concepts")
        result = {row[0] for row in cur.fetchall()}
        con.close()
        return result
    except sqlite3.OperationalError:
        # table doesn't exist yet
        return set()


def run_tier2_service(
    *,
    doc_meta,
    concepts_to_run,
    db_path,
    index           = None,
    lookup          = None,
    false_positives = None,
    clear           = False,
    diagnostics     = False,
    depth           = 1,
    emit            = None
):
    concept_names = [name for name, _ in concepts_to_run]
    logger = setEmit(
        emit,
        "[tier2]",
        {"concepts": concept_names},
    )
    logger.info(f"[tier2.run_tier2_service] Enter")

    index = index or EeboFaissIndex.load(FAISS_TIER1_INDEX)
    if index.ntotal == 0:
        raise RuntimeError( "FAISS index is empty — run tier1_5_build_faiss_index.py first" )

    output = run_tier2_core(
        index            = index,
        doc_meta         = doc_meta,
        concepts_to_run  = concepts_to_run,
        lookup           = lookup,
        false_positives  = false_positives,
        diagnostics      = diagnostics,
        depth            = depth,
        emit             = emit
    )

    logger.info(f"[tier2.run_tier2_service] Write SQL")
    write_sqlite( output, db_path, clear=clear, doc_meta=doc_meta)

    logger.info(f"[tier2.run_tier2_service] Done")
    return output


def run_tier2_core(
    *,
    index,
    doc_meta,
    concepts_to_run,
    lookup          = None,
    false_positives = None,
    diagnostics     = False,
    target_forms    = None,
    depth           = 1,
    emit            = None,
):
    logger.info("[tier2.run_tier2_core] Enter")
    output = {}

    if diagnostics:
        knn_diagnostics( lookup, index, CONCEPT_SETS["PREROGATIVE"]["forms"], )
        knn_diagnostics( lookup, index, CONCEPT_SETS["LAW"]["forms"], )

    for concept_name, concept in concepts_to_run:
        output[concept_name] = analyse_concept(
            doc_meta,
            index,
            lookup,
            concept_name,
            concept,
            diagnostics=diagnostics,
            depth=depth,
        )
    logger.info("[tier2.run_tier2_core] Leave")
    return output


def main():
    logger.info("[tier2] Enter")

    parser = argparse.ArgumentParser()
    parser.add_argument( "--concept", type=str, default=None, help="Run analysis for a single concept (case-insensitive)", )
    parser.add_argument( "--forms", type=str, default=None, help="Comma-separated list of forms (required if --concept is not in CONCEPT_SETS)", )
    parser.add_argument( "--false-positives", type=str, default=None, help="Comma-separated list of false positive forms to exclude", )
    parser.add_argument( "--clear", action="store_true", help="Wipe and recreate SQLite database before writing", )
    parser.add_argument( "-d", "--diagnostics", action="store_true", help="Enable Tier2 diagnostics", )
    parser.add_argument( "--depth", type=int, default=1, choices=[1, 2], help="Neighbour depth: 1=direct only (default), 2=include neighbours-of-neighbours", )
    args = parser.parse_args()

    if args.clear and args.concept:
        logger.warning( "[tier2.main] --clear with --concept will wipe all concepts before writing one" )

    if args.clear:
        if CORPUS_TIER2_DB_PATH.exists():
            logger.warning(f"[tier2.main] deleting SQLite DB: {CORPUS_TIER2_DB_PATH}")
            os.remove(CORPUS_TIER2_DB_PATH)
        else:
            logger.info("[tier2] reset-sqlite requested but DB does not exist")


    # If a single concept is requested, restrict the lookup to its forms
    # so that only matching events are loaded into memory.
    target_forms = None
    target_fps = None

    if args.concept:
        concept_name = args.concept.upper()

        if args.forms:
            target_forms = {
                f.strip()
                for f in args.forms.split(",")
            }
            target_fps = (
                {f.strip() for f in args.false_positives.split(",")}
                if args.false_positives
                else None
            )
        else:
            target_forms = set(CONCEPT_SETS[concept_name]["forms"])
            target_fps = set(
                CONCEPT_SETS[concept_name].get("false_positives", [])
            )

        logger.info( f"[tier2.main] single-concept mode: {concept_name} forms={target_forms}" )

    conn = get_connection()
    doc_meta = load_doc_metadata(conn)
    output = {}

    already_processed = (
        set()
        if args.clear
        else get_processed_concepts(CORPUS_TIER2_DB_PATH)
    )

    concepts_to_run = [
        (concept_name, concept)
        for concept_name, concept in resolve_concepts(
            concept=args.concept,
            false_positives=args.false_positives,
        )
        if concept_name not in already_processed
    ]

    if not concepts_to_run:
        logger.info( "[tier2.main] nothing to write — all concepts already processed" )
        return

    lookup = ZarrEventLookup(
        ZARR_ROOT / "tier1",
        forms=target_forms,
        false_positives=target_fps,
    )

    run_tier2_service(
        doc_meta        = doc_meta,
        concepts_to_run = concepts_to_run,
        db_path         = CORPUS_TIER2_DB_PATH,
        lookup          = lookup,
        false_positives = target_fps,
        clear           = args.clear,
        diagnostics     = args.diagnostics,
        depth           = args.depth,
        emit            = None
    )
    logger.info(f"[tier2.main] Complete, wrote {CORPUS_TIER2_DB_PATH}")


if __name__ == "__main__":
    main()

