#!/usr/bin/env python
"""
tier2_0_concept_events.py - Tier2: neighbourhood analysis over event-space substrate

Core invariant
--------------

Tier 1:
    token x window -> embedding event (Zarr), keyed by event_id

Tier 2:
    event -> geometric neighbourhood (FAISS) -> contextual analysis

FAISS is a geometric operator only.
FAISS ids are event_ids (stable, globally unique observation identity).
vector_id is lexical identity only and is NOT used as a lookup key.

No aggregation or centroid reconstruction is performed.
Events remain atomic observations with full provenance.

Performance model
-----------------

ZarrEventLookup._build reads the Tier 1 observation store in batches,
materialising one numpy array per dataset per batch before iterating in
Python. This avoids the element-by-element Zarr reads that would otherwise
pay decompression overhead on every scalar access.

When a single concept is queried (--concept), only events whose token
matches one of the concept's forms are loaded into memory. This makes
single-concept runs substantially faster and much lighter on memory than
full-corpus loads. When no concept filter is active, all events are loaded.

STORAGE MODEL (struct-of-arrays)
---------------------------------

ZarrEventLookup stores event metadata as parallel numpy arrays (one array
per field, indexed by row position), plus a single event_id -> row position
dict (`_pos`). This replaces the earlier design of one Python dict per
event (`by_event_id: dict[int, dict]`), which for a multi-million event
corpus meant millions of small heap objects, dict-hashing overhead per
field access, and poor cache locality.

The struct-of-arrays layout means:

    - _pos:        dict[int, int]      event_id -> row position (one dict)
    - event_id:    np.ndarray[int64]
    - vector_id:   np.ndarray[int64]
    - doc_id:      np.ndarray[object]  (interned strings)
    - token:       np.ndarray[object]  (interned strings)
    - token_idx:   np.ndarray[int64]
    - window_id:   np.ndarray[int64]
    - window_token_pos: np.ndarray[int64]  (-1 where absent)
    - embeddings:  np.ndarray[float32, shape=(n, dim)]

A "row" is the same row position across all of these arrays. get_event()
and get_pos() operate on row positions rather than per-event dicts, and
record construction for SQLite reads directly from these arrays.

Embeddings are currently stored in ZarrEventLookup alongside metadata so
that FAISS queries can be issued using the canonical Zarr vector rather than
relying on FAISS internal storage. This is correct and safe for IndexFlatIP,
which stores vectors verbatim, but it has two consequences:

    1. Memory: all embeddings for the full corpus are resident in the lookup
       when no concept filter is active. For a large corpus this can
       be several GB.

    2. Coupling: the lookup is responsible for both metadata and vector
       storage, which conflates two concerns.

A cleaner long-term approach is to drop the embeddings array from
ZarrEventLookup entirely, and instead reconstruct vectors from the FAISS
index at query time via EeboFaissIndex.reconstruct(). See eebo_faiss.py
for the reconstruct() method and usage notes.

This migration is deferred until the index type (exact vs. approximate)
is confirmed stable, because IndexHNSWFlat does not support vector
reconstruction.

window_id scoping invariant
---------------------------

window_counter keys are (doc_id, window_id) because window_id is defined
as a document-local coordinate in the Tier 1 store (it is the token-space
start offset of the transformer window within that document). Treating
window_id as globally unique across documents would silently merge windows
from different documents that happen to share the same offset. This
invariant is enforced in Tier 1 but is not re-checked here; if Tier 1
were ever rebuilt with a global window_id scheme this counter would
become incorrect without raising an error.

SQLite schema
-------------

Five tables:

    events
        One row per query event (globally unique by event_id).

    neighbours
        One row per (query event, neighbour, depth) triple.
        depth=1 rows are direct FAISS neighbours (original behaviour).
        depth=2 rows are neighbours-of-neighbours; via_event_id records
        which depth-1 event produced the result, and event_id still refers
        to the original concept query event (fan-out model).
        Foreign key: event_id -> events.event_id.

    concept_aggregate
        Flattened top_tokens / top_docs / top_windows rows.
        kind = 'token' | 'doc' | 'window'.
        For token/doc rows: value holds the token or doc_id; window_doc_id
        and window_id are NULL.
        For window rows: window_doc_id and window_id hold the tuple
        components; value is NULL.

    concepts
        One row per concept with n_events summary.

    concept_forms
        One row per concept + exemplar of form.

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

        # event_id -> row position (single dict, replaces by_event_id)
        self._pos: dict[int, int] = {}

        # Per-batch staging lists, concatenated into arrays in _finalize.
        self._chunks: dict[str, list[np.ndarray]] = {field: [] for field in self._FIELDS}
        self._emb_chunks: list[np.ndarray] = []

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
        Load events from one Zarr store into the staging columns.

        Reads each dataset as a contiguous numpy array per batch, then
        applies the forms/false_positives mask with vectorised numpy
        boolean indexing (np.isin) rather than per-row Python checks.
        Matching rows are appended to the staging arrays as whole arrays
        (no per-event dicts are created).
        """
        if "event_id" not in e:
            raise KeyError(f"Missing event_id in {store_dir} - rebuild Tier 1")

        wpos = e["window_token_pos"] if "window_token_pos" in e else None
        n    = e["event_id"].shape[0]

        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)

            b_eids = e["event_id"][start:end]
            b_vids = e["vector_id"][start:end]
            b_docs = e["doc_id"][start:end]
            b_toks = e["token"][start:end]
            b_idxs = e["token_idx"][start:end]
            b_wins = e["window_id"][start:end]
            b_embs = e["emb_raw"][start:end]
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

            if not keep.any():
                continue

            if b_wpos is not None:
                wpos_col = np.asarray(b_wpos, dtype=np.int64)[keep]
            else:
                wpos_col = np.full(int(keep.sum()), _NO_WPOS, dtype=np.int64)

            self._chunks["event_id"].append(np.asarray(b_eids, dtype=np.int64)[keep])
            self._chunks["vector_id"].append(np.asarray(b_vids, dtype=np.int64)[keep])
            self._chunks["doc_id"].append(b_docs[keep])
            self._chunks["token"].append(b_toks[keep])
            self._chunks["token_idx"].append(np.asarray(b_idxs, dtype=np.int64)[keep])
            self._chunks["window_id"].append(np.asarray(b_wins, dtype=np.int64)[keep])
            self._chunks["window_token_pos"].append(wpos_col)

            self._emb_chunks.append(np.asarray(b_embs, dtype=np.float32)[keep])

    def _finalize(self):
        """
        Concatenate staged per-batch arrays into the final columnar arrays
        and build the event_id -> row position map.

        If no events were loaded (e.g. an empty corpus or a concept with
        no matches), all columns are initialised as empty arrays with the
        correct dtype/shape so downstream code can operate uniformly.
        """
        n_total = sum(arr.shape[0] for arr in self._chunks["event_id"])

        if n_total == 0:
            for field, dtype in self._FIELDS.items():
                setattr(self, field, np.empty(0, dtype=dtype))
            self.embeddings = np.empty((0, 0), dtype=np.float32)
            self._chunks = {}
            self._emb_chunks = []
            return

        for field, dtype in self._FIELDS.items():
            setattr(self, field, np.concatenate(self._chunks[field]).astype(dtype, copy=False))

        self.embeddings = np.concatenate(self._emb_chunks, axis=0)

        # event_id -> row position. event_id is globally unique (Tier 1
        # invariant), so this is a straight bijection; last-write-wins is
        # not expected to occur but if Tier 1 ever produced a duplicate
        # this would silently keep the last row, matching prior dict
        # behaviour (by_event_id[eid] = ... overwrote on duplicates too).
        self._pos = {int(eid): pos for pos, eid in enumerate(self.event_id)}

        # Free staging buffers now that the columnar arrays own the data.
        self._chunks = {}
        self._emb_chunks = []

    # Row access

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
        return self._row_to_dict(pos)

    def get_pos(self, event_id: int) -> int:
        """event_id -> row position. Raises KeyError if not present."""
        return self._pos[int(event_id)]

    def _row_to_dict(self, pos: int) -> dict:
        wpos = int(self.window_token_pos[pos])
        return {
            "event_id":         int(self.event_id[pos]),
            "vector_id":        int(self.vector_id[pos]),
            "doc_id":           str(self.doc_id[pos]),
            "token":            str(self.token[pos]),
            "token_idx":        int(self.token_idx[pos]),
            "window_id":        int(self.window_id[pos]),
            "window_token_pos": None if wpos == _NO_WPOS else wpos,
            "embedding":        self.embeddings[pos],
        }

    def iter_matching_event_ids(self, forms, false_positives=None):
        """
        Yield event_ids whose token matches `forms` and is not in
        `false_positives`. Vectorised over the token column.
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

    depth=1 (default): direct FAISS neighbours only. Identical to the
    original behaviour; all existing call sites are unaffected.

    depth=2: additionally computes neighbours-of-neighbours. Each depth-1
    neighbour is used as a query vector for a second FAISS search. Results
    are fanned out to the original concept query event (event_id always
    refers to a concept event). via_event_id records which depth-1 event
    produced the depth-2 result. depth-2 rows exclude: the original query
    events themselves, any event already present as a depth-1 neighbour of
    that query event, and false_positive tokens.

    Record construction (event/neighbour rows) reads directly from the
    columnar arrays in `lookup` via row positions. The returned
    "events"/"neighbours" entries are still plain dicts (one per query
    event and one per surviving neighbour), since that shape is what
    write_sqlite consumes. pub_year is looked up once per query event.
    """
    # CONCEPT_SET entry union with any args supplied to this routine
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

    query_vecs = lookup.embeddings[event_pos]

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

    L_event_id  = lookup.event_id
    L_vector_id = lookup.vector_id
    L_doc_id    = lookup.doc_id
    L_token     = lookup.token
    L_token_idx = lookup.token_idx
    L_window_id = lookup.window_id
    L_wpos      = lookup.window_token_pos

    # Set of original query event_ids — used to exclude self-matches at
    # depth 2 without repeated list membership tests.
    query_event_id_set = set(event_ids)

    for i, eid in enumerate(event_ids):
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

    # ------------------------------------------------------------------
    # Depth-2: neighbours-of-neighbours
    # ------------------------------------------------------------------
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
                d1_vecs = lookup.embeddings[d1_positions]
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

CREATE INDEX IF NOT EXISTS idx_concept_forms_concept ON concept_forms(concept);
CREATE INDEX IF NOT EXISTS idx_concept_forms_form    ON concept_forms(form);

-- Should probably split into concept_events
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
    local_x          REAL,
    local_y          REAL,
    global_x         REAL,
    global_y         REAL,
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
    local_x             REAL,
    local_y             REAL,
    global_x            REAL,
    global_y            REAL,
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
"""

_SCHEMA_CLEAR = """
DROP TABLE IF EXISTS concept_aggregate;
DROP TABLE IF EXISTS neighbours;
DROP TABLE IF EXISTS events;
DROP TABLE IF EXISTS concepts;
"""

_DELETE_CONCEPT = [
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

        # Remove existing rows for this concept in dependency order
        # before rewriting, so a rerun of a single concept is idempotent.
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

        con.executemany( """INSERT OR IGNORE INTO events
            (event_id, concept, vector_id, token, doc_id, pub_year,
                token_idx, window_id, window_token_pos, geom, lat, lng)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    e["geom"],
                    e["lat"],
                    e["lng"]
                )
                for e in data["events"]
            ],
        )

        con.executemany(
            """INSERT OR IGNORE INTO neighbours
            (event_id, neighbour_event_id, depth, via_event_id, vector_id,
                token, doc_id, pub_year, token_idx, window_id, window_token_pos,
                score, geom, lat, lng)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    e["event_id"],
                    n["event_id"],
                    n.get("depth", 1),
                    n["via_event_id"],
                    n["vector_id"],
                    n["token"],
                    n["doc_id"],
                    n["pub_year"],
                    n["token_idx"],
                    n["window_id"],
                    n["window_token_pos"],
                    n["score"],
                    n["geom"],
                    n["lat"],
                    n["lng"],
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

    con.close()
    logger.info(f"[tier2] sqlite write complete: {db_path}")


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

