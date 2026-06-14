#!/usr/bin/env python
"""
tier2_0_sql_concept_events.py - Tier2: neighbourhood analysis over event-space substrate

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

Embeddings are currently stored in ZarrEventLookup alongside metadata so
that FAISS queries can be issued using the canonical Zarr vector rather than
relying on FAISS internal storage. This is correct and safe for IndexFlatIP,
which stores vectors verbatim, but it has two consequences:

    1. Memory: all embeddings for the full corpus are resident in the lookup
       dict when no concept filter is active. For a large corpus this can
       be several GB.

    2. Coupling: the lookup is responsible for both metadata and vector
       storage, which conflates two concerns.

A cleaner long-term approach is to drop the "embedding" field from
ZarrEventLookup.by_event_id entirely, and instead reconstruct vectors
from the FAISS index at query time via EeboFaissIndex.reconstruct().
See eebo_faiss.py for the reconstruct() method and usage notes.

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

Four tables:

    events
        One row per query event (globally unique by event_id).

    neighbours
        One row per (query event, neighbour) pair.
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
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import zarr

from lib.eebo_config import CONCEPT_SETS, INDEXES_DIR, FAISS_TIER1_INDEX, ZARR_ROOT, OUT_DIR, SQLITE_DB_PATH
from lib.eebo_faiss import EeboFaissIndex
from lib.eebo_logging import logger
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

# Event lookup
class ZarrEventLookup:
    """
    In-memory index of Tier 1 observation events, keyed by event_id.

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

    def __init__(self, root, forms: set[str] | None = None, false_positives: set[str] | None = None):
        self.root            = root
        self.forms           = {f.lower() for f in forms} if forms else None
        self.false_positives = {f.lower() for f in false_positives} if false_positives else set()
        self.by_event_id     = {}
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

        logger.info(f"[tier2] events={len(self.by_event_id)}")


    def _load_store(self, e, store_dir):
        """
        Load events from one Zarr store into by_event_id.

        Reads each dataset as a contiguous numpy array per batch.
        Indexing into numpy in the inner loop is cheap; indexing directly
        into Zarr would trigger per-element decompression.

        If self.forms is set, only tokens matching a known form are
        retained, skipping all other rows without building Python objects
        for them.
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

            for i in range(end - start):
                token = str(b_toks[i])
                token_lower = token.lower()
                if self.forms and token_lower not in self.forms:
                    continue
                if token_lower in self.false_positives:
                    continue

                eid = int(b_eids[i])
                self.by_event_id[eid] = {
                    "event_id":         eid,
                    "vector_id":        int(b_vids[i]),
                    "doc_id":           str(b_docs[i]),
                    "token":            token,
                    "token_idx":        int(b_idxs[i]),
                    "window_id":        int(b_wins[i]),
                    "window_token_pos": int(b_wpos[i]) if b_wpos is not None else None,
                    # NOTE: storing embeddings here holds all matching
                    # embeddings in memory. See module docstring for the
                    # deferred migration to EeboFaissIndex.reconstruct().
                    "embedding":        np.asarray(b_embs[i], dtype=np.float32),
                }

    def get_event(self, event_id: int) -> dict:
        return self.by_event_id[int(event_id)]

    def iter_matching_event_ids(self, forms, false_positives=None):
        forms = {f.lower() for f in forms}
        false_positives = {f.lower() for f in (false_positives or [])}
        for eid, event in self.by_event_id.items():
            token = event["token"].lower()
            if token in forms and token not in false_positives:
                yield eid


# Document metadata
def load_doc_metadata(conn) -> dict:
    """
    Build a doc_id -> metadata mapping from pamphlet_tokens.
    pub_year and title are stable per doc_id; we take the first occurrence.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT ON (doc_id)
            doc_id,
            pub_year,
            title
        FROM documents
        ORDER BY doc_id
    """)
    return {
        row[0]: {"pub_year": row[1], "title": row[2]}
        for row in cur.fetchall()
    }


# Concept analysis
def _event_record(event, doc_meta):
    """Serialisable dict for one query event (without neighbours)."""
    return {
        "event_id":         int(event["event_id"]),
        "vector_id":        event["vector_id"],
        "token":            event["token"],
        "doc_id":           event["doc_id"],
        "pub_year":         doc_meta.get(event["doc_id"], {}).get("pub_year"),
        "token_idx":        event["token_idx"],
        "window_id":        event["window_id"],
        "window_token_pos": event["window_token_pos"],
    }


def _neighbour_record(n_event, query_event, doc_meta, score):
    """Serialisable dict for one neighbour of a query event."""
    return {
        "event_id":         int(n_event["event_id"]),
        "vector_id":        n_event["vector_id"],
        "token":            n_event["token"],
        "doc_id":           n_event["doc_id"],
        "pub_year":         doc_meta.get(query_event["doc_id"], {}).get("pub_year"),
        "token_idx":        n_event["token_idx"],
        "window_id":        n_event["window_id"],
        "window_token_pos": n_event["window_token_pos"],
        "score":            float(score),
    }


def analyse_concept(doc_meta, index, lookup, concept_name, concept, top_n=K, *, diagnostics=False):
    """Compute neighbourhood structure for all events matching a concept."""
    forms           = set(concept["forms"])
    false_positives = {f.lower() for f in concept.get("false_positives", [])}

    event_ids = list(lookup.iter_matching_event_ids(forms, false_positives))

    logger.info(f"[tier2] concept={concept_name}")
    if false_positives:
        logger.info(f"[tier2] excluding false_positives={false_positives}")

    if not event_ids:
        return {"concept": concept_name, "empty": True}

    query_vecs = np.stack([
        lookup.get_event(eid)["embedding"] for eid in event_ids
    ])

    logger.info(f"[tier2] query_events={len(event_ids)}")
    logger.info(f"[tier2] sample_event_id={event_ids[0]}")
    logger.info(f"[tier2] sample_embedding_shape={query_vecs.shape}")

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

    for i, eid in enumerate(event_ids):
        event      = lookup.get_event(eid)
        neighbours = []

        for nid, score in zip(all_neigh_ids[i], all_scores[i]):
            if nid == -1 or int(nid) == int(eid):
                continue

            n_event = lookup.get_event(int(nid))

            if n_event["token"].lower() in false_positives:
                continue

            token_counter[n_event["token"]]                           += 1
            doc_counter[n_event["doc_id"]]                            += 1
            window_counter[(n_event["doc_id"], n_event["window_id"])] += 1

            neighbours.append(_neighbour_record(n_event, event, doc_meta, score))

        results.append({**_event_record(event, doc_meta), "neighbours": neighbours})

    return {
        "concept":   concept_name,
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
    FOREIGN KEY (concept) REFERENCES concepts(concept)
);

CREATE TABLE IF NOT EXISTS neighbours (
    event_id             INTEGER NOT NULL,
    neighbour_event_id   INTEGER NOT NULL,
    vector_id            INTEGER,
    token                TEXT,
    doc_id               TEXT,
    pub_year             INTEGER,
    token_idx            INTEGER,
    window_id            INTEGER,
    window_token_pos     INTEGER,
    score                REAL,
    PRIMARY KEY (event_id, neighbour_event_id),
    FOREIGN KEY (event_id) REFERENCES events(event_id)
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


def write_sqlite(output: dict, db_path, *, clear: bool = False):
    """
    Write analyse_concept output to a normalised SQLite database.

    If clear=True, all existing tables are dropped and recreated before
    writing. Use this when rebuilding the full corpus analysis from scratch.

    Otherwise, existing rows for each concept in output are deleted and
    rewritten, leaving all other concepts intact. This is the correct path
    for single-concept runs from the UI.
    """
    logger.info(f"[tier2] writing sqlite -> {db_path}")

    con = sqlite3.connect(db_path)

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
            "INSERT OR IGNORE INTO concepts VALUES (?, ?)",
            (concept_name, data["n_events"]),
        )

        con.executemany(
            """INSERT OR IGNORE INTO events
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                )
                for e in data["events"]
            ],
        )

        con.executemany(
            """INSERT OR IGNORE INTO neighbours
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    e["event_id"], n["event_id"], n["vector_id"], n["token"],
                    n["doc_id"], n["pub_year"], n["token_idx"],
                    n["window_id"], n["window_token_pos"], n["score"],
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

        con.commit()

    con.close()
    logger.info("[tier2] sqlite write complete")


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
        con = sqlite3.connect(db_path)
        cur = con.execute("SELECT concept FROM concepts")
        result = {row[0] for row in cur.fetchall()}
        con.close()
        return result
    except sqlite3.OperationalError:
        # table doesn't exist yet
        return set()


def main():
    logger.info("[tier2] init")

    parser = argparse.ArgumentParser()
    parser.add_argument( "--concept", type=str, default=None,
        help="Run analysis for a single concept (case-insensitive)",
    )
    parser.add_argument( "--forms", type=str, default=None,
        help="Comma-separated list of forms (required if --concept is not in CONCEPT_SETS)",
    )
    parser.add_argument( "--false-positives", type=str, default=None,
        help="Comma-separated list of false positive forms to exclude",
    )
    parser.add_argument( "--clear", action="store_true",
        help="Wipe and recreate SQLite database before writing",
    )
    parser.add_argument( "-d", "--diagnostics", action="store_true",
        help="Enable Tier2 diagnostics",
    )
    args = parser.parse_args()

    if args.clear and args.concept:
        logger.warning( "[tier2] --clear with --concept will wipe all concepts before writing one" )

    logger.info(f"[tier2] SQLITE_DB_PATH: {SQLITE_DB_PATH}")

    index = EeboFaissIndex.load(FAISS_TIER1_INDEX)

    if index.ntotal == 0:
        raise RuntimeError( "FAISS index is empty — run tier1_5_build_faiss_index.py first" )

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

        logger.info(
            f"[tier2] single-concept mode: {concept_name} forms={target_forms}"
        )

    conn = get_connection()
    doc_meta = load_doc_metadata(conn)
    output = {}

    already_processed = (
        set()
        if args.clear
        else get_processed_concepts(SQLITE_DB_PATH)
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
        logger.info(
            "[tier2] nothing to write — all concepts already processed"
        )
        return

    # Only build the lookup once we know there is work to do.
    # For single-concept runs, target_forms restricts loading to
    # matching events only, keeping memory use proportional to the
    # concept rather than the full corpus.
    lookup = ZarrEventLookup(
        ZARR_ROOT / "tier1",
        forms=target_forms,
        false_positives=target_fps,
    )

    if args.diagnostics:
        logger.info("--------------------------------------------------------")
        knn_diagnostics( lookup, index, CONCEPT_SETS["PREROGATIVE"]["forms"], )
        knn_diagnostics( lookup, index, CONCEPT_SETS["LAW"]["forms"], )
        logger.info("--------------------------------------------------------")

    for concept_name, concept in concepts_to_run:
        output[concept_name] = analyse_concept(
            doc_meta,
            index,
            lookup,
            concept_name,
            concept,
            diagnostics=args.diagnostics,
        )

    write_sqlite(
        output,
        SQLITE_DB_PATH,
        clear=args.clear,
    )

    logger.info(f"[tier2] wrote {SQLITE_DB_PATH}")


if __name__ == "__main__":
    main()
