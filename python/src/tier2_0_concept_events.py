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
Tier 1 observation metadata are streamed from Zarr into an in-memory struct-of-arrays lookup.
Embeddings remain in the per-year FAISS indices and are reconstructed lazily on demand,
keeping memory usage proportional to metadata rather than embedding dimensionality.

When analysing a single concept, only observations matching the requested
forms are loaded, making memory consumption proportional to the concept
rather than the corpus.

Storage model
-------------
Metadata are stored as parallel NumPy arrays indexed by row position,
together with an event_id → row lookup.

The lookup contains observation metadata together with lazy array-like accessors for local,
 medium and broad embeddings. Embeddings are reconstructed on demand from the corresponding
 per-year FAISS indices, preserving the previous API while avoiding eager loading of the full
 embedding matrices.

Ensemble embeddings are computed on demand rather than materialised separately.

This design provides good cache locality while avoiding millions of small Python objects,
which quickly becomes expensive.

Neighbourhood model
-------------------
Each query observation is searched against the global FAISS index using its
ensemble embedding.

Neighbourhoods were once expanded to two levels:

- depth 1: direct semantic neighbours
- depth 2: neighbours-of-neighbours (disabled for now)

Both levels retained full provenance and explicitly record the path through
which secondary neighbours were discovered. This has been dropped.

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
concepts while leaving to later tiers higher-level interpretation — such as
clustering, semantic field induction, temporal comparison and semantic drift.

This separation keeps retrieval, neighbourhood construction and semantic
interpretation as distinct stages of the pipeline, allowing each to evolve
independently without compromising provenance or reproducibility.
"""

from __future__ import annotations

import os
import argparse
import sqlite3
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from itertools import combinations
from pathlib import Path
from lib.zarr_event_lookup import ZarrEventLookup

import numpy as np

from lib.eebo_config import (
    CONCEPT_SETS, OUT_DIR,
    ZARR_PATH, MASKED_ZARR_PATH,
    CORPUS_TIER2_DB_PATH, CORPUS_TIER2_MASKED_DB_PATH,
    faiss_index_paths, discover_index_years
)
from lib.eebo_faiss import EeboFaissIndex, reciprocal_rank_fusion, multiscale_search
from lib.eebo_logging import logger, setEmit
from lib.concept_resolve import resolve_concepts
from lib.eebo_db import get_connection
from lib.tier2_diagnostics import (
    audit_embedding_diversity,
    audit_embedding_isotropy,
    audit_hubness,
    audit_neighbour_identity,
    audit_knn_stability,
    knn_diagnostics,
)

K          = 50    # kNN
RRF_K      = 60     # Reciprocal Rank Fusion smoothing - Smaller values make top ranks dominate more strongly; larger values flatten rank contribution.
OVERSAMPLE = 3      # pull top_n * OVERSAMPLE candidates per scale before fusing

# Sentinel for absent window_token_pos in the int64 column. -1 is never a
# valid token position, so it is unambiguous as "not present".
_NO_WPOS = -1


def load_doc_metadata(conn) -> dict:
    """
    Build doc_id -> metadata mapping from documents + place_normalization.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT ON (d.doc_id)
            d.doc_id,
            d.title,
            d.pub_year,
            d.pub_place,
            d.author,
            d.publisher,
            pn.normalized_places,
            pn.geom,
            ST_Y(pn.geom::geometry) AS lat,
            ST_X(pn.geom::geometry) AS lng
        FROM documents d
        LEFT JOIN place_normalization pn ON d.pub_place = pn.raw_place
        ORDER BY d.doc_id
    """)

    out = {}
    for (
        doc_id,
        title,
        pub_year,
        pub_place,
        author,
        publisher,
        places,
        geom,
        lat,
        lng,
    ) in cur.fetchall():
        out[doc_id] = {
            "pub_year": pub_year,
            "title": title,
            "author": author,
            "publisher": publisher,
            "pub_place": pub_place,
            "places": places,
            "geom": geom,   # ST_Point
            "lat": lat,
            "lng": lng,
        }
    return out


def populate_documents_table(con, doc_meta):
    """Populate or refresh the documents table with metadata and location info."""
    logger.info(f"[tier2] Populating documents table ({len(doc_meta)} entries)")

    con.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id            TEXT PRIMARY KEY,
            title             TEXT,
            author            TEXT,
            pub_year          INTEGER,
            publisher         TEXT,
            pub_place         TEXT,
            normalized_places TEXT, -- json
            geom              TEXT,
            lat               REAL,
            lng               REAL
        )
    """)

    con.execute("DELETE FROM documents")

    data = []
    for doc_id, meta in doc_meta.items():
        places = meta.get("places")
        places_json = json.dumps(places) if isinstance(places, (list, dict)) else str(places) if places else None
        data.append((
            doc_id,
            meta.get("title"),
            meta.get("author"),
            meta.get("pub_year"),
            meta.get("publisher"),
            meta.get("pub_place"),
            places_json,
            meta.get("geom"),
            meta.get("lat"),
            meta.get("lng")
        ))

    con.executemany("""
        INSERT INTO documents
        (doc_id, title, author, pub_year, publisher, pub_place,
         normalized_places, geom, lat, lng)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)

    con.executescript("""
        CREATE INDEX IF NOT EXISTS docs_pub_latlng_idx ON documents(lat, lng);
        CREATE INDEX IF NOT EXISTS idx_documents_pub_year_lat_lng ON documents(pub_year, lat, lng);
        CREATE INDEX IF NOT EXISTS docs_doc_idx ON documents(doc_id);
        CREATE INDEX IF NOT EXISTS docs_author_idx ON documents(author);
        CREATE INDEX IF NOT EXISTS docs_pub_year_idx ON documents(pub_year);
        CREATE INDEX IF NOT EXISTS docs_pub_place_idx ON documents(pub_place);
        CREATE INDEX IF NOT EXISTS idx_neighbours_neighbour_event_id_pubyear ON neighbours(neighbour_event_id, pub_year);
        CREATE INDEX IF NOT EXISTS idx_neighbours_pubyear ON neighbours(pub_year);
    """)


def analyse_concept(
    doc_meta, index, lookup, concept_name, concept,
    *,
    diagnostics     = False,
    false_positives = None,
    depth: int      = 1,
    top_n: int      = K,
    rrf_k: int      = RRF_K,
    oversample: int = OVERSAMPLE,
):
    """
    Compute neighbourhood structure for all events matching a concept.

    Parameters
    ----------
    index : dict[int, dict[str, EeboFaissIndex]]
        Per-year, per-scale FAISS indices: index[year]["local"/"medium"/"broad"].
        Neighbours are retrieved independently at each scale and fused via
        reciprocal rank fusion — see multiscale_search.
    depth : int
        1 = direct fused neighbours only (original behaviour)
        2 = also include neighbours-of-neighbours (depth-2 expansion)

    Neighbour year-scoping
    -----------------------
    Each query event is scoped to its OWN document's pub_year: it only
    searches for neighbours within that year's FAISS indices. This is a
    per-event scope, not a single global year for the whole concept run.

    Query events whose doc_id has no pub_year in doc_meta (pub_year is
    None) fall back to an unscoped search across all years in `index`,
    since there's no year to scope them to. A warning is logged with the
    count of such events so this doesn't happen silently.

    Returns
    -------
    dict with keys:
        - concept, forms, n_events
        - aggregate (top tokens/docs/windows)
        - events: list of dicts containing:
            * event metadata
            * neighbours: list of neighbour dicts (same structure)

    Note on "score"
    ----------------
    Neighbour "score" is a reciprocal-rank-fusion score (higher = more
    consistently ranked across the three scales), not a cosine similarity.
    It is not bounded to [-1, 1].
    """
    L_event_id   = lookup.event_id
    L_vector_id  = lookup.vector_id
    L_doc_id     = lookup.doc_id
    L_token      = lookup.token
    L_token_idx  = lookup.token_idx
    L_window_id  = lookup.window_id
    L_wpos       = lookup.window_token_pos
    L_pub_year   = lookup.pub_year

    forms = {
        f.lower()
        for f in concept.get("forms", []) or []
    }

    false_positives = {
        f.lower()
        for f in list(false_positives or []) + list(concept.get("false_positives", []))
    }

    event_ids = [
        eid
        for eid in lookup.iter_matching_event_ids(forms, false_positives)
        if pub_year is None or lookup.pub_year[lookup.get_pos(eid)] == pub_year
    ]

    logger.info(f"[tier2] concept={concept_name} | events={len(event_ids)}")
    if false_positives:
        logger.info(f"[tier2] excluding false_positives={false_positives}")

    if not event_ids:
        return {"concept": concept_name, "empty": True}

    event_pos = np.fromiter(
        (lookup.get_pos(eid) for eid in event_ids),
        dtype=np.int64,
        count=len(event_ids),
    )

    if diagnostics:
        logger.debug(f"[tier2] query_events={len(event_ids)}")
        logger.debug(f"[tier2] sample_event_id={event_ids[0]}")

    # Group query events by their own doc's pub_year, so each event only
    # searches for neighbours within its own year's indices. Events with
    # unknown pub_year (doc_id missing from doc_meta, or pub_year is None)
    # are grouped separately and searched unscoped (across all years).
    year_of_query: list[int | None] = []
    for eid in event_ids:
        pos = lookup.get_pos(eid)
        doc_id = str(L_doc_id[pos])
        year_of_query.append(doc_meta.get(doc_id, {}).get("pub_year"))

    # Group query events by their own pub_year (from zarr, same source that
    # build_indices.py partitioned the FAISS indices by — guaranteed to match
    # index keys, unlike doc_meta's Postgres pub_year which could drift).
    groups: dict[int, list[int]] = {}
    for i in range(len(event_ids)):
        pos = int(event_pos[i])
        year = int(L_pub_year[pos])
        groups.setdefault(year, []).append(i)

    fused_per_query = [None] * len(event_ids)

    for year, idxs in groups.items():
        group_pos = event_pos[idxs]
        group_fused = multiscale_search(
            index, lookup, group_pos, top_n,
            pub_year=year, rrf_k=rrf_k, oversample=oversample,
        )
        for local_i, global_i in enumerate(idxs):
            fused_per_query[global_i] = group_fused[local_i]

    token_counter  = Counter()
    doc_counter    = Counter()
    window_counter = Counter()
    results        = []

    query_event_id_set = set(event_ids)

    for i, eid in enumerate(event_ids):
        q_pos = int(event_pos[i])
        q_doc_id = str(L_doc_id[q_pos])
        q_pub_year = doc_meta.get(q_doc_id, {}).get("pub_year")
        q_wpos = int(L_wpos[q_pos])

        neighbours = []

        for entry in fused_per_query[i]:
            nid_int = entry["event_id"]
            if nid_int == eid:
                continue
            n_pos = lookup.get_pos(nid_int)
            n_token = str(L_token[n_pos])
            if n_token.lower() in false_positives:
                continue

            n_doc_id = str(L_doc_id[n_pos])
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
                "score":            entry["rrf_score"],
                "score_local":      entry["score_local"],
                "score_medium":     entry["score_medium"],
                "score_broad":      entry["score_broad"],
                "depth":            1,
                "via_event_id":     None,
            })

        results.append({
            "event_id":         int(L_event_id[q_pos]),
            "vector_id":        int(L_vector_id[q_pos]),
            "token":            str(L_token[q_pos]),
            "doc_id":           q_doc_id,
            "pub_year":         q_pub_year,
            "token_idx":        int(L_token_idx[q_pos]),
            "window_id":        int(L_window_id[q_pos]),
            "window_token_pos": None if q_wpos == _NO_WPOS else q_wpos,
            "neighbours":       neighbours,
        })

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
    score_local         REAL,
    score_medium        REAL,
    score_broad         REAL,
    nx                  REAL,
    ny                  REAL,
    gnx                 REAL,
    gny                 REAL,
    cluster_id          INTEGER,
    cluster_label       TEXT,
    PRIMARY KEY (event_id, neighbour_event_id, depth),
    FOREIGN KEY (event_id) REFERENCES events(event_id)
);

CREATE INDEX IF NOT EXISTS idx_events_concept_pubyear_nx ON events(concept, pub_year, nx, ny, gnx, gny);

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
    label          TEXT,
    description    TEXT,
    PRIMARY KEY (concept, cluster_id)
);
"""

_SCHEMA_CLEAR = """
    DROP TABLE IF EXISTS concept_forms;
    DROP TABLE IF EXISTS concepts;
    DROP TABLE IF EXISTS concept_cluster_info;
    DROP TABLE IF EXISTS concept_aggregate;
    DROP TABLE IF EXISTS neighbours;
    DROP TABLE IF EXISTS events;
    DROP TABLE IF EXISTS concepts;
    DROP TABLE IF EXISTS documents;
"""

_DELETE_CONCEPT = [
    "DELETE FROM concept_cluster_info WHERE concept = ?",
    "DELETE FROM concept_aggregate WHERE concept = ?",
    "DELETE FROM neighbours WHERE event_id IN (SELECT event_id FROM events WHERE concept = ?)",
    "DELETE FROM events WHERE concept = ?",
    "DELETE FROM concepts WHERE concept = ?",
]


def sqlite3_connection(db_path):
    logger.info(f"[tier2] sqlite3_connection {db_path}")
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

        # Events - NO location fields
        con.executemany( """
            INSERT OR IGNORE INTO events
            (event_id, concept, vector_id, token, doc_id, pub_year,
             token_idx, window_id, window_token_pos)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                )
                for e in data["events"]
            ],
        )

        # Neighbours - NO location fields
        con.executemany("""
            INSERT OR IGNORE INTO neighbours
            (event_id, neighbour_event_id, depth, via_event_id, vector_id,
             token, doc_id, pub_year, token_idx, window_id, window_token_pos,
             score, score_local, score_medium, score_broad)
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
                    n.get("score_local"),
                    n.get("score_medium"),
                    n.get("score_broad"),
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

    # Populate documents table (once, outside the loop)
    if doc_meta:
        populate_documents_table(con, doc_meta)

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



from concurrent.futures import ThreadPoolExecutor, as_completed

def run_tier2_service(
    *,
    doc_meta,
    concepts_to_run,
    db_path,
    faiss_index_paths_by_year,
    index           = None,
    lookup          = None,
    false_positives = None,
    clear           = False,
    diagnostics     = False,
    depth           = 1,
    max_load_workers = 6,
    emit            = None,
    top_n           = K,
    rrf_k           = RRF_K,
    oversample      = OVERSAMPLE,
    pub_year        = None,
):
    concept_names = [name for name, _ in concepts_to_run]
    logger = setEmit(emit, "[tier2]", {"concepts": concept_names})
    logger.info(f"[tier2.run_tier2_service] Enter")

    if lookup is None:
        raise RuntimeError("run_tier2_service() requires a ZarrEventLookup")

    if index is None:
        # Flatten to a list of (year, scale, path) load jobs and run them
        # concurrently — each EeboFaissIndex.load() is an independent file
        # read + faiss deserialization, so this is a straightforward
        # I/O-bound fan-out with no shared state between jobs.
        jobs = [
            (year, scale, path)
            for year, scale_paths in faiss_index_paths_by_year.items()
            for scale, path in scale_paths.items()
        ]

        index = {year: {} for year in faiss_index_paths_by_year}

        logger.info(f"[tier2] loading {len(jobs)} FAISS indices "
                    f"({max_load_workers} workers)")

        with ThreadPoolExecutor(max_workers=max_load_workers) as pool:
            future_to_job = {
                pool.submit(EeboFaissIndex.load, path): (year, scale)
                for year, scale, path in jobs
            }
            for future in as_completed(future_to_job):
                year, scale = future_to_job[future]
                index[year][scale] = future.result()  # raises here if load() failed?

        logger.info(f"[tier2] finished loading {len(jobs)} indices")

    for year, scale_indices in index.items():
        for scale, idx in scale_indices.items():
            if idx.ntotal == 0:
                raise RuntimeError(
                    f"FAISS '{scale}'/{year} index is empty — run build_indices.py first"
                )

    lookup.attach_index(index)

    output = run_tier2_core(
        index            = index,
        doc_meta         = doc_meta,
        concepts_to_run  = concepts_to_run,
        lookup           = lookup,
        false_positives  = false_positives,
        diagnostics      = diagnostics,
        depth            = depth,
        emit             = emit,
        top_n            = top_n,
        rrf_k            = rrf_k,
        oversample       = oversample,
        pub_year         = pub_year,
    )

    logger.info(f"[tier2.run_tier2_service] Write SQL")
    write_sqlite(output, db_path, clear=clear, doc_meta=doc_meta)

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
    pub_year        = None,
    emit            = None,
    top_n           = K,
    rrf_k           = RRF_K,
    oversample      = OVERSAMPLE,
):
    logger.info("[tier2.run_tier2_core] Enter")
    output = {}

    if diagnostics:
        # index is now {year: {scale: EeboFaissIndex}}. Diagnostics were
        # written for a single global index, so pick one representative
        # year's medium index rather than every year's — running
        # knn_diagnostics per year would be N_years times the cost for a
        # debug-only code path. Picking the largest year as most
        # representative of corpus-wide geometry; revisit if that's wrong.
        diag_year = max(index.keys(), key=lambda y: index[y]["medium"].ntotal)
        logger.info(f"[tier2] diagnostics using medium index for year={diag_year} "
                    f"(largest of {len(index)} years)")
        knn_diagnostics(lookup, index[diag_year]["medium"], CONCEPT_SETS["PREROGATIVE"]["forms"])
        knn_diagnostics(lookup, index[diag_year]["medium"], CONCEPT_SETS["LAW"]["forms"])

    for concept_name, concept in concepts_to_run:
        output[concept_name] = analyse_concept(
            doc_meta,
            index,
            lookup,
            concept_name,
            concept,
            false_positives = false_positives,
            diagnostics      = diagnostics,
            depth            = depth,
            rrf_k            = rrf_k,
            top_n            = top_n,
            oversample       = oversample,
            pub_year         = pub_year,
        )
    logger.info("[tier2.run_tier2_core] Leave")
    return output


def main():
    logger.info("[tier2.main] Enter")

    parser = argparse.ArgumentParser()
    parser.add_argument( "--concept", type=str, default=None, help="Run analysis for a single concept (case-insensitive)", )
    parser.add_argument( "--forms", type=str, default=None, help="Comma-separated list of forms (required if --concept is not in CONCEPT_SETS)", )
    parser.add_argument( "--false-positives", type=str, default=None, help="Comma-separated list of false positive forms to exclude", )
    parser.add_argument( "--clear", action="store_true", help="Wipe and recreate SQLite database before writing", )
    parser.add_argument( "-d", "--diagnostics", action="store_true", help="Enable Tier2 diagnostics", )
    parser.add_argument( "--depth", type=int, default=1, choices=[1, 2], help="Neighbour depth: 1=direct only (default), 2=include neighbours-of-neighbours", )
    parser.add_argument("--mask", action="store_true", help="With masked stores and indicies")
    parser.add_argument("--query-year", type=int, default=None, help="Restrict neighbour search to a single publication year (default: search all years)")
    parser.add_argument("--max-load-workers", type=int, default=6, help="Maximum number of workers to spawn to load indicies")
    parser.add_argument( "--rrf-k", type=int, default=RRF_K, help="Reciprocal rank fusion smoothing constant")
    parser.add_argument( "--k", type=int, default=K, help="Top nearest neighbours")
    parser.add_argument( "--oversample", type=int, default=OVERSAMPLE, help="Top-n * oversample")

    args = parser.parse_args()

    if args.mask:
        zarr_path = MASKED_ZARR_PATH
        db_path = CORPUS_TIER2_MASKED_DB_PATH
    else:
        zarr_path = ZARR_PATH
        db_path = CORPUS_TIER2_DB_PATH

    years = discover_index_years(args.mask)
    if not years:
        raise RuntimeError(
            f"No FAISS indices found for mode={'masked' if args.mask else 'unmasked'}. "
            f"Run build_indices.py first."
        )
    faiss_index_paths_by_year = {
        year: faiss_index_paths(masked = args.mask, year = year)
        for year in years
    }

    logger.info(f"[Tier 2.main] mode={'masked' if args.mask else 'unmasked'}")
    logger.info(f"[Tier 2.main] k={args.k}; rrf_k={args.rrf_k}; oversample={args.oversample}")

    if args.clear and args.concept:
        logger.warning( "[tier2.main] --clear with --concept will wipe all concepts before writing one" )

    if args.clear:
        if db_path.exists():
            logger.warning(f"[tier2.main] deleting SQLite DB: {db_path}")
            os.remove(db_path)
        else:
            logger.info("[tier2.main] reset-sqlite requested but DB does not exist")


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
        else get_processed_concepts(db_path)
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
        zarr_path,
        forms=target_forms,
        false_positives=target_fps,
    )

    run_tier2_service(
        doc_meta                    = doc_meta,
        concepts_to_run             = concepts_to_run,
        db_path                     = db_path,
        faiss_index_paths_by_year   = faiss_index_paths_by_year,
        lookup                      = lookup,
        false_positives             = target_fps,
        clear                       = args.clear,
        diagnostics                 = args.diagnostics,
        depth                       = args.depth,
        max_load_workers            = args.max_load_workers,
        emit                        = None,
        top_n                       = args.k,
        rrf_k                       = args.rrf_k,
        oversample                  = args.oversample,
        pub_year                    = args.query_year,
    )
    logger.info(f"[tier2.main] Complete, wrote {db_path}")


if __name__ == "__main__":
    main()

