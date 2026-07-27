#!/usr/bin/env python
"""
tier2_concept_events.py

Tier 2 semantic neighbourhood construction.

Responsibilities:

    Tier 1 Zarr observations
            |
            v
    yearly FAISS retrieval geometry
            |
            v
    SQLite semantic neighbourhood store

Tier 2 does not define concepts. It records the empirical neighbourhood
around supplied lexical seeds and preserves provenance back to corpus
events.

Important invariants:

- event_id is the atomic corpus occurrence.
- FAISS ids are retrieval mechanisms only.
- RRF scores are ranking scores, not distances.
- Retrieval is publication-year scoped.
- Lexical forms are query provenance, not semantic membership.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from lib.eebo_config import (
    CORPUS_TIER2_DB_PATH,
    CORPUS_TIER2_MASKED_DB_PATH,
    ZARR_PATH,
    MASKED_ZARR_PATH,
    faiss_index_paths,
    discover_index_years,
)

from lib.eebo_faiss import (
    EeboFaissIndex,
    multiscale_search,
)

from lib.eebo_db import get_connection
from lib.zarr_event_lookup import ZarrEventLookup
from lib.eebo_logging import logger, setEmit
from lib.concept_resolve import resolve_concepts
from lib.get_processed_concepts import get_processed_concepts


K = 50
RRF_K = 60
OVERSAMPLE = 3
_NO_WPOS = -1



# SQLite schema
#
# This database is disposable analysis output.
# No migration layer is required.


SCHEMA = """

CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT,
    author TEXT,
    pub_year INTEGER,
    publisher TEXT,
    pub_place TEXT,
    normalized_places TEXT,
    lat REAL,
    lng REAL
);

CREATE TABLE IF NOT EXISTS concepts (
    concept TEXT PRIMARY KEY,
    n_events INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS concept_forms (
    concept TEXT NOT NULL,
    form TEXT NOT NULL,
    is_false_positive INTEGER DEFAULT 0,
    PRIMARY KEY(concept, form)
);

CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY,
    concept TEXT,
    event_role TEXT NOT NULL DEFAULT 'neighbour',
    vector_id INTEGER,
    token TEXT,
    doc_id TEXT,
    pub_year INTEGER,
    token_idx INTEGER,
    window_id INTEGER,
    window_token_pos INTEGER,

    -- Tier 3 derived geometry
    nx REAL,
    ny REAL,
    gnx REAL,
    gny REAL,

    -- Tier 3 clustering
    cluster_id INTEGER,
    cluster_label TEXT
);


CREATE TABLE IF NOT EXISTS neighbours (
    event_id INTEGER NOT NULL,
    neighbour_event_id INTEGER NOT NULL,

    depth INTEGER NOT NULL DEFAULT 1,
    via_event_id INTEGER,

    score REAL,
    score_local REAL,
    score_medium REAL,
    score_broad REAL,

    PRIMARY KEY(
        event_id,
        neighbour_event_id,
        depth
    )
);

--  concept_field_events is the authoritative relationship
CREATE TABLE IF NOT EXISTS concept_field_events (
    concept TEXT NOT NULL,
    event_id INTEGER NOT NULL,
    role TEXT NOT NULL,

    PRIMARY KEY(
        concept,
        event_id
    )
);

CREATE INDEX IF NOT EXISTS idx_field_events_concept ON concept_field_events(concept);
CREATE INDEX IF NOT EXISTS idx_field_events_event   ON concept_field_events(event_id);

CREATE TABLE IF NOT EXISTS concept_aggregate (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    concept TEXT NOT NULL,
    kind TEXT NOT NULL,
    rank INTEGER NOT NULL,
    value TEXT,
    window_doc_id TEXT,
    window_id INTEGER,
    count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS concept_cluster_info (
    concept TEXT NOT NULL,
    cluster_id INTEGER NOT NULL,
    cluster_label TEXT,
    centroid_nx REAL,
    centroid_ny REAL,
    centroid_gnx REAL,
    centroid_gny REAL,
    centroid_vector BLOB,
    point_count INTEGER,
    description TEXT,
    llm_model TEXT,
    llm_prompt TEXT,
    llm_timestamp TEXT,
    llm_sample_size INTEGER,
    llm_sample_event_ids TEXT,
    llm_concentration REAL,
    PRIMARY KEY(
        concept,
        cluster_id
    )
);

CREATE INDEX IF NOT EXISTS idx_events_concept               ON events(concept);
CREATE INDEX IF NOT EXISTS idx_events_year                  ON events(concept, pub_year);
CREATE INDEX IF NOT EXISTS idx_events_event_id              ON events(event_id);
CREATE INDEX IF NOT EXISTS idx_neighbours_event             ON neighbours(event_id);
CREATE INDEX IF NOT EXISTS idx_neighbours_neighbour_event   ON neighbours(neighbour_event_id);
CREATE INDEX IF NOT EXISTS idx_aggregate_concept            ON concept_aggregate(concept);

"""


def sqlite_connection(path: Path):
    """
    SQLite settings chosen for concurrent readers during later visualisation.
    """
    con = sqlite3.connect(path)
    con.execute( "PRAGMA journal_mode=WAL" )
    con.execute( "PRAGMA synchronous=NORMAL" )
    con.execute( "PRAGMA busy_timeout=5000" )
    return con


def initialise_database(path: Path, clear: bool = False):
    """
    Create the analysis database.

    The database is disposable. Rebuilding is preferred over migrations.
    """

    if clear and path.exists():
        logger.info(f"Removing SQLite3 db from {path}")
        path.unlink()
    con = sqlite_connection(path)
    con.executescript(SCHEMA)
    con.commit()
    return con



def ensure_documents(con, doc_ids):
    rows = [
        (str(doc_id),)
        for doc_id in doc_ids
    ]
    con.executemany(
        "INSERT OR IGNORE INTO documents ( doc_id ) VALUES (?)",
        rows,
    )


def chunks(seq, size):
    """
    Yield successive slices from a sequence.
    """
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def enrich_documents(con, pg_connection, batch_size=1000):
    doc_ids = [
        row[0]
        for row in con.execute("SELECT doc_id FROM documents WHERE title IS NULL")
    ]

    for batch in chunks(doc_ids, batch_size):
        rows = pg_connection.execute(
            """
            SELECT
                doc_id,
                title,
                author,
                pub_year,
                publisher,
                pub_place
            FROM documents
            WHERE doc_id = ANY(%s)
            """,
            (batch,),
        )

        con.executemany(
            """
            UPDATE documents
            SET
                title=?,
                author=?,
                pub_year=?,
                publisher=?,
                pub_place=?
            WHERE doc_id=?
            """,
            [
                (
                    r[1],  # title
                    r[2],  # author
                    r[3],  # pub_year
                    r[4],  # publisher
                    r[5],  # pub_place
                    r[0],  # doc_id
                )
                for r in rows
            ]
        )


def ensure_seed_events(con, lookup, event_ids):
    existing = {
        row[0]
        for row in con.execute( "SELECT event_id FROM events" )
    }

    missing = set(event_ids) - existing
    if not missing:
        return

    rows = []

    for eid in missing:
        event = lookup.get_event(eid)

        if event is None:
            raise RuntimeError( f"Tier1 event missing: {eid}" )

        rows.append( (
            int(eid),
            None,
            "neighbour",
            int(event["vector_id"]),
            event["token"],
            event["doc_id"],
            int(event["pub_year"]),
            int(event["token_idx"]),
            int(event["window_id"]),
            int(event["window_token_pos"]),
        ) )

    if not rows:
        return

    con.executemany(
        """
        INSERT OR IGNORE INTO events (
            event_id,
            concept,
            event_role,
            vector_id,
            token,
            doc_id,
            pub_year,
            token_idx,
            window_id,
            window_token_pos
        )
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )

    con.commit()


# FAISS loading
#
# Each year has independent local / medium / broad indices.
# Loading is isolated from analysis so failures are visible.

def load_indices(paths_by_year, workers=6):
    jobs = [
        (year, scale, path)
        for year, scales in paths_by_year.items()
        for scale, path in scales.items()
    ]

    logger.info( f"[tier2] loading {len(jobs)} FAISS indices" )

    indexes = {
        year: {}
        for year in paths_by_year
    }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                EeboFaissIndex.load,
                path,
            ):
            (year, scale)
            for year, scale, path in jobs
        }

        for future in as_completed(futures):
            year, scale = futures[future]
            indexes[year][scale] = future.result()

    for year, scales in indexes.items():
        for scale, index in scales.items():
            if index.ntotal == 0:
                raise RuntimeError( f"Empty FAISS index: {year}/{scale}" )

    return indexes


def analyse_concept(
    *,
    concept_name,
    concept,
    lookup,
    indexes,
    top_n=K,
    rrf_k=RRF_K,
    oversample=OVERSAMPLE,
    false_positives=None,
):
    forms = {
        f.lower()
        for f in concept.get("forms", [])
    }

    false_positives = {
        x.lower()
        for x in (false_positives or [])
    }

    logger.info( f"[tier2] {concept_name} forms: {sorted(forms)[:50]}" )

    event_ids = lookup.find_matching_event_ids(
        forms,
        false_positives,
    )

    logger.info( f"[tier2] {concept_name}: {len(event_ids)} events" )

    event_ids = lookup.find_matching_event_ids(
        forms,
        false_positives,
    )

    logger.info( f"[tier2] {concept_name}: {len(event_ids)} events" )

    if not event_ids:
        return {
            "concept": concept_name,
            "empty": True,
        }


    positions = np.asarray(
        [
            lookup.get_pos(eid)
            for eid in event_ids
        ],
        dtype=np.int64,
    )

    # Group by publication year.
    #
    # This is deliberately based on Tier 1 Zarr metadata.
    # Document metadata may be incomplete or normalised differently.

    by_year = {}
    for pos in positions:
        year = int( lookup.pub_year[pos] )

        by_year.setdefault(
            year,
            []
        ).append(pos)

    fused = {}
    for year, year_positions in by_year.items():
        result = multiscale_search(
            indexes,
            lookup,
            np.asarray(
                year_positions,
                dtype=np.int64,
            ),
            top_n,
            pub_year=year,
            rrf_k=rrf_k,
            oversample=oversample,
        )

        for pos, neighbours in zip(
            year_positions,
            result,
        ):
            fused[ int(lookup.event_id[pos]) ] = neighbours

    token_counts = Counter()
    doc_counts = Counter()
    window_counts = Counter()

    output_events = []
    for pos in positions:
        event_id = int( lookup.event_id[pos] )

        neighbours_out = []
        for item in fused[event_id]:
            neighbour_id = item["event_id"]
            if neighbour_id == event_id:
                continue

            npos = lookup.get_pos( neighbour_id )
            token = str( lookup.token[npos] )

            if token.lower() in false_positives:
                continue

            doc_id = str( lookup.doc_id[npos] )
            window_id = int( lookup.window_id[npos] )
            wpos = int( lookup.window_token_pos[npos] )
            token_counts[token] += 1
            doc_counts[doc_id] += 1

            window_counts[
                (
                    doc_id,
                    window_id,
                )
            ] += 1

            neighbours_out.append(
                {
                    "event_id": neighbour_id,
                    "vector_id": int( lookup.vector_id[npos] ),
                    "token": token,
                    "doc_id": doc_id,
                    "pub_year": int( lookup.pub_year[npos] ),
                    "token_idx": int( lookup.token_idx[npos] ),
                    "window_id": window_id,
                    "window_token_pos":
                        None
                        if wpos == _NO_WPOS
                        else wpos,
                    "score": item["rrf_score"],
                    "score_local": item.get( "score_local" ),
                    "score_medium": item.get( "score_medium" ),
                    "score_broad": item.get( "score_broad" ),
                    "depth": 1,
                    "via_event_id": None,
                }
            )

        output_events.append(
            {
                "event_id": event_id,
                "vector_id": int( lookup.vector_id[pos] ),
                "token": str( lookup.token[pos] ),
                "doc_id": str( lookup.doc_id[pos] ),
                "pub_year": int( lookup.pub_year[pos] ),
                "token_idx": int( lookup.token_idx[pos] ),
                "window_id": int( lookup.window_id[pos] ),
                "window_token_pos":
                    None
                    if int(
                        lookup.window_token_pos[pos]
                    ) == _NO_WPOS
                    else int(
                        lookup.window_token_pos[pos]
                    ),
                "neighbours": neighbours_out, }
        )

    return {
        "concept": concept_name,
        "forms": forms,
        "n_events": len(event_ids),
        "events": output_events,
        "aggregate":
            {
                "top_tokens": token_counts.most_common(top_n),
                "top_docs": doc_counts.most_common(top_n),
                "top_windows": window_counts.most_common(top_n),
            },
    }



# SQLite output
#
# The database is an analysis artefact. Writes are replace-oriented:
# rerunning a concept replaces its rows rather than merging history.

def delete_concept(con, concept):

    con.execute(
        """
        DELETE FROM neighbours
        WHERE event_id IN (
            SELECT event_id
            FROM concept_field_events
            WHERE concept = ?
              AND role = 'seed'
        )
        """,
        (concept,),
    )

    con.execute(
        """
        DELETE FROM concept_field_events
        WHERE concept = ?
        """,
        (concept,),
    )

    con.execute(
        "DELETE FROM concept_forms WHERE concept = ?",
        (concept,),
    )

    con.execute(
        "DELETE FROM concept_aggregate WHERE concept = ?",
        (concept,),
    )

    con.execute(
        "DELETE FROM concepts WHERE concept = ?",
        (concept,),
    )


def write_concept(con, data, lookup):
    concept = data["concept"]

    delete_concept(con, concept)

    seed_events = data["events"]

    event_ids = {
        event["event_id"]
        for event in seed_events
    }

    doc_ids = {
        event["doc_id"]
        for event in seed_events
    }

    ensure_seed_events(con, lookup, event_ids)

    # Documents belong to materialised Tier 2 events only.
    ensure_documents(con, doc_ids)

    field_rows = []

    for event in seed_events:
        field_rows.append(
            (
                concept,
                event["event_id"],
                "seed",
            )
        )

        for neighbour in event["neighbours"]:
            field_rows.append(
                (
                    concept,
                    neighbour["event_id"],
                    "neighbour",
                )
            )

    con.execute(
        """
        INSERT INTO concepts (
            concept,
            n_events
        )
        VALUES (?,?)
        """,
        (
            concept,
            data["n_events"],
        ),
    )

    for form in data.get("forms", []):
        con.execute(
            """
            INSERT INTO concept_forms (
                concept,
                form
            )
            VALUES (?,?)
            """,
            (
                concept,
                form,
            ),
        )

    # Update only seed events.
    # Retrieved neighbours remain Tier 1 references only.
    con.executemany(
        """
        UPDATE events
        SET
            concept = ?,
            vector_id = ?,
            token = ?,
            doc_id = ?,
            pub_year = ?,
            token_idx = ?,
            window_id = ?,
            window_token_pos = ?
        WHERE event_id = ?
        """,
        [
            (
                concept,
                event["vector_id"],
                event["token"],
                event["doc_id"],
                event["pub_year"],
                event["token_idx"],
                event["window_id"],
                event["window_token_pos"],
                event["event_id"],
            )
            for event in seed_events
        ],
    )

    neighbour_rows = []

    for event in seed_events:
        for neighbour in event["neighbours"]:
            neighbour_rows.append(
                (
                    event["event_id"],
                    neighbour["event_id"],
                    neighbour["depth"],
                    neighbour["via_event_id"],
                    neighbour["score"],
                    neighbour["score_local"],
                    neighbour["score_medium"],
                    neighbour["score_broad"],
                )
            )

    con.executemany(
        """
        INSERT OR REPLACE INTO neighbours (
            event_id,
            neighbour_event_id,
            depth,
            via_event_id,
            score,
            score_local,
            score_medium,
            score_broad
        )
        VALUES (?,?,?,?,?,?,?,?)
        """,
        neighbour_rows,
    )

    con.executemany(
        """
        INSERT OR REPLACE INTO concept_field_events (
            concept,
            event_id,
            role
        )
        VALUES (?,?,?)
        """,
        field_rows,
    )

    aggregate = data["aggregate"]

    for rank, (value, count) in enumerate(
        aggregate["top_tokens"]
    ):
        con.execute(
            """
            INSERT INTO concept_aggregate (
                concept,
                kind,
                rank,
                value,
                count
            )
            VALUES (?,?,?,?,?)
            """,
            (
                concept,
                "token",
                rank,
                value,
                count,
            ),
        )

    for rank, (value, count) in enumerate(
        aggregate["top_docs"]
    ):
        con.execute(
            """
            INSERT INTO concept_aggregate (
                concept,
                kind,
                rank,
                value,
                count
            )
            VALUES (?,?,?,?,?)
            """,
            (
                concept,
                "doc",
                rank,
                value,
                count,
            ),
        )

    for rank, ((doc_id, window_id), count) in enumerate(
        aggregate["top_windows"]
    ):
        con.execute(
            """
            INSERT INTO concept_aggregate (
                concept,
                kind,
                rank,
                window_doc_id,
                window_id,
                count
            )
            VALUES (?,?,?,?,?,?)
            """,
            (
                concept,
                "window",
                rank,
                doc_id,
                window_id,
                count,
            ),
        )


# Core is pure analysis, no I/O
def run_tier2_core(
    *,
    lookup,
    indexes,
    concepts_to_run,
    top_n: int = K,
    rrf_k: int = RRF_K,
    oversample: int = OVERSAMPLE,
    false_positives=None,
    emit=None,
):
    """
    Heart of Tier 2: neighbourhood retrieval for every requested concept.

    Takes already-constructed resources and returns a pure result dict.
    No database writes, no index loading, no side effects beyond logging.
    """
    logger.info("[tier2.run_tier2_core] Enter")
    output = {}

    for concept_name, concept in concepts_to_run:
        if emit:
            emit("concept_start", {"concept": concept_name})

        output[concept_name] = analyse_concept(
            concept_name=concept_name,
            concept=concept,
            lookup=lookup,
            indexes=indexes,
            top_n=top_n,
            rrf_k=rrf_k,
            oversample=oversample,
            false_positives=false_positives,
        )

        if emit:
            emit("concept_done", {"concept": concept_name})

    logger.info("[tier2.run_tier2_core] Leave")
    return output


# Service accepts persistent resources, orchestrates core + persistence
def run_tier2_service(
    *,
    lookup,
    indexes,
    concepts_to_run,
    db_path,
    clear: bool = False,
    top_n: int = K,
    rrf_k: int = RRF_K,
    oversample: int = OVERSAMPLE,
    false_positives=None,
    emit=None,
):
    """
    Reusable entry point for long-lived processes (UI, FastAPI, etc.).

    Expects already-built lookup and FAISS indexes.
    Calls core, then writes results to SQLite.
    """
    concept_names = [name for name, _ in concepts_to_run]
    logger = setEmit(emit, "[tier2]", {"concepts": concept_names})
    logger.info("[tier2.run_tier2_service] Enter")

    # Attach indexes so any lookup helpers that need them can see them
    if hasattr(lookup, "attach_index"):
        lookup.attach_index(indexes)

    con = initialise_database(db_path, clear=clear)

    output = run_tier2_core(
        lookup=lookup,
        indexes=indexes,
        concepts_to_run=concepts_to_run,
        top_n=top_n,
        rrf_k=rrf_k,
        oversample=oversample,
        false_positives=false_positives,
        emit=emit,
    )

    logger.info("[tier2.run_tier2_service] Writing results")
    for concept_name, data in output.items():
        if data.get("empty"):
            continue
        write_concept(con, data, lookup)

    # Enrich any newly-inserted document stubs
    try:
        pg = get_connection()
        try:
            enrich_documents(con, pg)
        finally:
            pg.close()
    except Exception as exc:
        logger.warning(f"[tier2] document enrichment skipped: {exc}")

    con.commit()
    con.close()

    logger.info("[tier2.run_tier2_service] Done")
    return output


# CLI  creates resources and hands them to the service
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--concept", default=None)
    parser.add_argument("-m", "--mask", action="store_true")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("-k", "--k", type=int, default=K)
    parser.add_argument("--rrf-k", type=int, default=RRF_K)
    parser.add_argument("--oversample", type=int, default=OVERSAMPLE)
    parser.add_argument( "-w", "--max-load-workers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("-fp", "--false-positives", type=str, default=None)
    args = parser.parse_args()

    # Paths
    if args.mask:
        zarr_path = MASKED_ZARR_PATH
        db_path = CORPUS_TIER2_MASKED_DB_PATH
    else:
        zarr_path = ZARR_PATH
        db_path = CORPUS_TIER2_DB_PATH

    # Discover & load FAISS indexes (resource construction lives here)
    years = discover_index_years(args.mask)
    if not years:
        raise RuntimeError("No FAISS indices found")

    index_paths = {
        year: faiss_index_paths(masked=args.mask, year=year)
        for year in years
    }
    indexes = load_indices(index_paths, workers=args.max_load_workers)

    # Build the Zarr event lookup
    # Restrict forms only when a single concept is requested (keeps memory proportional)
    target_forms = None
    target_fps = None
    if args.concept:
        concept_name = args.concept.upper()
        resolved = dict(resolve_concepts(
            concept=concept_name,
            false_positives=args.false_positives,
        ))
        concept = resolved[concept_name]
        target_forms = set(concept.get("forms", []))
        target_fps = set(concept.get("false_positives", []))

    lookup = ZarrEventLookup(
        zarr_path,
        # forms=target_forms,
        # false_positives=target_fps,
    )

    # Resolve which concepts still need work
    concepts = list(resolve_concepts(
        concept=args.concept,
        false_positives=args.false_positives,
    ))
    logger.info(
        "[tier2] resolved concepts: %d %s",
        len(concepts),
        [c[0] for c in concepts[:20]],
    )

    processed = set() if args.clear else get_processed_concepts(db_path)
    concepts_to_run = [c for c in concepts if c[0] not in processed]

    if not concepts_to_run:
        logger.info("[tier2.main] nothing to write — all concepts already processed")
        return

    # Hand live resources to the service
    run_tier2_service(
        lookup=lookup,
        indexes=indexes,
        concepts_to_run=concepts_to_run,
        db_path=db_path,
        clear=args.clear,
        top_n=args.k,
        rrf_k=args.rrf_k,
        oversample=args.oversample,
        false_positives=target_fps,
        emit=None,
    )

    logger.info(f"[tier2.main] Complete → {db_path}")



if __name__ == "__main__":
    main()
