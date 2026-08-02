#!/usr/bin/env python
"""
tier2/tier2_create_populate.py

Tier 2 semantic neighbourhood store — schema creation & persistence.

Responsibilities:

    in-memory neighbourhood result (from tier2_analyse)
            |
            v
    SQLite semantic neighbourhood store

Tier 2 does not define concepts. It records the empirical neighbourhood
around supplied lexical seeds and preserves provenance back to corpus
events. This module owns everything that touches the SQLite database:
schema, document/event materialisation, and concept writes. It does not
perform retrieval itself — that lives in tier2_analyse.py.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from psycopg import sql

from lib.corpus_config import (
    CORPUS_TIER2_DB_PATH,
    CORPUS_TIER2_MASKED_DB_PATH,
)

from lib.corpus_db import get_connection
from lib.corpus_logging import logger, setEmit
from lib.get_processed_concepts import get_processed_concepts

from tier2.tier2_analyse import (
    K,
    RRF_K,
    OVERSAMPLE,
    run_tier2_core,
    build_resources,
)



# SQLite schema
#
# This database is disposable analysis output.
# No migration layer is required.
# SCHEMA_PATH = Path(__file__).with_name("schema.sql")
# with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
#     SCHEMA = f.read()


SCHEMA = """

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

CREATE TABLE IF NOT EXISTS documents (
    corpus TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    title TEXT,
    author TEXT,
    pub_year INTEGER,
    publisher TEXT,
    pub_place TEXT,
    normalized_places TEXT,
    lat REAL,
    lng REAL,
    PRIMARY KEY(corpus, doc_id)
);

CREATE TABLE IF NOT EXISTS events (
    corpus TEXT NOT NULL,
    event_id INTEGER PRIMARY KEY,
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
    concept TEXT NOT NULL,
    event_id INTEGER NOT NULL,
    neighbour_event_id INTEGER NOT NULL,

    depth INTEGER NOT NULL DEFAULT 1,
    via_event_id INTEGER,

    score REAL,
    score_local REAL,
    score_medium REAL,
    score_broad REAL,

    PRIMARY KEY(
        concept,
        event_id,
        neighbour_event_id,
        depth
    )
);

-- concept_field_events is the authoritative relationship between
-- corpus observations and concepts. Event role is concept-relative
-- and therefore does not belong on events.
CREATE TABLE IF NOT EXISTS concept_field_events (
    concept TEXT NOT NULL,
    event_id INTEGER NOT NULL,
    role TEXT NOT NULL,

    PRIMARY KEY(
        concept,
        event_id
    )
);

CREATE INDEX IF NOT EXISTS idx_field_events_concept      ON concept_field_events(concept);
CREATE INDEX IF NOT EXISTS idx_field_events_event        ON concept_field_events(event_id);
CREATE INDEX IF NOT EXISTS idx_field_events_concept_role ON concept_field_events(concept, role);

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

CREATE INDEX IF NOT EXISTS idx_events_document              ON events(corpus, doc_id);
CREATE INDEX IF NOT EXISTS idx_events_event_id              ON events(event_id);
CREATE INDEX IF NOT EXISTS idx_neighbours_concept_event     ON neighbours(concept,event_id);
CREATE INDEX IF NOT EXISTS idx_neighbours_concept_neighbour ON neighbours(concept, neighbour_event_id);
CREATE INDEX IF NOT EXISTS idx_aggregate_concept            ON concept_aggregate(concept);

CREATE INDEX IF NOT EXISTS idx_events_year                  ON events(corpus, pub_year);
CREATE INDEX IF NOT EXISTS idx_events_document              ON events(corpus, doc_id);
CREATE INDEX IF NOT EXISTS idx_events_event_id              ON events(event_id);
CREATE INDEX IF NOT EXISTS idx_events_token                 ON events(token);
CREATE INDEX IF NOT EXISTS idx_events_token_position        ON events(corpus, doc_id, token_idx);

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
        (corpus, str(doc_id))
        for corpus, doc_id in doc_ids
    ]
    con.executemany(
        "INSERT OR IGNORE INTO documents ( corpus, doc_id ) VALUES (?, ?)",
        rows,
    )


def chunks(seq, size):
    """
    Yield successive slices from a sequence.
    """
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def enrich_documents(con, pg_connection, batch_size=1000):
    doc_keys = [
        (row[0], row[1])
        for row in con.execute(
            """
            SELECT corpus, doc_id
            FROM documents
            WHERE title IS NULL
            """
        )
    ]

    for batch in chunks(doc_keys, batch_size):

        placeholders = sql.SQL(",").join(
            sql.SQL("(%s,%s)")
            for _ in batch
        )

        rows = pg_connection.execute(
            sql.SQL(
                """
                SELECT
                    corpus,
                    doc_id,
                    title,
                    author,
                    pub_year,
                    publisher,
                    pub_place
                FROM documents
                WHERE (corpus, doc_id) IN ({})
                """
            ).format(placeholders),
            [
                value
                for key in batch
                for value in key
            ],
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
            WHERE corpus=? AND doc_id=?
            """,
            [
                (
                    r[2],
                    r[3],
                    r[4],
                    r[5],
                    r[6],
                    r[0],
                    r[1],
                )
                for r in rows
            ],
        )

def ensure_events(con, lookup, event_ids):
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

        rows.append((
            int(eid),
            event["corpus"],
            int(event["vector_id"]),
            event["token"],
            event["doc_id"],
            int(event["pub_year"]),
            int(event["token_idx"]),
            int(event["window_id"]),
            int(event["window_token_pos"]),
        ))

    if not rows:
        return

    con.executemany(
        """
        INSERT OR IGNORE INTO events (
            event_id,
            corpus,
            vector_id,
            token,
            doc_id,
            pub_year,
            token_idx,
            window_id,
            window_token_pos
        )
        VALUES (?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )

    con.commit()


# SQLite output
#
# The database is an analysis artefact. Writes are replace-oriented:
# rerunning a concept replaces its rows rather than merging history.

def delete_concept(con, concept):

    con.execute(
        """ DELETE FROM neighbours WHERE concept = ? """,
        (concept,),
    )

    con.execute(
        """ DELETE FROM concept_field_events WHERE concept = ? """,
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



def write_concept(con, data, lookup):
    concept = data["concept"]

    delete_concept(con, concept)

    seed_events = data["events"]

    event_ids = {
        event["event_id"]
        for event in seed_events
    }

    ensure_events(con, lookup, event_ids)

    neighbour_event_ids = {
        neighbour["event_id"]
        for event in seed_events
        for neighbour in event["neighbours"]
    }

    ensure_events(con, lookup, neighbour_event_ids)

    all_event_ids = event_ids | neighbour_event_ids

    doc_keys = {
        (
            str(lookup.corpus[pos]),
            str(lookup.doc_id[pos]),
        )
        for eid in all_event_ids
        for pos in [lookup.get_pos(eid)]
    }

    # Documents belong to materialised Tier 2 events only.
    ensure_documents(con, doc_keys)

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
        "INSERT INTO concepts ( concept, n_events ) VALUES (?,?)",
        (
            concept,
            data["n_events"],
        ),
    )

    for form in data.get("forms", []):
        con.execute(
            "INSERT INTO concept_forms ( concept, form ) VALUES (?,?)",
            (
                concept,
                form,
            ),
        )

    con.executemany(
        """
        UPDATE events
        SET
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
                    concept,
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
            concept,
            event_id,
            neighbour_event_id,
            depth,
            via_event_id,
            score,
            score_local,
            score_medium,
            score_broad
        )
        VALUES (?,?,?,?,?,?,?,?,?)
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


# Service-style entrypoint: takes a pre-computed analysis output dict
# (e.g. from tier2_analyse.run_tier2_core) plus the lookup used to build
# it, and persists everything to SQLite. No retrieval happens here.
def run_tier2_populate_service(
    *,
    output,
    lookup,
    db_path,
    clear: bool = False,
    emit=None,
):
    """
    Reusable entry point for long-lived processes (UI, FastAPI, etc.)
    that already have a computed analysis result and just need it
    written to the store.
    """
    concept_names = list(output.keys())
    logger = setEmit(emit, "[tier2.populate]", {"concepts": concept_names})
    logger.info("[tier2.run_tier2_populate_service] Enter")

    con = initialise_database(db_path, clear=clear)

    logger.info("[tier2.run_tier2_populate_service] Writing results")
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

    logger.info("[tier2.run_tier2_populate_service] Done")
    return output


# Combined service entrypoint: runs analysis (tier2_analyse.run_tier2_core)
# then persists it. Kept here so callers that want "analyse + write" in
# one call don't have to wire the two modules together themselves.
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
    Calls the analysis core, then writes results to SQLite.
    """
    concept_names = [name for name, _ in concepts_to_run]
    logger = setEmit(emit, "[tier2]", {"concepts": concept_names})
    logger.info("[tier2.run_tier2_service] Enter")

    # Attach indexes so any lookup helpers that need them can see them
    if hasattr(lookup, "attach_index"):
        lookup.attach_index(indexes)

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

    return run_tier2_populate_service(
        output=output,
        lookup=lookup,
        db_path=db_path,
        clear=clear,
        emit=emit,
    )


# CLI — creates resources and either loads a precomputed analysis dump
# (--input, produced by tier2_analyse.py --out) or runs analysis itself,
# then populates the database.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--concept", default=None)
    parser.add_argument("-m", "--mask", action="store_true")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("-k", "--k", type=int, default=K)
    parser.add_argument("--rrf-k", type=int, default=RRF_K)
    parser.add_argument("--oversample", type=int, default=OVERSAMPLE)
    parser.add_argument( "-w", "--max-load-workers", type=int, default=6)
    parser.add_argument("-fp", "--false-positives", type=str, default=None)
    parser.add_argument("-i", "--input", type=str, default=None,
                         help="Path to a JSON analysis dump from tier2_analyse.py --out, "
                              "instead of recomputing it here")
    args = parser.parse_args()

    # Paths
    if args.mask:
        db_path = CORPUS_TIER2_MASKED_DB_PATH
    else:
        db_path = CORPUS_TIER2_DB_PATH

    lookup, indexes, concepts, target_fps = build_resources(
        concept=args.concept,
        mask=args.mask,
        false_positives=args.false_positives,
        max_load_workers=args.max_load_workers,
    )

    logger.info(
        "[tier2_create_populate] resolved concepts: %d %s",
        len(concepts),
        [c[0] for c in concepts[:20]],
    )

    processed = set() if args.clear else get_processed_concepts(db_path)
    concepts_to_run = [c for c in concepts if c[0] not in processed]

    if not concepts_to_run and not args.input:
        logger.info("[tier2_create_populate.main] nothing to write — all concepts already processed")
        return

    if args.input:
        import json
        with open(args.input) as f:
            output = json.load(f)
        run_tier2_populate_service(
            output=output,
            lookup=lookup,
            db_path=db_path,
            clear=args.clear,
            emit=None,
        )
    else:
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

    logger.info(f"[tier2_create_populate.main] Complete → {db_path}")


if __name__ == "__main__":
    main()
