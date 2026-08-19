"""
tier2.persistence

SQLite schema, connection helpers, and all write paths that materialise
Tier 2 analysis results.

The database is disposable analysis output.
No migration layer is required.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from lib.corpus_db import get_connection
from lib.corpus_logging import logger

# SQLite schema
#
# This database is disposable analysis output.
# No migration layer is required.
#
# Tables and indexes are split so bulk loads can create tables only,
# stream all rows, then build indexes once at the end.


SCHEMA_TABLES = """

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

"""

SCHEMA_INDEXES = """

CREATE INDEX IF NOT EXISTS idx_field_events_concept ON concept_field_events(concept);
CREATE INDEX IF NOT EXISTS idx_field_events_event   ON concept_field_events(event_id);
CREATE INDEX IF NOT EXISTS idx_events_concept               ON events(concept);
CREATE INDEX IF NOT EXISTS idx_events_year                  ON events(concept, pub_year);
CREATE INDEX IF NOT EXISTS idx_events_event_id              ON events(event_id);
CREATE INDEX IF NOT EXISTS idx_neighbours_event             ON neighbours(event_id);
CREATE INDEX IF NOT EXISTS idx_neighbours_neighbour_event   ON neighbours(neighbour_event_id);
CREATE INDEX IF NOT EXISTS idx_aggregate_concept            ON concept_aggregate(concept);

"""

# Full schema (tables + indexes) for callers that want everything at once.
SCHEMA = SCHEMA_TABLES + SCHEMA_INDEXES


def sqlite_connection(path: Path):
    """
    Open SQLite with settings suitable for concurrent readers.
    """
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA busy_timeout=5000")
    return con


def begin_bulk_load(con):
    """
    PRAGMAs optimised for streaming writes. Indexes must not exist yet
    (see initialise_database / create_indexes). Call restore_reader_pragmas
    after create_indexes when the DB is ready for the UI.
    """
    con.execute("PRAGMA journal_mode=MEMORY")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA temp_store=MEMORY")
    # Negative cache_size = KB; ~256 MB page cache during load.
    con.execute("PRAGMA cache_size=-262144")
    con.execute("PRAGMA busy_timeout=5000")


def restore_reader_pragmas(con):
    """
    Settings chosen for concurrent readers during later visualisation.
    """
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA temp_store=DEFAULT")
    con.execute("PRAGMA cache_size=-2000")
    con.execute("PRAGMA busy_timeout=5000")


def initialise_database(path: Path, clear: bool = False):
    """
    Create the analysis database: tables only, bulk-load PRAGMAs on.

    Indexes are deliberately deferred until create_indexes() so the
    streaming write path does not maintain secondary B-trees per row.
    """
    if clear and path.exists():
        logger.info(f"Removing SQLite3 db from {path}")
        path.unlink()
    con = sqlite3.connect(path)
    begin_bulk_load(con)
    con.executescript(SCHEMA_TABLES)
    con.commit()
    return con


def create_indexes(con):
    """
    Build secondary indexes after all bulk writes have finished.
    """
    logger.info("[tier2] creating SQLite indexes")
    con.executescript(SCHEMA_INDEXES)
    con.commit()


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


def ensure_events(con, lookup, event_ids):
    existing = {
        row[0]
        for row in con.execute("SELECT event_id FROM events")
    }

    missing = set(event_ids) - existing
    if not missing:
        return

    rows = []

    for eid in missing:
        event = lookup.get_event(eid)

        if event is None:
            raise RuntimeError(f"Tier1 event missing: {eid}")

        rows.append((
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
        ))

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

    ensure_events(con, lookup, event_ids)

    neighbour_event_ids = {
        neighbour["event_id"]
        for event in seed_events
        for neighbour in event["neighbours"]
    }

    ensure_events(con, lookup, neighbour_event_ids)

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
            # A neighbour that is itself one of this concept's seed
            # events must stay marked 'seed', not get demoted to
            # 'neighbour'. concept_field_events is keyed on
            # (concept, event_id) with INSERT OR REPLACE, so without
            # this check the final role for such an event would depend
            # on write order rather than on what it actually is —
            # entirely plausible with real semantic neighbours, where
            # two seed events can easily rank as each other's nearest
            # neighbours.
            if neighbour["event_id"] in event_ids:
                continue

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


# --------------------------------------------------------------------
# Streaming write path
#
# Mirrors write_concept above, but split into three calls so a caller
# can write one bounded-size chunk of a concept's seed events at a
# time (see tier2.analysis.iter_concept_batches), instead of building
# the concept's whole events/neighbours payload before writing
# anything. For concepts matching a small number of events this is
# equivalent to write_concept; for concepts matching hundreds of
# thousands of events (common words in a large corpus), it's the
# difference between bounded and unbounded peak memory.
# --------------------------------------------------------------------

def start_concept(con, concept):
    """
    Clear out any prior materialisation of `concept` before writing a
    fresh streamed run. Call once, before the first write_concept_batch
    call for this concept.
    """
    delete_concept(con, concept)


def write_concept_batch(con, concept, lookup, seed_events_batch, seed_ids=None):
    """
    Write one chunk of a concept's seed events (each already carrying
    its resolved "neighbours" list) to SQLite. Safe to call repeatedly
    for the same concept, and safe to commit after each call — nothing
    here depends on a later batch.

    `seed_ids` should be the *full* set of this concept's seed event
    IDs (not just this batch's) — e.g. resolved["event_ids_set"] from
    tier2.analysis.resolve_concept_positions. It's used to avoid ever
    writing a 'neighbour' row for an event that is itself one of this
    concept's seeds; without it, whether such an event ends up correctly
    marked 'seed' or wrongly demoted to 'neighbour' would depend on
    which batch happens to write last, since concept_field_events is
    keyed on (concept, event_id) with INSERT OR REPLACE. If seed_ids
    isn't supplied, this batch falls back to checking only against
    seed_events_batch itself, which is not sufficient across batches.

    Does not touch the concepts / concept_forms / concept_aggregate
    tables; call finish_concept once, after the last batch, for those.
    """
    if not seed_events_batch:
        return

    event_ids = {
        event["event_id"]
        for event in seed_events_batch
    }

    known_seed_ids = seed_ids if seed_ids is not None else event_ids

    doc_ids = {
        event["doc_id"]
        for event in seed_events_batch
    }

    ensure_events(con, lookup, event_ids)

    neighbour_event_ids = {
        neighbour["event_id"]
        for event in seed_events_batch
        for neighbour in event["neighbours"]
    }

    if neighbour_event_ids:
        ensure_events(con, lookup, neighbour_event_ids)

    # Documents belong to materialised Tier 2 events only.
    ensure_documents(con, doc_ids)

    field_rows = []

    for event in seed_events_batch:
        field_rows.append(
            (
                concept,
                event["event_id"],
                "seed",
            )
        )

        for neighbour in event["neighbours"]:
            # See docstring: never mark a concept's own seed event as
            # a 'neighbour', regardless of which batch writes it last.
            if neighbour["event_id"] in known_seed_ids:
                continue

            field_rows.append(
                (
                    concept,
                    neighbour["event_id"],
                    "neighbour",
                )
            )

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
            for event in seed_events_batch
        ],
    )

    neighbour_rows = []

    for event in seed_events_batch:
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


def finish_concept(con, concept, forms, n_events, aggregate):
    """
    Write the summary rows for a concept — concepts, concept_forms,
    concept_aggregate — once all of its batches have been written via
    write_concept_batch. Call exactly once per concept, last.
    """
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
            n_events,
        ),
    )

    for form in forms or []:
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


# --------------------------------------------------------------------
# Postgres staging
#
# Bulk construction target. Analysis streams into UNLOGGED stage tables;
# when the run finishes, dump_pg_stage_to_sqlite() publishes the familiar
# SQLite artifact (tables first, indexes last). The browser / WASM UI
# keeps consuming SQLite; Postgres is only the write engine.
# --------------------------------------------------------------------

PG_STAGE_SCHEMA = """
CREATE SCHEMA IF NOT EXISTS tier2_stage;

DROP TABLE IF EXISTS tier2_stage.concept_cluster_info CASCADE;
DROP TABLE IF EXISTS tier2_stage.concept_aggregate CASCADE;
DROP TABLE IF EXISTS tier2_stage.concept_forms CASCADE;
DROP TABLE IF EXISTS tier2_stage.concepts CASCADE;
DROP TABLE IF EXISTS tier2_stage.concept_field_events CASCADE;
DROP TABLE IF EXISTS tier2_stage.neighbours CASCADE;
DROP TABLE IF EXISTS tier2_stage.events CASCADE;
DROP TABLE IF EXISTS tier2_stage.documents CASCADE;

CREATE UNLOGGED TABLE tier2_stage.documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT,
    author TEXT,
    pub_year INTEGER,
    publisher TEXT,
    pub_place TEXT,
    normalized_places TEXT,
    lat DOUBLE PRECISION,
    lng DOUBLE PRECISION
);

CREATE UNLOGGED TABLE tier2_stage.concepts (
    concept TEXT PRIMARY KEY,
    n_events INTEGER NOT NULL
);

CREATE UNLOGGED TABLE tier2_stage.concept_forms (
    concept TEXT NOT NULL,
    form TEXT NOT NULL,
    is_false_positive INTEGER DEFAULT 0,
    PRIMARY KEY (concept, form)
);

CREATE UNLOGGED TABLE tier2_stage.events (
    event_id BIGINT PRIMARY KEY,
    concept TEXT,
    event_role TEXT NOT NULL DEFAULT 'neighbour',
    vector_id BIGINT,
    token TEXT,
    doc_id TEXT,
    pub_year INTEGER,
    token_idx INTEGER,
    window_id INTEGER,
    window_token_pos INTEGER,
    nx DOUBLE PRECISION,
    ny DOUBLE PRECISION,
    gnx DOUBLE PRECISION,
    gny DOUBLE PRECISION,
    cluster_id INTEGER,
    cluster_label TEXT
);

CREATE UNLOGGED TABLE tier2_stage.neighbours (
    event_id BIGINT NOT NULL,
    neighbour_event_id BIGINT NOT NULL,
    depth INTEGER NOT NULL DEFAULT 1,
    via_event_id BIGINT,
    score DOUBLE PRECISION,
    score_local DOUBLE PRECISION,
    score_medium DOUBLE PRECISION,
    score_broad DOUBLE PRECISION,
    PRIMARY KEY (event_id, neighbour_event_id, depth)
);

CREATE UNLOGGED TABLE tier2_stage.concept_field_events (
    concept TEXT NOT NULL,
    event_id BIGINT NOT NULL,
    role TEXT NOT NULL,
    PRIMARY KEY (concept, event_id)
);

CREATE UNLOGGED TABLE tier2_stage.concept_aggregate (
    id BIGSERIAL PRIMARY KEY,
    concept TEXT NOT NULL,
    kind TEXT NOT NULL,
    rank INTEGER NOT NULL,
    value TEXT,
    window_doc_id TEXT,
    window_id INTEGER,
    count INTEGER NOT NULL
);

CREATE UNLOGGED TABLE tier2_stage.concept_cluster_info (
    concept TEXT NOT NULL,
    cluster_id INTEGER NOT NULL,
    cluster_label TEXT,
    centroid_nx DOUBLE PRECISION,
    centroid_ny DOUBLE PRECISION,
    centroid_gnx DOUBLE PRECISION,
    centroid_gny DOUBLE PRECISION,
    centroid_vector BYTEA,
    point_count INTEGER,
    description TEXT,
    llm_model TEXT,
    llm_prompt TEXT,
    llm_timestamp TEXT,
    llm_sample_size INTEGER,
    llm_sample_event_ids TEXT,
    llm_concentration DOUBLE PRECISION,
    PRIMARY KEY (concept, cluster_id)
);
"""


def _pg_executemany(pg, sql, rows):
    """
    executemany across psycopg2 connections and bare cursors.
    """
    if not rows:
        return
    if hasattr(pg, "executemany") and not hasattr(pg, "cursor"):
        # already a cursor-like object
        pg.executemany(sql, rows)
        return
    cur = pg.cursor() if hasattr(pg, "cursor") else pg
    cur.executemany(sql, rows)
    if hasattr(pg, "commit") and cur is not pg:
        pass  # caller commits


def _pg_execute(pg, sql, params=None):
    if hasattr(pg, "cursor"):
        cur = pg.cursor()
        if params is None:
            cur.execute(sql)
        else:
            cur.execute(sql, params)
        return cur
    if params is None:
        return pg.execute(sql)
    return pg.execute(sql, params)


def initialise_pg_stage(pg):
    """
    Drop and recreate UNLOGGED stage tables. Disposable; no migrations.
    """
    logger.info("[tier2] initialising Postgres stage schema")
    if hasattr(pg, "cursor"):
        cur = pg.cursor()
        cur.execute(PG_STAGE_SCHEMA)
        pg.commit()
    else:
        # connection-like that supports executescript-style multi-statement
        for stmt in PG_STAGE_SCHEMA.split(";"):
            stmt = stmt.strip()
            if stmt:
                pg.execute(stmt)
        if hasattr(pg, "commit"):
            pg.commit()


def delete_concept_pg(pg, concept):
    _pg_execute(
        pg,
        """
        DELETE FROM tier2_stage.neighbours
        WHERE event_id IN (
            SELECT event_id FROM tier2_stage.concept_field_events
            WHERE concept = %s AND role = 'seed'
        )
        """,
        (concept,),
    )
    _pg_execute(
        pg,
        "DELETE FROM tier2_stage.concept_field_events WHERE concept = %s",
        (concept,),
    )
    _pg_execute(
        pg,
        "DELETE FROM tier2_stage.concept_forms WHERE concept = %s",
        (concept,),
    )
    _pg_execute(
        pg,
        "DELETE FROM tier2_stage.concept_aggregate WHERE concept = %s",
        (concept,),
    )
    _pg_execute(
        pg,
        "DELETE FROM tier2_stage.concepts WHERE concept = %s",
        (concept,),
    )


def start_concept_pg(pg, concept):
    delete_concept_pg(pg, concept)


def ensure_documents_pg(pg, doc_ids):
    rows = [(str(d),) for d in doc_ids]
    _pg_executemany(
        pg,
        """
        INSERT INTO tier2_stage.documents (doc_id)
        VALUES (%s)
        ON CONFLICT (doc_id) DO NOTHING
        """,
        rows,
    )


def ensure_events_pg(pg, lookup, event_ids):
    if not event_ids:
        return
    rows = []
    for eid in event_ids:
        event = lookup.get_event(int(eid))
        if event is None:
            raise RuntimeError(f"Tier1 event missing: {eid}")
        rows.append((
            int(eid),
            None,
            "neighbour",
            int(event["vector_id"]),
            event["token"],
            event["doc_id"],
            int(event["pub_year"]),
            int(event["token_idx"]),
            int(event["window_id"]),
            int(event["window_token_pos"])
            if event["window_token_pos"] is not None
            else None,
        ))
    _pg_executemany(
        pg,
        """
        INSERT INTO tier2_stage.events (
            event_id, concept, event_role, vector_id, token,
            doc_id, pub_year, token_idx, window_id, window_token_pos
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (event_id) DO NOTHING
        """,
        rows,
    )


def write_concept_batch_pg(pg, concept, lookup, seed_events_batch, seed_ids=None):
    """
    Postgres counterpart of write_concept_batch. Same seed-role rules.
    """
    if not seed_events_batch:
        return

    event_ids = {event["event_id"] for event in seed_events_batch}
    known_seed_ids = seed_ids if seed_ids is not None else event_ids
    doc_ids = {event["doc_id"] for event in seed_events_batch}

    ensure_events_pg(pg, lookup, event_ids)

    neighbour_event_ids = {
        neighbour["event_id"]
        for event in seed_events_batch
        for neighbour in event["neighbours"]
    }
    if neighbour_event_ids:
        ensure_events_pg(pg, lookup, neighbour_event_ids)

    ensure_documents_pg(pg, doc_ids)

    field_rows = []
    for event in seed_events_batch:
        field_rows.append((concept, event["event_id"], "seed"))
        for neighbour in event["neighbours"]:
            if neighbour["event_id"] in known_seed_ids:
                continue
            field_rows.append(
                (concept, neighbour["event_id"], "neighbour")
            )

    # Stamp seed rows with concept metadata
    _pg_executemany(
        pg,
        """
        UPDATE tier2_stage.events SET
            concept = %s,
            vector_id = %s,
            token = %s,
            doc_id = %s,
            pub_year = %s,
            token_idx = %s,
            window_id = %s,
            window_token_pos = %s
        WHERE event_id = %s
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
            for event in seed_events_batch
        ],
    )

    neighbour_rows = []
    for event in seed_events_batch:
        for neighbour in event["neighbours"]:
            neighbour_rows.append((
                event["event_id"],
                neighbour["event_id"],
                neighbour["depth"],
                neighbour["via_event_id"],
                neighbour["score"],
                neighbour["score_local"],
                neighbour["score_medium"],
                neighbour["score_broad"],
            ))

    _pg_executemany(
        pg,
        """
        INSERT INTO tier2_stage.neighbours (
            event_id, neighbour_event_id, depth, via_event_id,
            score, score_local, score_medium, score_broad
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (event_id, neighbour_event_id, depth) DO UPDATE SET
            via_event_id = EXCLUDED.via_event_id,
            score = EXCLUDED.score,
            score_local = EXCLUDED.score_local,
            score_medium = EXCLUDED.score_medium,
            score_broad = EXCLUDED.score_broad
        """,
        neighbour_rows,
    )

    _pg_executemany(
        pg,
        """
        INSERT INTO tier2_stage.concept_field_events (
            concept, event_id, role
        )
        VALUES (%s,%s,%s)
        ON CONFLICT (concept, event_id) DO UPDATE SET
            role = EXCLUDED.role
        """,
        field_rows,
    )


def finish_concept_pg(pg, concept, forms, n_events, aggregate):
    _pg_execute(
        pg,
        """
        INSERT INTO tier2_stage.concepts (concept, n_events)
        VALUES (%s, %s)
        ON CONFLICT (concept) DO UPDATE SET n_events = EXCLUDED.n_events
        """,
        (concept, n_events),
    )

    form_rows = [(concept, form) for form in (forms or [])]
    _pg_executemany(
        pg,
        """
        INSERT INTO tier2_stage.concept_forms (concept, form)
        VALUES (%s, %s)
        ON CONFLICT (concept, form) DO NOTHING
        """,
        form_rows,
    )

    agg_rows = []
    for rank, (value, count) in enumerate(aggregate["top_tokens"]):
        agg_rows.append((concept, "token", rank, value, None, None, count))
    for rank, (value, count) in enumerate(aggregate["top_docs"]):
        agg_rows.append((concept, "doc", rank, value, None, None, count))
    for rank, ((doc_id, window_id), count) in enumerate(aggregate["top_windows"]):
        agg_rows.append((concept, "window", rank, None, doc_id, window_id, count))

    _pg_executemany(
        pg,
        """
        INSERT INTO tier2_stage.concept_aggregate (
            concept, kind, rank, value, window_doc_id, window_id, count
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s)
        """,
        agg_rows,
    )


def _pg_fetchall(pg, sql):
    cur = _pg_execute(pg, sql)
    if hasattr(cur, "fetchall"):
        return cur.fetchall()
    return list(cur)


def dump_pg_stage_to_sqlite(pg, sqlite_path, clear=True):
    """
    Publish the stage tables into a SQLite file (tables → rows → indexes).
    """
    logger.info(f"[tier2] dumping Postgres stage → {sqlite_path}")
    con = initialise_database(sqlite_path, clear=clear)

    def copy_table(pg_sql, insert_sql, batch_size=5000):
        rows = _pg_fetchall(pg, pg_sql)
        for batch in chunks(rows, batch_size):
            con.executemany(insert_sql, batch)
        con.commit()

    copy_table(
        "SELECT doc_id, title, author, pub_year, publisher, pub_place, "
        "normalized_places, lat, lng FROM tier2_stage.documents",
        "INSERT OR IGNORE INTO documents ("
        "doc_id, title, author, pub_year, publisher, pub_place, "
        "normalized_places, lat, lng) VALUES (?,?,?,?,?,?,?,?,?)",
    )

    copy_table(
        "SELECT event_id, concept, event_role, vector_id, token, doc_id, "
        "pub_year, token_idx, window_id, window_token_pos, "
        "nx, ny, gnx, gny, cluster_id, cluster_label "
        "FROM tier2_stage.events",
        "INSERT OR IGNORE INTO events ("
        "event_id, concept, event_role, vector_id, token, doc_id, "
        "pub_year, token_idx, window_id, window_token_pos, "
        "nx, ny, gnx, gny, cluster_id, cluster_label) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
    )

    copy_table(
        "SELECT event_id, neighbour_event_id, depth, via_event_id, "
        "score, score_local, score_medium, score_broad "
        "FROM tier2_stage.neighbours",
        "INSERT OR REPLACE INTO neighbours ("
        "event_id, neighbour_event_id, depth, via_event_id, "
        "score, score_local, score_medium, score_broad) "
        "VALUES (?,?,?,?,?,?,?,?)",
    )

    copy_table(
        "SELECT concept, event_id, role FROM tier2_stage.concept_field_events",
        "INSERT OR REPLACE INTO concept_field_events ("
        "concept, event_id, role) VALUES (?,?,?)",
    )

    copy_table(
        "SELECT concept, n_events FROM tier2_stage.concepts",
        "INSERT OR REPLACE INTO concepts (concept, n_events) VALUES (?,?)",
    )

    copy_table(
        "SELECT concept, form, is_false_positive FROM tier2_stage.concept_forms",
        "INSERT OR IGNORE INTO concept_forms ("
        "concept, form, is_false_positive) VALUES (?,?,?)",
    )

    copy_table(
        "SELECT concept, kind, rank, value, window_doc_id, window_id, count "
        "FROM tier2_stage.concept_aggregate ORDER BY id",
        "INSERT INTO concept_aggregate ("
        "concept, kind, rank, value, window_doc_id, window_id, count) "
        "VALUES (?,?,?,?,?,?,?)",
    )


    copy_table(
        "SELECT concept, cluster_id, cluster_label, "
        "centroid_nx, centroid_ny, centroid_gnx, centroid_gny, "
        "centroid_vector, point_count, description, "
        "llm_model, llm_prompt, llm_timestamp, llm_sample_size, "
        "llm_sample_event_ids, llm_concentration "
        "FROM tier2_stage.concept_cluster_info",
        "INSERT OR REPLACE INTO concept_cluster_info ("
        "concept, cluster_id, cluster_label, "
        "centroid_nx, centroid_ny, centroid_gnx, centroid_gny, "
        "centroid_vector, point_count, description, "
        "llm_model, llm_prompt, llm_timestamp, llm_sample_size, "
        "llm_sample_event_ids, llm_concentration) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
    )

    create_indexes(con)
    restore_reader_pragmas(con)
    con.commit()
    con.close()
    logger.info(f"[tier2] SQLite dump complete → {sqlite_path}")


# --------------------------------------------------------------------
# Tier 3 read/write against the live Postgres stage
#
# Intended flow:
#   Tier 2  -> UNLOGGED tier2_stage.*  (keep alive)
#   Tier 3  -> read field events, write geometry + cluster_info in stage
#   publish -> dump_pg_stage_to_sqlite() when the UI needs a file
# --------------------------------------------------------------------


def load_event_rows_pg(pg, concept):
    """
    Load the empirical semantic field for a concept from the PG stage.
    Same contract as lib.cluster.load_event_rows (sqlite).
    """
    cur = _pg_execute(
        pg,
        """
        SELECT e.event_id, e.vector_id
        FROM tier2_stage.concept_field_events f
        JOIN tier2_stage.events e ON e.event_id = f.event_id
        WHERE f.concept = %s
        ORDER BY e.event_id
        """,
        (concept,),
    )
    if hasattr(cur, "fetchall"):
        return cur.fetchall()
    return list(cur)


def list_stage_concepts_pg(pg):
    cur = _pg_execute(
        pg,
        "SELECT concept FROM tier2_stage.concepts ORDER BY concept",
    )
    rows = cur.fetchall() if hasattr(cur, "fetchall") else list(cur)
    return [r[0] for r in rows]


def write_geometry_pg(
    pg,
    event_ids,
    local_coords,
    global_coords,
    clusters,
):
    rows = []
    for idx, event_id in enumerate(event_ids):
        rows.append((
            float(local_coords[idx][0]),
            float(local_coords[idx][1]),
            float(global_coords[idx][0]),
            float(global_coords[idx][1]),
            int(clusters[idx]),
            ("noise" if int(clusters[idx]) == -1 else None),
            int(event_id),
        ))
    _pg_executemany(
        pg,
        """
        UPDATE tier2_stage.events SET
            nx = %s,
            ny = %s,
            gnx = %s,
            gny = %s,
            cluster_id = %s,
            cluster_label = %s
        WHERE event_id = %s
        """,
        rows,
    )


def write_cluster_info_pg(
    pg,
    concept,
    vectors,
    local_coords,
    global_coords,
    clusters,
    vector_to_blob,
):
    """
    Replace concept_cluster_info rows for `concept` in the PG stage.
    `vector_to_blob` is injected to avoid a hard dep on lib.sqlite_vector_blob
    from this module.
    """
    _pg_execute(
        pg,
        "DELETE FROM tier2_stage.concept_cluster_info WHERE concept = %s",
        (concept,),
    )

    data = []
    for cluster_id in sorted(set(int(x) for x in clusters)):
        mask = (clusters == cluster_id)
        if not mask.any():
            continue
        centroid_vector = vectors[mask].mean(axis=0).astype("float32")
        data.append((
            concept,
            int(cluster_id),
            "noise" if cluster_id == -1 else None,
            float(local_coords[mask, 0].mean()),
            float(local_coords[mask, 1].mean()),
            float(global_coords[mask, 0].mean()),
            float(global_coords[mask, 1].mean()),
            vector_to_blob(centroid_vector),
            int(mask.sum()),
            None,  # description
        ))

    _pg_executemany(
        pg,
        """
        INSERT INTO tier2_stage.concept_cluster_info (
            concept, cluster_id, cluster_label,
            centroid_nx, centroid_ny,
            centroid_gnx, centroid_gny,
            centroid_vector, point_count, description
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        data,
    )
