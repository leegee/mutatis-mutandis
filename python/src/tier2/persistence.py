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

from lib.eebo_db import get_connection
from lib.eebo_logging import logger

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
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA busy_timeout=5000")
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
