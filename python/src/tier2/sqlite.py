"""
tier2/sqlite.py
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path

from lib.corpus_logging import logger


_SCHEMA_INIT = """
CREATE TABLE IF NOT EXISTS concepts (
    concept  TEXT PRIMARY KEY,
    n_events INTEGER NOT NULL
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
    event_id             INTEGER NOT NULL,
    neighbour_event_id   INTEGER NOT NULL,
    vector_id            INTEGER,
    token                TEXT,
    doc_id               TEXT,
    pub_year             INTEGER,
    token_idx             INTEGER,
    window_id             INTEGER,
    window_token_pos     INTEGER,
    score                REAL,

    PRIMARY KEY (event_id, neighbour_event_id),
    FOREIGN KEY (event_id) REFERENCES events(event_id)
);

CREATE TABLE IF NOT EXISTS concept_field_events (
    concept  TEXT    NOT NULL,
    event_id INTEGER NOT NULL,

    PRIMARY KEY (concept, event_id),
    FOREIGN KEY (concept) REFERENCES concepts(concept),
    FOREIGN KEY (event_id) REFERENCES events(event_id)
);

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

CREATE TABLE IF NOT EXISTS concept_cluster_info (
    concept          TEXT    NOT NULL,
    cluster_id       INTEGER NOT NULL,
    cluster_label    TEXT,
    centroid_nx      REAL,
    centroid_ny      REAL,
    centroid_gnx     REAL,
    centroid_gny     REAL,
    centroid_vector  BLOB,
    point_count      INTEGER NOT NULL,
    description      TEXT,

    PRIMARY KEY (concept, cluster_id),
    FOREIGN KEY (concept) REFERENCES concepts(concept)
);

CREATE INDEX IF NOT EXISTS idx_events_concept
    ON events(concept);

CREATE INDEX IF NOT EXISTS idx_events_token
    ON events(token);

CREATE INDEX IF NOT EXISTS idx_events_event_id
    ON events(event_id);

CREATE INDEX IF NOT EXISTS idx_events_doc_id
    ON events(doc_id);

CREATE INDEX IF NOT EXISTS idx_events_concept_year
    ON events(concept, pub_year);

CREATE INDEX IF NOT EXISTS idx_neighbours_event_id
    ON neighbours(event_id);

CREATE INDEX IF NOT EXISTS idx_neighbours_token
    ON neighbours(token);

CREATE INDEX IF NOT EXISTS idx_field_events_concept
    ON concept_field_events(concept);

CREATE INDEX IF NOT EXISTS idx_field_events_event
    ON concept_field_events(event_id);

CREATE INDEX IF NOT EXISTS idx_aggregate_concept
    ON concept_aggregate(concept, kind);
"""

_SCHEMA_CLEAR = """
DROP TABLE IF EXISTS concept_cluster_info;
DROP TABLE IF EXISTS concept_aggregate;
DROP TABLE IF EXISTS concept_field_events;
DROP TABLE IF EXISTS neighbours;
DROP TABLE IF EXISTS events;
DROP TABLE IF EXISTS concepts;
"""


_DELETE_CONCEPT = (
    "DELETE FROM concept_aggregate WHERE concept = ?",
    """
    DELETE FROM neighbours
    WHERE event_id IN (
        SELECT event_id FROM events WHERE concept = ?
    )
    """,
    "DELETE FROM events WHERE concept = ?",
    "DELETE FROM concepts WHERE concept = ?",
)


def _aggregate_rows(
    concept_name: str,
    events: list[dict],
):
    """
    Yield the established concept_aggregate rows.

    Aggregates describe the retrieved neighbourhood, not the lexical seed
    set. A neighbour contributes once for each seed relationship in which it
    occurs, so the same event may legitimately contribute multiple times.

    Windows use the neighbour's local window because that is the window
    represented by the existing Tier 2 aggregate schema.

    Failure mode:
        A neighbour without a local_window_id cannot contribute to
        top_windows, but remains eligible for token and document aggregates.
    """
    token_counts: Counter[str] = Counter()
    doc_counts: Counter[str] = Counter()
    window_counts: Counter[tuple[str, int]] = Counter()

    for event in events:
        for neighbour in event.get("neighbours", []):
            token = neighbour.get("token")
            if token is not None:
                token_counts[str(token)] += 1

            doc_id = neighbour.get("doc_id")
            if doc_id is not None:
                doc_id = str(doc_id)
                doc_counts[doc_id] += 1

            window_id = neighbour.get("local_window_id")
            if doc_id is not None and window_id is not None:
                window_counts[(doc_id, int(window_id))] += 1

    for rank, (token, count) in enumerate(
        token_counts.most_common()
    ):
        yield (
            concept_name,
            "token",
            rank,
            token,
            None,
            None,
            count,
        )

    for rank, (doc_id, count) in enumerate(
        doc_counts.most_common()
    ):
        yield (
            concept_name,
            "doc",
            rank,
            doc_id,
            None,
            None,
            count,
        )

    for rank, ((doc_id, window_id), count) in enumerate(
        window_counts.most_common()
    ):
        yield (
            concept_name,
            "window",
            rank,
            None,
            doc_id,
            window_id,
            count,
        )


def write_tier2_sqlite(
    *,
    db_path: str | Path,
    concept_name: str,
    events: list[dict],
    clear: bool = False,
):
    """
    Write one concept's Tier 2 results to the established SQLite schema.

    Existing rows for this concept are removed before replacement so repeated
    runs remain idempotent without disturbing other concepts.

    Failure mode:
        neighbour membership belongs to the seed event. The same neighbour
        may therefore legitimately appear under many rows in neighbours.

        Aggregates are derived from those same relationships so their meaning
        cannot diverge from the persisted neighbour data.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir( parents=True, exist_ok=True, )

    logger.info( "[tier2] writing sqlite -> %s", db_path, )

    con = sqlite3.connect(db_path)

    try:
        con.execute("PRAGMA foreign_keys = ON")

        if clear:
            logger.info("[tier2] clearing sqlite database")
            con.executescript(_SCHEMA_CLEAR)

        con.executescript(_SCHEMA_INIT)

        con.execute("BEGIN")

        for statement in _DELETE_CONCEPT:
            con.execute(
                statement,
                (concept_name,),
            )

        con.execute(
            """
            INSERT INTO concepts (
                concept,
                n_events
            )
            VALUES (?, ?)
            """,
            (
                concept_name,
                len(events),
            ),
        )

        event_rows = []
        field_event_rows = []
        neighbour_rows = []

        event_rows = []
        field_event_rows = []
        neighbour_rows = []

        for event in events:
            event_id = int(event["event_id"])

            event_rows.append(
                (
                    event_id,
                    concept_name,
                    None,
                    event["token"],
                    event["doc_id"],
                    int(event["pub_year"]),
                    int(event["token_idx"]),
                    event["local_window_id"],
                    event["local_window_token_pos"],
                )
            )

            field_event_rows.append(
                (
                    concept_name,
                    event_id,
                )
            )

            for neighbour in event.get("neighbours", []):
                neighbour_rows.append(
                    (
                        event_id,
                        int(neighbour["event_id"]),
                        None,
                        neighbour["token"],
                        neighbour["doc_id"],
                        int(neighbour["pub_year"]),
                        int(neighbour["token_idx"]),
                        neighbour["local_window_id"],
                        neighbour["local_window_token_pos"],
                        float(neighbour["score"]),
                    )
                )

        con.executemany(
            """
            INSERT INTO events (
                event_id,
                concept,
                vector_id,
                token,
                doc_id,
                pub_year,
                token_idx,
                window_id,
                window_token_pos
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            event_rows,
        )

        con.executemany(
            """
            INSERT INTO concept_field_events (
                concept,
                event_id
            )
            VALUES (?, ?)
            """,
            field_event_rows,
        )

        con.executemany(
            """
            INSERT INTO neighbours (
                event_id,
                neighbour_event_id,
                vector_id,
                token,
                doc_id,
                pub_year,
                token_idx,
                window_id,
                window_token_pos,
                score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            neighbour_rows,
        )

        aggregate_rows = list(
            _aggregate_rows(
                concept_name,
                events,
            )
        )

        if aggregate_rows:
            con.executemany(
                """
                INSERT INTO concept_aggregate (
                    concept,
                    kind,
                    rank,
                    value,
                    window_doc_id,
                    window_id,
                    count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                aggregate_rows,
            )

        con.commit()

    except Exception:
        con.rollback()
        raise

    finally:
        con.close()

    logger.info(
        "[tier2] sqlite write complete: "
        "concept=%s events=%d neighbours=%d",
        concept_name,
        len(events),
        len(neighbour_rows),
    )
