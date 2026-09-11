"""
Tier 2 persistence.

PostgreSQL is authoritative for corpus event identity and provenance.
Lance is authoritative for embedding geometry.

Tier 2 SQLite stores analytical provenance and derived products:

    concepts
        conceptual queries being analysed

    concept_seeds
        deliberate membership of corpus events in a concept's seed population

    retrieval_runs
        configuration and identity of one retrieval experiment

    neighbour_edges
        relationships discovered by a retrieval run

    concept_aggregate
        derived token/document/window summaries

    concept_cluster_info
        derived spatial/cluster summaries

Event metadata is deliberately not duplicated here. It remains available
from PostgreSQL through event_id.

Temporal interval is retrieval provenance rather than corpus metadata. It
identifies which independently replaceable search interval produced a seed
or retrieval run.

Failure mode:
    A Tier 2 row referring to an event that no longer exists in PostgreSQL
    represents broken cross-store provenance and should be detected by
    validation rather than silently repaired here.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from lib.corpus_logging import logger
from lib.corpus_db import analysis_db_connection


_SCHEMA_INIT = """
CREATE TABLE IF NOT EXISTS concepts (
    concept  TEXT PRIMARY KEY,
    n_events INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS concept_seeds (
    concept       TEXT    NOT NULL,
    from_year     INTEGER NOT NULL,
    to_year       INTEGER NOT NULL,
    event_id      INTEGER NOT NULL,
    role          TEXT    NOT NULL,

    PRIMARY KEY (concept, from_year, to_year, event_id),

    FOREIGN KEY (concept)
        REFERENCES concepts(concept)
);

CREATE TABLE IF NOT EXISTS retrieval_runs (
    run_id                INTEGER PRIMARY KEY AUTOINCREMENT,
    concept               TEXT    NOT NULL,
    from_year             INTEGER NOT NULL,
    to_year               INTEGER NOT NULL,
    seed_population       TEXT    NOT NULL,
    neighbour_population  TEXT    NOT NULL,
    scales                TEXT    NOT NULL,
    top_n                 INTEGER NOT NULL,
    rrf_k                 INTEGER NOT NULL,
    oversample            INTEGER NOT NULL,
    model                 TEXT,
    created_at            TEXT    NOT NULL,

    FOREIGN KEY (concept)
        REFERENCES concepts(concept)
);

CREATE TABLE IF NOT EXISTS neighbour_edges (
    run_id             INTEGER NOT NULL,
    seed_event_id      INTEGER NOT NULL,
    neighbour_event_id INTEGER NOT NULL,

    depth              INTEGER NOT NULL,
    via_event_id       INTEGER,

    rank               INTEGER NOT NULL,
    score              REAL,
    score_local        REAL,
    score_medium       REAL,
    score_broad        REAL,

    PRIMARY KEY (
        run_id,
        seed_event_id,
        neighbour_event_id,
        depth,
        via_event_id
    ),

    FOREIGN KEY (run_id)
        REFERENCES retrieval_runs(run_id)
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
    score         REAL NOT NULL,

    FOREIGN KEY (concept)
        REFERENCES concepts(concept)
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

    FOREIGN KEY (concept)
        REFERENCES concepts(concept)
);

CREATE INDEX IF NOT EXISTS idx_concept_seeds_concept
    ON concept_seeds(concept);

CREATE INDEX IF NOT EXISTS idx_concept_seeds_event
    ON concept_seeds(event_id);

CREATE INDEX IF NOT EXISTS idx_concept_seeds_interval
    ON concept_seeds(concept, from_year, to_year);

CREATE INDEX IF NOT EXISTS idx_retrieval_runs_concept
    ON retrieval_runs(concept);

CREATE INDEX IF NOT EXISTS idx_retrieval_runs_interval
    ON retrieval_runs(concept, from_year, to_year);

CREATE INDEX IF NOT EXISTS idx_neighbour_edges_run
    ON neighbour_edges(run_id);

CREATE INDEX IF NOT EXISTS idx_neighbour_edges_seed
    ON neighbour_edges(seed_event_id);

CREATE INDEX IF NOT EXISTS idx_neighbour_edges_neighbour
    ON neighbour_edges(neighbour_event_id);

CREATE INDEX IF NOT EXISTS idx_neighbour_edges_run_seed
    ON neighbour_edges(run_id, seed_event_id);

CREATE INDEX IF NOT EXISTS idx_aggregate_concept
    ON concept_aggregate(concept, kind);
"""

_SCHEMA_CLEAR = (
    "DELETE FROM neighbour_edges",
    "DELETE FROM retrieval_runs",
    "DELETE FROM concept_seeds",
    "DELETE FROM concept_cluster_info",
    "DELETE FROM concept_aggregate",
    "DELETE FROM concepts",
)


def _maybe_float(value):
    return None if value is None else float(value)


def _delete_interval(
    con,
    *,
    concept_name: str,
    from_year: int,
    to_year: int,
) -> None:
    """
    Remove one independently replaceable retrieval interval.

    Whole-concept derived products are deliberately left untouched. They
    are rebuilt after all intervals have been persisted.

    Failure mode:
        Deleting seeds by the current event IDs would leave stale seeds if
        the rerun produces a different seed population. Interval provenance
        on concept_seeds makes replacement independent of that population.
    """
    run_ids = [
        row[0]
        for row in con.execute(
            """
            SELECT run_id
            FROM retrieval_runs
            WHERE concept = ?
              AND from_year = ?
              AND to_year = ?
            """,
            (
                concept_name,
                from_year,
                to_year,
            ),
        ).fetchall()
    ]

    if run_ids:
        placeholders = ",".join("?" for _ in run_ids)

        con.execute(
            f"""
            DELETE FROM neighbour_edges
            WHERE run_id IN ({placeholders})
            """,
            run_ids,
        )

        con.execute(
            f"""
            DELETE FROM retrieval_runs
            WHERE run_id IN ({placeholders})
            """,
            run_ids,
        )

    con.execute(
        """
        DELETE FROM concept_seeds
        WHERE concept = ?
          AND from_year = ?
          AND to_year = ?
        """,
        (
            concept_name,
            from_year,
            to_year,
        ),
    )


def _aggregate_rows(
    concept_name: str,
    events: list[dict],
):
    """
    Yield concept_aggregate rows for the retrieved neighbourhood.

    Token and document rankings are weighted by each neighbour's RRF-fused
    score, not just by how many seed events happened to retrieve it.

    A seed event may retrieve the same token or document more than once;
    retain its strongest contribution so repeated retrieval of the same
    target by one seed does not artificially inflate its aggregate.

    Failure mode:
        A neighbour without a local_window_id cannot contribute to
        top_windows, but remains eligible for token and document aggregates.
    """
    token_seed_weight: dict[str, dict[int, float]] = {}
    doc_seed_weight: dict[str, dict[int, float]] = {}
    window_counts: Counter[tuple[str, int]] = Counter()

    for event in events:
        event_id = int(event["event_id"])

        for neighbour in event.get("neighbours", []):
            weight = float(neighbour.get("score", 0.0))

            token = neighbour.get("token")

            if token is not None:
                token = str(token)
                per_event = token_seed_weight.setdefault(token, {})
                per_event[event_id] = max(
                    per_event.get(event_id, 0.0),
                    weight,
                )

            doc_id = neighbour.get("doc_id")

            if doc_id is not None:
                doc_id = str(doc_id)
                per_event = doc_seed_weight.setdefault(doc_id, {})
                per_event[event_id] = max(
                    per_event.get(event_id, 0.0),
                    weight,
                )

            window_id = neighbour.get("local_window_id")

            if doc_id is not None and window_id is not None:
                window_counts[(doc_id, int(window_id))] += 1

    token_ranked = sorted(
        token_seed_weight.items(),
        key=lambda item: sum(item[1].values()),
        reverse=True,
    )

    for rank, (token, per_event) in enumerate(token_ranked):
        yield (
            concept_name,
            "token",
            rank,
            token,
            None,
            None,
            len(per_event),
        )

    doc_ranked = sorted(
        doc_seed_weight.items(),
        key=lambda item: sum(item[1].values()),
        reverse=True,
    )

    for rank, (doc_id, per_event) in enumerate(doc_ranked):
        yield (
            concept_name,
            "doc",
            rank,
            doc_id,
            None,
            None,
            len(per_event),
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


def _normalise_scales(scales) -> str:
    """
    Persist scale configuration deterministically.

    Retrieval provenance must not depend on the incidental ordering of a
    caller's collection.
    """
    return ",".join(sorted(str(scale) for scale in scales))


def _insert_retrieval_run(
    con,
    *,
    concept_name: str,
    from_year: int,
    to_year: int,
    seed_population: str,
    neighbour_population: str,
    scales,
    top_n: int,
    rrf_k: int,
    oversample: int,
    model: str | None,
) -> int:
    created_at = datetime.now(timezone.utc).isoformat()

    cursor = con.execute(
        """
        INSERT INTO retrieval_runs (
            concept,
            from_year,
            to_year,
            seed_population,
            neighbour_population,
            scales,
            top_n,
            rrf_k,
            oversample,
            model,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            concept_name,
            int(from_year),
            int(to_year),
            seed_population,
            neighbour_population,
            _normalise_scales(scales),
            int(top_n),
            int(rrf_k),
            int(oversample),
            model,
            created_at,
        ),
    )

    return int(cursor.lastrowid)


def _insert_seed_rows(
    con,
    *,
    concept_name: str,
    from_year: int,
    to_year: int,
    events: list[dict],
):
    rows = [
        (
            concept_name,
            int(from_year),
            int(to_year),
            int(event["event_id"]),
            "seed",
        )
        for event in events
    ]

    if not rows:
        return

    con.executemany(
        """
        INSERT INTO concept_seeds (
            concept,
            from_year,
            to_year,
            event_id,
            role
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )


def _insert_neighbour_edges(
    con,
    *,
    run_id: int,
    events: list[dict],
):
    """
    Persist first-order retrieval relationships.

    Event metadata is intentionally absent. event_id is the stable bridge
    back to PostgreSQL, while this table records what the retrieval operation
    discovered and the evidence supporting that relationship.
    """
    rows = []

    for event in events:
        seed_event_id = int(event["event_id"])

        for neighbour in event.get("neighbours", []):
            rows.append(
                (
                    int(run_id),
                    seed_event_id,
                    int(neighbour["event_id"]),
                    1,
                    None,
                    int(neighbour["rank"]),
                    _maybe_float(neighbour.get("score")),
                    _maybe_float(neighbour.get("score_local")),
                    _maybe_float(neighbour.get("score_medium")),
                    _maybe_float(neighbour.get("score_broad")),
                )
            )

    if not rows:
        return 0

    con.executemany(
        """
        INSERT INTO neighbour_edges (
            run_id,
            seed_event_id,
            neighbour_event_id,
            depth,
            via_event_id,
            rank,
            score,
            score_local,
            score_medium,
            score_broad
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )

    return len(rows)


def write_tier2_sqlite(
    *,
    db_path: str | Path,
    concept_name: str,
    events: list[dict],
    from_year: int,
    to_year: int,
    clear: bool = False,
    seed_population: str = "lexical_forms",
    neighbour_population: str = "temporal_year",
    scales=("local", "medium", "broad"),
    top_n: int = 60,
    rrf_k: int = 60,
    oversample: int = 5,
    model: str | None = None,
):
    """
    Persist one Tier 2 retrieval interval.

    The concept seed membership and retrieval relationships are stored
    separately. A temporal interval is independently replaceable, while
    concept-wide derived products remain untouched until they are rebuilt.

    Failure mode:
        concept_aggregate must not be written from the current interval
        because its rankings are whole-concept products. Writing them here
        would create competing rank=0 rows for different intervals.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "[tier2] writing sqlite -> %s",
        db_path,
    )

    con = analysis_db_connection(db_path)

    run_id = None
    neighbour_count = 0

    try:
        con.executescript(_SCHEMA_INIT)

        con.execute("BEGIN")
        con.execute("PRAGMA foreign_keys = ON")

        if clear:
            logger.info("[tier2] clearing sqlite database")

            for statement in _SCHEMA_CLEAR:
                con.execute(statement)

        else:
            logger.info(
                "[tier2] replacing interval %s-%s for concept=%s",
                from_year,
                to_year,
                concept_name,
            )

            _delete_interval(
                con,
                concept_name=concept_name,
                from_year=from_year,
                to_year=to_year,
            )

        con.execute(
            """
            INSERT INTO concepts (
                concept,
                n_events
            )
            VALUES (?, 0)
            ON CONFLICT(concept) DO NOTHING
            """,
            (concept_name,),
        )

        _insert_seed_rows(
            con,
            concept_name=concept_name,
            from_year=from_year,
            to_year=to_year,
            events=events,
        )

        con.execute(
            """
            UPDATE concepts
            SET n_events = (
                SELECT COUNT(*)
                FROM concept_seeds
                WHERE concept = ?
            )
            WHERE concept = ?
            """,
            (
                concept_name,
                concept_name,
            ),
        )

        run_id = _insert_retrieval_run(
            con,
            concept_name=concept_name,
            from_year=from_year,
            to_year=to_year,
            seed_population=seed_population,
            neighbour_population=neighbour_population,
            scales=scales,
            top_n=top_n,
            rrf_k=rrf_k,
            oversample=oversample,
            model=model,
        )

        neighbour_count = _insert_neighbour_edges(
            con,
            run_id=run_id,
            events=events,
        )

        con.commit()

    except Exception:
        con.rollback()
        raise

    finally:
        con.close()

    logger.info(
        "[tier2] sqlite write complete: "
        "concept=%s interval=%s-%s run=%d seeds=%d neighbours=%d",
        concept_name,
        from_year,
        to_year,
        run_id,
        len(events),
        neighbour_count,
    )
