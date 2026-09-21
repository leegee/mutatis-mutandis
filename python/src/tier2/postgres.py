# tier2/postgres.py

from __future__ import annotations

from datetime import datetime, timezone

from lib.corpus_logging import logger
from tier2.analysis import TIER2_SCALES

def _maybe_float(value):
    return None if value is None else float(value)


def _normalise_scales(scales) -> str:
    """
    Persist scale configuration deterministically.

    Retrieval provenance must not depend on incidental ordering in the
    caller's collection.
    """
    return ",".join(sorted(str(scale) for scale in scales))


def _delete_interval(
    con,
    *,
    concept_name: str,
    from_year: int,
    to_year: int,
) -> None:
    """
    Remove one independently replaceable retrieval interval.

    Seeds and retrieval edges belong to the interval. event_field is rebuilt
    afterwards from the remaining complete concept population.
    """
    with con.cursor() as cur:
        cur.execute(
            """
            SELECT run_id
            FROM tier2.retrieval_runs
            WHERE concept = %s
              AND from_year = %s
              AND to_year = %s
            """,
            (concept_name, from_year, to_year),
        )
        run_ids = [int(row[0]) for row in cur.fetchall()]

        if run_ids:
            cur.execute(
                """
                DELETE FROM tier2.neighbour_edges
                WHERE run_id = ANY(%s)
                """,
                (run_ids,),
            )

            cur.execute(
                """
                DELETE FROM tier2.retrieval_runs
                WHERE run_id = ANY(%s)
                """,
                (run_ids,),
            )

        cur.execute(
            """
            DELETE FROM tier2.concept_seeds
            WHERE concept = %s
              AND from_year = %s
              AND to_year = %s
            """,
            (concept_name, from_year, to_year),
        )


def _insert_seed_rows(
    con,
    *,
    concept_name: str,
    from_year: int,
    to_year: int,
    events: list[dict],
) -> None:
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

    with con.cursor() as cur:
        with cur.copy(
            """
            COPY tier2.concept_seeds (
                concept,
                from_year,
                to_year,
                event_id,
                role
            )
            FROM STDIN
            """
        ) as copy:
            for row in rows:
                copy.write_row(row)


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
    created_at = datetime.now(timezone.utc)

    with con.cursor() as cur:
        cur.execute(
            """
            INSERT INTO tier2.retrieval_runs (
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
            VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s
            )
            RETURNING run_id
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

        row = cur.fetchone()

    if row is None:
        raise RuntimeError("PostgreSQL did not return retrieval run ID")

    return int(row[0])


def _insert_neighbour_edges(
    con,
    *,
    run_id: int,
    events: list[dict],
) -> int:
    """
    Persist first-order retrieval relationships.

    Event metadata remains in PostgreSQL events. This table contains only
    retrieval provenance and scores.
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

    with con.cursor() as cur:
        with cur.copy(
            """
            COPY tier2.neighbour_edges (
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
            FROM STDIN
            """
        ) as copy:
            for row in rows:
                copy.write_row(row)

    return len(rows)


def _rebuild_event_field(
    con,
    *,
    concept_name: str,
) -> None:
    """
    Rebuild the complete event population for one concept.

    The field is derived from all currently persisted seeds and retrieval
    relationships, so replacing one interval cannot leave stale membership.
    """
    with con.cursor() as cur:
        cur.execute(
            """
            DELETE FROM tier2.event_field
            WHERE concept = %s
            """,
            (concept_name,),
        )

        cur.execute(
            """
            INSERT INTO tier2.event_field (
                concept,
                event_id,
                role
            )
            SELECT
                %s,
                s.event_id,
                CASE
                    WHEN EXISTS (
                        SELECT 1
                        FROM tier2.neighbour_edges ne
                        JOIN tier2.retrieval_runs rr
                          ON rr.run_id = ne.run_id
                        WHERE rr.concept = %s
                          AND ne.neighbour_event_id = s.event_id
                    )
                    THEN 'both'
                    ELSE 'seed'
                END
            FROM tier2.concept_seeds s
            WHERE s.concept = %s

            UNION

            SELECT
                %s,
                ne.neighbour_event_id,
                'neighbour'
            FROM tier2.neighbour_edges ne
            JOIN tier2.retrieval_runs rr
              ON rr.run_id = ne.run_id
            WHERE rr.concept = %s
              AND NOT EXISTS (
                  SELECT 1
                  FROM tier2.concept_seeds s
                  WHERE s.concept = %s
                    AND s.event_id = ne.neighbour_event_id
              )
            """,
            (
                concept_name,
                concept_name,
                concept_name,
                concept_name,
                concept_name,
                concept_name,
            ),
        )


def write_tier2_postgres(
    *,
    connection,
    concept_name: str,
    events: list[dict],
    from_year: int,
    to_year: int,
    clear: bool = False,
    seed_population: str = "lexical_forms",
    neighbour_population: str = "same_publication_year",
    scales=TIER2_SCALES,
    top_n: int = 60,
    rrf_k: int = 60,
    oversample: int = 5,
    model: str | None = None,
) -> int:
    """
    Persist one Tier 2 retrieval interval to PostgreSQL.

    PostgreSQL owns analytical persistence. Lance remains authoritative for
    vector geometry, while event IDs bridge retrieval results to events.
    """
    logger.info(
        "[tier2] writing PostgreSQL: concept=%s interval=%s-%s",
        concept_name,
        from_year,
        to_year,
    )

    run_id = None
    neighbour_count = 0

    try:
        with connection.transaction():
            if clear:
                logger.info("[tier2] clearing PostgreSQL Tier 2 data")

                with connection.cursor() as cur:
                    cur.execute("DELETE FROM tier2.neighbour_edges")
                    cur.execute("DELETE FROM tier2.retrieval_runs")
                    cur.execute("DELETE FROM tier2.event_field")
                    cur.execute("DELETE FROM tier2.concept_seeds")
                    cur.execute("DELETE FROM tier2.concept_aggregate")
                    cur.execute("DELETE FROM tier2.concepts")

            else:
                logger.info(
                    "[tier2] replacing interval %s-%s for concept=%s",
                    from_year,
                    to_year,
                    concept_name,
                )

                _delete_interval(
                    connection,
                    concept_name=concept_name,
                    from_year=from_year,
                    to_year=to_year,
                )

            with connection.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO tier2.concepts (
                        concept,
                        n_events
                    )
                    VALUES (%s, 0)
                    ON CONFLICT (concept) DO NOTHING
                    """,
                    (concept_name,),
                )

            _insert_seed_rows(
                connection,
                concept_name=concept_name,
                from_year=from_year,
                to_year=to_year,
                events=events,
            )

            with connection.cursor() as cur:
                cur.execute(
                    """
                    UPDATE tier2.concepts
                    SET n_events = (
                        SELECT COUNT(*)
                        FROM tier2.concept_seeds
                        WHERE concept = %s
                    )
                    WHERE concept = %s
                    """,
                    (concept_name, concept_name),
                )

            run_id = _insert_retrieval_run(
                connection,
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
                connection,
                run_id=run_id,
                events=events,
            )

            _rebuild_event_field(
                connection,
                concept_name=concept_name,
            )

    except Exception:
        logger.exception(
            "[tier2] PostgreSQL write failed: concept=%s interval=%s-%s",
            concept_name,
            from_year,
            to_year,
        )
        raise

    logger.info(
        "[tier2] PostgreSQL write complete: "
        "concept=%s interval=%s-%s run=%d seeds=%d neighbours=%d",
        concept_name,
        from_year,
        to_year,
        run_id,
        len(events),
        neighbour_count,
    )

    return int(run_id)
