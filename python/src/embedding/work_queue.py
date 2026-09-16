# embedding/work_queue.py

from __future__ import annotations

from dataclasses import dataclass

from lib.corpus_db import get_connection


DEFAULT_LEASE_SECONDS = 900


@dataclass(frozen=True)
class EmbeddingModel:
    model_id: int
    model_key: str
    model_revision: str
    embedding_dimension: int


@dataclass(frozen=True)
class EmbeddingWork:
    work_id: int
    model_id: int
    event_ids: tuple[int, ...]
    status: str
    attempt_count: int
    observation_count: int
    completed_count: int


def register_model(
    *,
    model_key: str,
    model_revision: str,
    embedding_dimension: int,
) -> EmbeddingModel:
    if not model_key:
        raise ValueError("model_key must not be empty")

    if not model_revision:
        raise ValueError("model_revision must not be empty")

    if embedding_dimension <= 0:
        raise ValueError("embedding_dimension must be positive")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO embedding.models (
                    model_key,
                    model_revision,
                    embedding_dimension
                )
                VALUES (%s, %s, %s)
                ON CONFLICT (model_key, model_revision)
                DO NOTHING
                RETURNING
                    model_id,
                    model_key,
                    model_revision,
                    embedding_dimension
                """,
                (
                    model_key,
                    model_revision,
                    embedding_dimension,
                ),
            )

            row = cur.fetchone()

            if row is None:
                cur.execute(
                    """
                    SELECT
                        model_id,
                        model_key,
                        model_revision,
                        embedding_dimension
                    FROM embedding.models
                    WHERE
                        model_key = %s
                        AND model_revision = %s
                    """,
                    (
                        model_key,
                        model_revision,
                    ),
                )

                row = cur.fetchone()

                if row is None:
                    raise RuntimeError(
                        "model disappeared after conflict"
                    )

                existing_dimension = row[3]

                if existing_dimension != embedding_dimension:
                    raise ValueError(
                        f"embedding dimension mismatch for "
                        f"{model_key!r} revision {model_revision!r}: "
                        f"database has {existing_dimension}, "
                        f"requested {embedding_dimension}"
                    )

        conn.commit()

    return EmbeddingModel(
        model_id=row[0],
        model_key=row[1],
        model_revision=row[2],
        embedding_dimension=row[3],
    )


def reset_model_work(
    *,
    model_id: int,
    clear_inventory: bool = False,
) -> None:
    """Reset embedding work for a model.

    Work rows are disposable scheduling state. Inventory is durable evidence
    that a vector exists, so it is only removed when explicitly requested.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            if clear_inventory:
                cur.execute(
                    """
                    DELETE FROM embedding.inventory
                    WHERE model_id = %s
                    """,
                    (model_id,),
                )

            cur.execute(
                """
                DELETE FROM embedding.work
                WHERE model_id = %s
                """,
                (model_id,),
            )

        conn.commit()


def create_work(
    *,
    model_id: int,
    batch_size: int,
    limit: int | None = None,
) -> int:
    """
    Create pending batches for events without an inventory record.

    The event IDs are stored explicitly rather than represented as a numeric
    range because event IDs are identifiers, not a downstream ordering
    invariant.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM embedding.work
                WHERE
                    model_id = %s
                    AND status IN ('pending', 'claimed', 'embedding')
                LIMIT 1
                """,
                (model_id,),
            )

            if cur.fetchone() is not None:
                raise RuntimeError(
                    f"unfinished embedding work already exists "
                    f"for model {model_id}"
                )

            limit_sql = ""
            params: list[object] = [model_id]

            if limit is not None:
                limit_sql = "LIMIT %s"
                params.append(limit)

            params.extend((batch_size, model_id))

            cur.execute(
                f"""
                WITH missing AS (
                    SELECT e.event_id
                    FROM events AS e
                    LEFT JOIN embedding.inventory AS i
                        ON i.event_id = e.event_id
                       AND i.model_id = %s
                    WHERE i.event_id IS NULL
                    ORDER BY e.event_id
                    {limit_sql}
                ),
                numbered AS (
                    SELECT
                        event_id,
                        (ROW_NUMBER() OVER (ORDER BY event_id) - 1)
                            / %s AS batch_no
                    FROM missing
                ),
                batches AS (
                    SELECT
                        batch_no,
                        array_agg(event_id ORDER BY event_id) AS event_ids
                    FROM numbered
                    GROUP BY batch_no
                )
                INSERT INTO embedding.work (
                    model_id,
                    event_ids,
                    observation_count
                )
                SELECT
                    %s,
                    event_ids,
                    cardinality(event_ids)
                FROM batches
                RETURNING work_id
                """,
                params,
            )

            work_ids = cur.fetchall()

        conn.commit()

    return len(work_ids)


def claim_next_work(
    *,
    model_id: int,
    worker_id: str,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
) -> EmbeddingWork | None:
    if not worker_id:
        raise ValueError("worker_id must not be empty")

    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH candidate AS (
                    SELECT work_id
                    FROM embedding.work
                    WHERE
                        model_id = %s
                        AND (
                            status = 'pending'
                            OR (
                                status IN ('claimed', 'embedding')
                                AND lease_expires_at < now()
                            )
                        )
                    ORDER BY work_id
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                UPDATE embedding.work AS w
                SET
                    status = 'claimed',
                    worker_id = %s,
                    attempt_count = w.attempt_count + 1,
                    claimed_at = now(),
                    heartbeat_at = now(),
                    lease_expires_at =
                        now() + (%s * INTERVAL '1 second'),
                    started_at =
                        COALESCE(w.started_at, now()),
                    last_error = NULL
                FROM candidate
                WHERE w.work_id = candidate.work_id
                RETURNING
                    w.work_id,
                    w.model_id,
                    w.event_ids,
                    w.status,
                    w.attempt_count,
                    w.observation_count,
                    w.completed_count
                """,
                (
                    model_id,
                    worker_id,
                    lease_seconds,
                ),
            )

            row = cur.fetchone()

        conn.commit()

    if row is None:
        return None

    return EmbeddingWork(
        work_id=row[0],
        model_id=row[1],
        event_ids=tuple(row[2]),
        status=row[3],
        attempt_count=row[4],
        observation_count=row[5],
        completed_count=row[6],
    )


def begin_embedding(
    *,
    work_id: int,
    worker_id: str,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
) -> None:
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE embedding.work
                SET
                    status = 'embedding',
                    heartbeat_at = now(),
                    lease_expires_at =
                        now() + (%s * INTERVAL '1 second')
                WHERE
                    work_id = %s
                    AND worker_id = %s
                    AND status = 'claimed'
                RETURNING work_id
                """,
                (
                    lease_seconds,
                    work_id,
                    worker_id,
                ),
            )

            if cur.fetchone() is None:
                raise RuntimeError(
                    f"work {work_id} is no longer owned by {worker_id}"
                )

        conn.commit()


def heartbeat(
    *,
    work_id: int,
    worker_id: str,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
) -> None:
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE embedding.work
                SET
                    heartbeat_at = now(),
                    lease_expires_at =
                        now() + (%s * INTERVAL '1 second')
                WHERE
                    work_id = %s
                    AND worker_id = %s
                    AND status IN ('claimed', 'embedding')
                RETURNING work_id
                """,
                (
                    lease_seconds,
                    work_id,
                    worker_id,
                ),
            )

            if cur.fetchone() is None:
                raise RuntimeError(
                    f"work {work_id} is no longer owned by {worker_id}"
                )

        conn.commit()


def record_inventory(
    *,
    work_id: int,
    worker_id: str,
    event_id: int,
    embedding_key: str,
) -> bool:
    """
    Record a successfully persisted vector belonging to an owned work item.

    The work row is authoritative for model identity and event membership.
    Inventory insertion remains idempotent through the unique
    (event_id, model_id) constraint.
    """
    if not worker_id:
        raise ValueError("worker_id must not be empty")

    if not embedding_key:
        raise ValueError("embedding_key must not be empty")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO embedding.inventory (
                    event_id,
                    model_id,
                    embedding_key
                )
                SELECT
                    %s,
                    w.model_id,
                    %s
                FROM embedding.work AS w
                WHERE
                    w.work_id = %s
                    AND w.worker_id = %s
                    AND w.status = 'embedding'
                    AND %s = ANY(w.event_ids)
                ON CONFLICT (event_id, model_id)
                DO NOTHING
                RETURNING event_id
                """,
                (
                    event_id,
                    embedding_key,
                    work_id,
                    worker_id,
                    event_id,
                ),
            )

            row = cur.fetchone()

            if row is not None:
                inserted = True
            else:
                # Distinguish an idempotent retry from an ownership error.
                cur.execute(
                    """
                    SELECT
                        w.model_id,
                        w.event_ids,
                        w.worker_id,
                        w.status
                    FROM embedding.work AS w
                    WHERE w.work_id = %s
                    """,
                    (work_id,),
                )

                work = cur.fetchone()

                if work is None:
                    raise RuntimeError(
                        f"work {work_id} does not exist"
                    )

                model_id, event_ids, owner, status = work

                if owner != worker_id:
                    raise RuntimeError(
                        f"work {work_id} is owned by "
                        f"{owner!r}, not {worker_id!r}"
                    )

                if status != "embedding":
                    raise RuntimeError(
                        f"work {work_id} is not embedding "
                        f"(status={status!r})"
                    )

                if event_id not in event_ids:
                    raise RuntimeError(
                        f"event {event_id} does not belong to "
                        f"work {work_id}"
                    )

                # The vector is already inventoried. This is the expected
                # result when a batch is being retried after Lance succeeded.
                inserted = False

        conn.commit()

    return inserted


def complete_work(
    *,
    work_id: int,
    worker_id: str,
) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE embedding.work AS w
                SET
                    completed_count = (
                        SELECT COUNT(*)
                        FROM embedding.inventory AS i
                        WHERE
                            i.model_id = w.model_id
                            AND i.event_id = ANY(w.event_ids)
                    )
                WHERE
                    w.work_id = %s
                    AND w.worker_id = %s
                    AND w.status = 'embedding'
                RETURNING
                    w.observation_count,
                    w.completed_count
                """,
                (
                    work_id,
                    worker_id,
                ),
            )

            row = cur.fetchone()

            if row is None:
                raise RuntimeError(
                    f"work {work_id} is no longer owned by {worker_id}"
                )

            expected, completed = row

            if completed != expected:
                conn.rollback()
                raise RuntimeError(
                    f"work {work_id} incomplete: "
                    f"{completed}/{expected} embeddings"
                )

            cur.execute(
                """
                UPDATE embedding.work
                SET
                    status = 'completed',
                    completed_at = now(),
                    lease_expires_at = NULL
                WHERE
                    work_id = %s
                    AND worker_id = %s
                    AND status = 'embedding'
                RETURNING work_id
                """,
                (
                    work_id,
                    worker_id,
                ),
            )

            if cur.fetchone() is None:
                raise RuntimeError(
                    f"work {work_id} could not be completed"
                )

        conn.commit()


def fail_work(
    *,
    work_id: int,
    worker_id: str,
    error: str,
) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE embedding.work
                SET
                    status = 'failed',
                    last_error = %s,
                    lease_expires_at = NULL
                WHERE
                    work_id = %s
                    AND worker_id = %s
                    AND status IN ('claimed', 'embedding')
                RETURNING work_id
                """,
                (
                    error,
                    work_id,
                    worker_id,
                ),
            )

            if cur.fetchone() is None:
                raise RuntimeError(
                    f"work {work_id} is no longer owned by {worker_id}"
                )

        conn.commit()


def retry_failed_work(
    *,
    work_id: int,
) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE embedding.work
                SET
                    status = 'pending',
                    worker_id = NULL,
                    claimed_at = NULL,
                    heartbeat_at = NULL,
                    lease_expires_at = NULL,
                    last_error = NULL
                WHERE
                    work_id = %s
                    AND status = 'failed'
                RETURNING work_id
                """,
                (work_id,),
            )

            if cur.fetchone() is None:
                raise RuntimeError(
                    f"work {work_id} is not failed"
                )

        conn.commit()


def work_summary(
    *,
    model_id: int,
) -> list[tuple]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    status,
                    COUNT(*) AS work_items,
                    SUM(observation_count) AS observations,
                    SUM(completed_count) AS completed
                FROM embedding.work
                WHERE model_id = %s
                GROUP BY status
                ORDER BY status
                """,
                (model_id,),
            )

            return cur.fetchall()
