from typing import Sequence
from psycopg import Connection
from lib.corpus_logging import logger


EVENT_COLUMNS = (
    "event_id",
    "corpus",
    "doc_id",
    "token",
    "token_idx",
    "pub_year",
    "scale",
    "window_id",
    "window_token_pos",
)


def create_events_table(conn: Connection) -> None:
    """
    Create the durable observation identity and metadata table.

    event_id is the stable identity shared by PostgreSQL and Lance.
    Vector data is deliberately not stored here.

    An event identifies one corpus token occurrence in one particular
    contextual representation.
    """
    logger.info("[corpus_db] Creating events table")

    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("""
                CREATE SEQUENCE IF NOT EXISTS event_id_seq;

                CREATE TABLE IF NOT EXISTS events (
                    event_id BIGINT PRIMARY KEY
                        DEFAULT nextval('event_id_seq'),

                    corpus TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    token TEXT NOT NULL,
                    token_idx INTEGER NOT NULL,
                    pub_year INTEGER,

                    scale TEXT NOT NULL,
                    window_id BIGINT NOT NULL,
                    window_token_pos INTEGER NOT NULL,

                    CONSTRAINT events_token_fk
                        FOREIGN KEY (doc_id, token_idx)
                        REFERENCES tokens(doc_id, token_idx)
                        ON DELETE CASCADE
                );
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_corpus_doc_token
                ON events(corpus, doc_id, token_idx);
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_doc_token
                ON events(doc_id, token_idx);
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_pub_year
                ON events(pub_year);
            """)

            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS
                    uq_events_observation
                ON events(
                    corpus,
                    doc_id,
                    token_idx,
                    scale,
                    window_id,
                    window_token_pos
                );
            """)

    logger.info("[corpus_db] Events table created")


def migrate_events_table(conn: Connection) -> None:
    """
    Migrate the original scale-specific event provenance columns to the
    normalized scale/window representation.

    Existing observations are preserved with their existing event IDs.

    If multiple events represent the same normalized observation provenance,
    retain the event referenced by embedding.inventory when possible;
    otherwise retain the lowest event_id. Remove the redundant event rows
    before creating the unique provenance index.
    """
    logger.info("[corpus_db] Migrating events table schema")

    with conn.cursor() as cur:
        # Already migrated.
        cur.execute("""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                    AND table_name = 'events'
                    AND column_name = 'scale'
            )
        """)

        if cur.fetchone()[0]:
            logger.info(
                "[corpus_db] Events table already uses "
                "normalized provenance"
            )
            return

        # Check that the expected old schema exists.
        cur.execute("""
            SELECT COUNT(*)
            FROM information_schema.columns
            WHERE table_schema = current_schema()
                AND table_name = 'events'
                AND column_name IN (
                    'local_window_id',
                    'local_window_token_pos',
                    'medium_window_id',
                    'medium_window_token_pos',
                    'broad_window_id',
                    'broad_window_token_pos'
                )
        """)

        if cur.fetchone()[0] != 6:
            raise RuntimeError(
                "events table is neither the old scale-specific "
                "schema nor the new normalized schema"
            )

        # Every existing event must represent exactly one scale.
        cur.execute("""
            SELECT COUNT(*)
            FROM events
            WHERE
                (
                    local_window_id IS NOT NULL
                    OR local_window_token_pos IS NOT NULL
                )::int
                + (
                    medium_window_id IS NOT NULL
                    OR medium_window_token_pos IS NOT NULL
                )::int
                + (
                    broad_window_id IS NOT NULL
                    OR broad_window_token_pos IS NOT NULL
                )::int
                <> 1
        """)

        ambiguous = int(cur.fetchone()[0])

        if ambiguous:
            raise RuntimeError(
                "Cannot migrate events table: "
                f"{ambiguous} events have zero or multiple "
                "scale provenances"
            )

        # Each populated scale must have both fields.
        cur.execute("""
            SELECT COUNT(*)
            FROM events
            WHERE
                (
                    local_window_id IS NULL
                    AND local_window_token_pos IS NOT NULL
                )
                OR (
                    local_window_id IS NOT NULL
                    AND local_window_token_pos IS NULL
                )
                OR (
                    medium_window_id IS NULL
                    AND medium_window_token_pos IS NOT NULL
                )
                OR (
                    medium_window_id IS NOT NULL
                    AND medium_window_token_pos IS NULL
                )
                OR (
                    broad_window_id IS NULL
                    AND broad_window_token_pos IS NOT NULL
                )
                OR (
                    broad_window_id IS NOT NULL
                    AND broad_window_token_pos IS NULL
                )
        """)

        half_populated = int(cur.fetchone()[0])

        if half_populated:
            raise RuntimeError(
                "Cannot migrate events table: "
                f"{half_populated} events have half-populated "
                "window provenance"
            )

        logger.info("Add the normalized provenance columns.")

        # Add the normalized provenance columns.
        cur.execute("""
            ALTER TABLE events
                ADD COLUMN scale TEXT,
                ADD COLUMN window_id BIGINT,
                ADD COLUMN window_token_pos INTEGER;
        """)

        logger.info("Convert the old scale-specific representation.")

        # Convert the old scale-specific representation.
        cur.execute("""
            UPDATE events
            SET
                scale = CASE
                    WHEN local_window_id IS NOT NULL
                        THEN 'local'
                    WHEN medium_window_id IS NOT NULL
                        THEN 'medium'
                    WHEN broad_window_id IS NOT NULL
                        THEN 'broad'
                END,
                window_id = COALESCE(
                    local_window_id,
                    medium_window_id,
                    broad_window_id
                ),
                window_token_pos = COALESCE(
                    local_window_token_pos,
                    medium_window_token_pos,
                    broad_window_token_pos
                );
        """)

        # Verify that conversion produced complete rows.
        cur.execute("""
            SELECT COUNT(*)
            FROM events
            WHERE scale IS NULL
                OR window_id IS NULL
                OR window_token_pos IS NULL
        """)

        incomplete = int(cur.fetchone()[0])

        if incomplete:
            raise RuntimeError(
                "Events migration produced "
                f"{incomplete} incomplete events"
            )

        # The normalized provenance is now complete enough to identify
        # duplicate observations.
        #
        # Prefer an event referenced by embedding.inventory. If neither
        # duplicate is referenced, retain the lowest event_id.
        logger.info(
            "[corpus_db] Reconciling duplicate event provenance"
        )

        cur.execute("""
            WITH ranked AS (
                SELECT
                    e.event_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY
                            e.corpus,
                            e.doc_id,
                            e.token_idx,
                            e.scale,
                            e.window_id,
                            e.window_token_pos
                        ORDER BY
                            CASE
                                WHEN EXISTS (
                                    SELECT 1
                                    FROM embedding.inventory i
                                    WHERE i.event_id = e.event_id
                                )
                                THEN 0
                                ELSE 1
                            END,
                            e.event_id
                    ) AS rn
                FROM events e
            )
            DELETE FROM events e
            USING ranked r
            WHERE e.event_id = r.event_id
                AND r.rn > 1
        """)

        duplicates_removed = cur.rowcount

        logger.info(
            "[corpus_db] Removed %d duplicate event rows",
            duplicates_removed,
        )

        # Now the normalized provenance can safely become unique.
        cur.execute("""
            ALTER TABLE events
                ALTER COLUMN scale SET NOT NULL,
                ALTER COLUMN window_id SET NOT NULL,
                ALTER COLUMN window_token_pos SET NOT NULL;
        """)

        # Remove the obsolete scale-specific representation.
        cur.execute("""
            ALTER TABLE events
                DROP COLUMN local_window_id,
                DROP COLUMN local_window_token_pos,
                DROP COLUMN medium_window_id,
                DROP COLUMN medium_window_token_pos,
                DROP COLUMN broad_window_id,
                DROP COLUMN broad_window_token_pos;
        """)

        # Enforce the normalized event identity.
        cur.execute("""
            CREATE UNIQUE INDEX uq_events_observation
            ON events(
                corpus,
                doc_id,
                token_idx,
                scale,
                window_id,
                window_token_pos
            );
        """)

        conn.commit()

    logger.info(
        "[corpus_db] Events table migration complete"
    )


def drop_events_table(conn: Connection) -> None:
    """
    Drop the event table and its ID sequence.
    """
    logger.info("[corpus_db] Dropping events table")

    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS events CASCADE;")
            cur.execute("DROP SEQUENCE IF EXISTS event_id_seq;")

    logger.info("[corpus_db] Events table dropped")


def sync_event_id_sequence(conn: Connection) -> None:
    """
    Move the event ID sequence beyond the highest imported event ID.
    """
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(event_id) FROM events")
            max_id = cur.fetchone()[0]

            if max_id is None:
                cur.execute(
                    "ALTER SEQUENCE event_id_seq RESTART WITH 1"
                )
            else:
                cur.execute(
                    "SELECT setval('event_id_seq', %s, true)",
                    (int(max_id),),
                )


def allocate_event_ids(
    conn: Connection,
    count: int,
) -> list[int]:
    if count < 0:
        raise ValueError("count must be non-negative")

    if count == 0:
        return []

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT nextval('event_id_seq')
            FROM generate_series(1, %s);
            """,
            (count,),
        )
        return [int(row[0]) for row in cur.fetchall()]


def insert_events(
    conn: Connection,
    *,
    event_id: Sequence[int],
    corpus: Sequence[str],
    doc_id: Sequence[str],
    token: Sequence[str],
    token_idx: Sequence[int],
    pub_year: Sequence[int | None],
    scale: Sequence[str],
    window_id: Sequence[int],
    window_token_pos: Sequence[int],
) -> None:
    """
    Insert a batch of authoritative event identities.

    Duplicate observation provenance is rejected by the database-level
    unique index. Callers that need idempotent creation should resolve
    existing observations before allocating new event IDs.
    """
    n = len(event_id)

    columns = {
        "event_id": event_id,
        "corpus": corpus,
        "doc_id": doc_id,
        "token": token,
        "token_idx": token_idx,
        "pub_year": pub_year,
        "scale": scale,
        "window_id": window_id,
        "window_token_pos": window_token_pos,
    }

    for name, values in columns.items():
        if len(values) != n:
            raise ValueError(
                f"{name} length {len(values)} != event_id length {n}"
            )

    with conn.cursor() as cur:
        with cur.copy("""
            COPY events (
                event_id,
                corpus,
                doc_id,
                token,
                token_idx,
                pub_year,
                scale,
                window_id,
                window_token_pos
            )
            FROM STDIN
        """) as copy:
            for i in range(n):
                copy.write_row((
                    int(event_id[i]),
                    corpus[i],
                    doc_id[i],
                    token[i],
                    int(token_idx[i]),
                    pub_year[i],
                    scale[i],
                    int(window_id[i]),
                    int(window_token_pos[i]),
                ))
