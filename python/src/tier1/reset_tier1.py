# reset_tier1

"""
Reset Tier 1 seed-ingestor state.

Deletes:
    - PostgreSQL Tier 1 events table (including its event ID sequence)
    - Tier 1 LanceDB tables/files

Preserves:
    - corpus documents/tokens
    - Tier 2
    - Tier 3
    - embedding_jobs

Run before re-running tier1_seeds2events.py after changing seed forms.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from lib.corpus_db import get_connection
from lib.corpus_logging import logger
from lib.corpus_config import LANCE_INDEXES_DIR
from tier1.db_observation_backend import drop_events_table


def reset_postgres() -> None:
    """Drop the Tier 1 events table and its associated sequence."""
    conn = get_connection()

    try:
        logger.info("Dropping Tier 1 events table...")
        drop_events_table(conn)
        conn.commit()
        logger.info("Tier 1 PostgreSQL state reset.")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def reset_lance() -> None:
    """Remove all Tier 1 LanceDB tables."""
    lance_root = Path(LANCE_INDEXES_DIR)

    if not lance_root.exists():
        logger.info("LanceDB directory does not exist: %s", lance_root)
        return

    if not lance_root.is_dir():
        raise RuntimeError(
            f"Expected LanceDB directory, but found non-directory: {lance_root}"
        )

    logger.info("Removing Tier 1 LanceDB directory: %s", lance_root)
    shutil.rmtree(lance_root)

    lance_root.mkdir(parents=True, exist_ok=True)

    logger.info("Tier 1 LanceDB state reset.")


def main() -> None:
    logger.info("=== Tier 1 RESET ===")

    reset_postgres()
    reset_lance()

    logger.info("=== Tier 1 RESET COMPLETE ===")
    logger.info("Corpus documents/tokens were not modified.")
    logger.info("Tier 2 and Tier 3 were not modified.")
    logger.info("embedding_jobs was not modified.")


if __name__ == "__main__":
    main()
