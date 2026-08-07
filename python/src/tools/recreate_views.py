#!/usr/bin/env python3
"""
recreate_views.py

Drop + recreate the materialized views (so new eebo_config values
are baked into the view definitions), then rebuild the associated
indexes.  Underlying documents / tokens tables are never touched.
"""

from lib.eebo_db import (
    get_connection,
    create_views,
    create_tiered_token_indexes,
    create_concurrent_indexes,
)
from lib.corpus_logging import logger


def main() -> None:
    logger.info("Rebuilding materialized views with current eebo_config values")

    with get_connection(application_name="eebo-rebuild-views") as conn:
        # Drops existing MVs (CASCADE) and recreates them with the
        # current CORPUS_MIN_YEAR / CORPUS_MAX_YEAR / MIN_TOKENS_IN_DOC /
        # MAX_TOKENS_IN_DOC settings.  Also creates the non-concurrent
        # indexes that belong to the views.
        create_views(conn)

        # GIN + supporting indexes on document_search
        create_tiered_token_indexes(conn)

    # Concurrent indexes must run outside a transaction
    # (and therefore open their own autocommit connection).
    create_concurrent_indexes()

    logger.info("Views and indexes successfully rebuilt")


if __name__ == "__main__":
    main()
