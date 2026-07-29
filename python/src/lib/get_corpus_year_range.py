from psycopg import sql

from lib.eebo_db import get_connection
from lib.eebo_logging import logger


def get_corpus_year_range(view_name: str = "pamphlet_corpus") -> tuple[int | None, int | None]:
    """
    Return the minimum and maximum publication years from a document
    materialized view.
    """

    logger.info(f"Getting year range from {view_name}")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("""
                    SELECT MIN(pub_year), MAX(pub_year)
                    FROM {view}
                    WHERE pub_year IS NOT NULL;
                """).format(
                    view=sql.Identifier(view_name)
                )
            )

            min_year, max_year = cur.fetchone()

    logger.info(f"{view_name} year range: {min_year} - {max_year}")

    return min_year, max_year
