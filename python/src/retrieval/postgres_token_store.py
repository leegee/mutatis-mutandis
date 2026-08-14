# retrieval/postgres_token_store.py

from __future__ import annotations

from psycopg import Connection

from retrieval.parquet_context import ContextToken


class PostgresTokenStore:
    """
    Retrieves canonical corpus tokens from PostgreSQL.

    This class knows nothing about observations, embeddings, ANN
    indexes or Parquet. It simply exposes the corpus token stream.
    """

    def __init__(
        self,
        conn: Connection,
    ) -> None:
        self._conn = conn

    def get_context(
        self,
        *,
        corpus: str,
        doc_id: str,
        token_idx: int,
        before: int,
        after: int,
    ) -> tuple[ContextToken, ...]:
        """
        Return the contiguous token sequence surrounding one corpus token.

        The returned tuple includes the centre token.
        """

        start_idx = max(
            0,
            token_idx - before,
        )

        end_idx = token_idx + after

        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    corpus,
                    doc_id,
                    token_idx,
                    token
                FROM tokens
                WHERE corpus = %s
                  AND doc_id = %s
                  AND token_idx BETWEEN %s AND %s
                ORDER BY token_idx
                """,
                (
                    corpus,
                    doc_id,
                    start_idx,
                    end_idx,
                ),
            )

            rows = cur.fetchall()

        return tuple(
            ContextToken(
                corpus=row[0],
                doc_id=row[1],
                token_idx=int(row[2]),
                token=str(row[3]),
            )
            for row in rows
        )
