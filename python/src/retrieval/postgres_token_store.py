from __future__ import annotations

from typing import Any

import psycopg

from .context_models import ContextToken


class PostgresTokenStore:
    """Retrieve corpus tokens from PostgreSQL."""

    def __init__(
        self,
        connection: psycopg.Connection,
    ) -> None:
        self._connection = connection

    def get_context(
        self,
        *,
        corpus: str,
        doc_id: str,
        token_idx: int,
        before: int,
        after: int,
    ) -> tuple[ContextToken, ...]:
        start_idx = max(
            0,
            token_idx - before,
        )

        end_idx = token_idx + after

        with self._connection.cursor() as cur:
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
                corpus=str(row[0]),
                doc_id=str(row[1]),
                token_idx=int(row[2]),
                token=str(row[3]),
            )
            for row in rows
        )
