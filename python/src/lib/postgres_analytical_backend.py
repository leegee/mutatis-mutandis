# lib/postgres_analytical_backend.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from psycopg import Connection

from lib.corpus_db import get_connection


@dataclass(frozen=True)
class FieldEvent:
    concept: str
    event_id: int
    role: str


@dataclass(frozen=True)
class RetrievalRun:
    run_id: int
    concept: str
    from_year: int
    to_year: int
    seed_population: str
    neighbour_population: str
    scales: str
    top_n: int
    rrf_k: int
    oversample: int
    model: str | None
    created_at: object


class PostgresAnalyticalBackend:
    """
    PostgreSQL access to Tier 2 and Tier 3 analytical state.

    Event identity and event metadata remain authoritative in `events`.
    Tier 2 and Tier 3 store references and derived analytical results,
    not duplicated corpus metadata.
    """

    def __init__(self, conn: Connection | None = None) -> None:
        self._owned_connection = conn is None
        self.conn = conn or get_connection()

    def close(self) -> None:
        if self._owned_connection:
            self.conn.close()

    def __enter__(self) -> PostgresAnalyticalBackend:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def concepts(self) -> list[str]:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT concept
                FROM tier2.concepts
                ORDER BY concept
            """)
            return [str(row[0]) for row in cur.fetchall()]

    def concept(self, concept: str) -> dict[str, object] | None:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT
                    concept,
                    n_events
                FROM tier2.concepts
                WHERE concept = %s
            """, (concept,))

            row = cur.fetchone()

        if row is None:
            return None

        return {
            "concept": row[0],
            "n_events": row[1],
        }

    def field_events(
        self,
        concept: str,
    ) -> list[FieldEvent]:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT
                    concept,
                    event_id,
                    role
                FROM tier2.event_field
                WHERE concept = %s
                ORDER BY event_id
            """, (concept,))

            return [
                FieldEvent(
                    concept=str(row[0]),
                    event_id=int(row[1]),
                    role=str(row[2]),
                )
                for row in cur.fetchall()
            ]

    def event_metadata(
        self,
        event_ids: Sequence[int],
    ) -> list[dict[str, object]]:
        if not event_ids:
            return []

        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT
                    event_id,
                    corpus,
                    doc_id,
                    token,
                    token_idx,
                    pub_year,
                    local_window_id,
                    local_window_token_pos,
                    medium_window_id,
                    medium_window_token_pos,
                    broad_window_id,
                    broad_window_token_pos
                FROM events
                WHERE event_id = ANY(%s)
                ORDER BY event_id
            """, (list(event_ids),))

            return [
                {
                    "event_id": int(row[0]),
                    "corpus": row[1],
                    "doc_id": row[2],
                    "token": row[3],
                    "token_idx": row[4],
                    "pub_year": row[5],
                    "local_window_id": row[6],
                    "local_window_token_pos": row[7],
                    "medium_window_id": row[8],
                    "medium_window_token_pos": row[9],
                    "broad_window_id": row[10],
                    "broad_window_token_pos": row[11],
                }
                for row in cur.fetchall()
            ]

    def retrieval_runs(
        self,
        concept: str,
    ) -> list[RetrievalRun]:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT
                    run_id,
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
                FROM tier2.retrieval_runs
                WHERE concept = %s
                ORDER BY run_id
            """, (concept,))

            return [
                RetrievalRun(
                    run_id=int(row[0]),
                    concept=str(row[1]),
                    from_year=int(row[2]),
                    to_year=int(row[3]),
                    seed_population=str(row[4]),
                    neighbour_population=str(row[5]),
                    scales=str(row[6]),
                    top_n=int(row[7]),
                    rrf_k=int(row[8]),
                    oversample=int(row[9]),
                    model=row[10],
                    created_at=row[11],
                )
                for row in cur.fetchall()
            ]

    def neighbour_edges(
        self,
        run_id: int,
        *,
        limit: int | None = None,
    ) -> list[dict[str, object]]:
        sql = """
            SELECT
                seed_event_id,
                neighbour_event_id,
                depth,
                via_event_id,
                rank,
                score,
                score_local,
                score_medium,
                score_broad
            FROM tier2.neighbour_edges
            WHERE run_id = %s
            ORDER BY rank, seed_event_id, neighbour_event_id
        """

        params: list[object] = [run_id]

        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be non-negative")
            sql += " LIMIT %s"
            params.append(limit)

        with self.conn.cursor() as cur:
            cur.execute(sql, params)

            return [
                {
                    "seed_event_id": int(row[0]),
                    "neighbour_event_id": int(row[1]),
                    "depth": int(row[2]),
                    "via_event_id": (
                        int(row[3])
                        if row[3] is not None
                        else None
                    ),
                    "rank": int(row[4]),
                    "score": row[5],
                    "score_local": row[6],
                    "score_medium": row[7],
                    "score_broad": row[8],
                }
                for row in cur.fetchall()
            ]

    def geometry(
        self,
        concept: str,
    ) -> list[dict[str, object]]:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT
                    event_id,
                    nx,
                    ny,
                    gnx,
                    gny,
                    cluster_id,
                    cluster_label
                FROM tier3.event_geometry
                WHERE concept = %s
                ORDER BY event_id
            """, (concept,))

            return [
                {
                    "event_id": int(row[0]),
                    "nx": float(row[1]),
                    "ny": float(row[2]),
                    "gnx": (
                        float(row[3])
                        if row[3] is not None
                        else None
                    ),
                    "gny": (
                        float(row[4])
                        if row[4] is not None
                        else None
                    ),
                    "cluster_id": int(row[5]),
                    "cluster_label": row[6],
                }
                for row in cur.fetchall()
            ]

    def clusters(
        self,
        concept: str,
    ) -> list[dict[str, object]]:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT
                    cluster_id,
                    cluster_label,
                    centroid_nx,
                    centroid_ny,
                    centroid_gnx,
                    centroid_gny,
                    centroid_vector,
                    point_count,
                    description
                FROM tier3.concept_cluster_info
                WHERE concept = %s
                ORDER BY cluster_id
            """, (concept,))

            return [
                {
                    "cluster_id": int(row[0]),
                    "cluster_label": row[1],
                    "centroid_nx": row[2],
                    "centroid_ny": row[3],
                    "centroid_gnx": row[4],
                    "centroid_gny": row[5],
                    "centroid_vector": row[6],
                    "point_count": int(row[7]),
                    "description": row[8],
                }
                for row in cur.fetchall()
            ]
