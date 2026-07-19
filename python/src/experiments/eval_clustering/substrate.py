#!/usr/bin/env python
"""
substrate.py

Read-only interface onto the existing semantic substrate.

The evaluation framework intentionally does not own any semantic data.
It consumes the existing corpus infrastructure without modifying it.

Canonical sources

    SQLite
        event metadata

    Zarr
        event lookup

    FAISS
        semantic geometry

The substrate exposes only the operations required by the experimental
clustering pipelines. It never writes back to the corpus.
"""

from __future__ import annotations


class Substrate:
    """
    Read-only view of the semantic substrate.

    The evaluation framework operates entirely from existing data.
    Embeddings are reconstructed directly from FAISS, making the FAISS
    index the single geometric authority.
    """

    def __init__(
        self,
        db,
        lookup,
        index,
    ):
        self.db = db
        self.lookup = lookup
        self.index = index

    def get_events_by_year(self, year: int):
        """
        Return all event IDs published in the supplied year.
        """
        cur = self.db.execute(
            """
            SELECT event_id
            FROM events
            WHERE pub_year = ?
            ORDER BY event_id
            """,
            (year,),
        )
        return [row[0] for row in cur.fetchall()]


    def get_events_by_concept(self, concept: str):
        """
        Return all event IDs belonging to a concept.
        """
        cur = self.db.execute(
            """
            SELECT event_id
            FROM events
            WHERE concept = ?
            ORDER BY event_id
            """,
            (concept,),
        )
        return [row[0] for row in cur.fetchall()]


    def get_events(self, year: int | None = None, concept: str | None = None):
        """
        Return event IDs satisfying the supplied filters.

        Both filters are optional and may be combined.
        """
        sql = ["SELECT event_id FROM events"]
        where = []
        params = []

        if year is not None:
            where.append("pub_year = ?")
            params.append(year)

        if concept is not None:
            where.append("concept = ?")
            params.append(concept)

        if where:
            sql.append("WHERE " + " AND ".join(where))

        sql.append("ORDER BY event_id")

        cur = self.db.execute(
            "\n".join(sql),
            params,
        )
        return [row[0] for row in cur.fetchall()]


    def get_embeddings(self, event_ids):
        return self.lookup.get_vectors(event_ids)


    def knn(self, event_ids, k):
        X = self.get_embeddings(event_ids)
        return self.index.search(X, k)
