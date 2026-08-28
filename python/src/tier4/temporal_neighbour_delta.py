#!/usr/bin/env python

"""
temporal_neighbour_delta.py

Builds neighbour-profile continuity signals for Tier 4 temporal lineage edges.

For each edge in temporal_cluster_edges, the module compares the lexical
neighbour fields of the source and target year-clusters.

The signal is deliberately separate from centroid similarity:

    centroid similarity
        = geometric continuity in embedding space

    neighbour delta
        = continuity/change in the lexical contextual field

This module is invoked by Tier 4 rather than being a separate pipeline stage.
It derives and persists the neighbour-delta table in the Tier 2/3 SQLite
database so that lineage export can subsequently read it.

The database is therefore the boundary between computation and presentation.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from typing import Any, Iterable, Optional, Sequence


DELTA_SCHEMA = """
CREATE TABLE IF NOT EXISTS temporal_neighbour_deltas (
    concept TEXT NOT NULL,
    source_year INTEGER NOT NULL,
    source_cluster INTEGER NOT NULL,
    target_year INTEGER NOT NULL,
    target_cluster INTEGER NOT NULL,

    jaccard REAL,
    cosine REAL,
    source_token_n INTEGER,
    target_token_n INTEGER,
    shared_token_n INTEGER,

    gained_json TEXT,
    lost_json TEXT,
    stable_json TEXT,

    PRIMARY KEY (
        concept,
        source_year,
        source_cluster,
        target_year,
        target_cluster
    )
);

CREATE INDEX IF NOT EXISTS idx_neighbour_deltas_concept
    ON temporal_neighbour_deltas (concept);

CREATE INDEX IF NOT EXISTS idx_neighbour_deltas_transition
    ON temporal_neighbour_deltas (
        concept,
        source_year,
        target_year
    );
"""


def initialise_delta_tables(
    con: sqlite3.Connection,
) -> None:
    con.executescript(DELTA_SCHEMA)
    con.commit()


def clear_deltas(
    con: sqlite3.Connection,
    concept: Optional[str] = None,
) -> None:
    if concept is None:
        con.execute(
            "DELETE FROM temporal_neighbour_deltas"
        )
    else:
        con.execute(
            """
            DELETE FROM temporal_neighbour_deltas
            WHERE concept=?
            """,
            (concept,),
        )

    con.commit()


def load_cluster_event_ids(
    con: sqlite3.Connection,
    concept: str,
    pub_year: int,
    cluster_id: int,
) -> list[int]:
    rows = con.execute(
        """
        SELECT event_id
        FROM concept_year_event_cluster
        WHERE concept=?
          AND pub_year=?
          AND cluster_id=?
        ORDER BY event_id
        """,
        (
            concept,
            pub_year,
            cluster_id,
        ),
    ).fetchall()

    return [int(row[0]) for row in rows]


def neighbour_token_counts(
    con: sqlite3.Connection,
    event_ids: Sequence[int],
    *,
    min_score: Optional[float] = None,
) -> Counter:
    """
    Aggregate lexical neighbour tokens for a set of cluster-member events.

    Each qualifying neighbour occurrence contributes one count. Repeated
    matches across cluster members therefore measure recurrence of a
    contextual neighbour within the cluster.

    The current neighbours schema has no depth field: every row in
    neighbours is treated as an available neighbour observation, filtered
    only by event membership and optional similarity threshold.
    """
    if not event_ids:
        return Counter()

    counts: Counter = Counter()

    chunk = 500

    for i in range(0, len(event_ids), chunk):
        batch = list(event_ids[i : i + chunk])
        placeholders = ",".join("?" * len(batch))

        params: list[Any] = list(batch)

        score_clause = ""

        if min_score is not None:
            score_clause = " AND n.score >= ? "
            params.append(float(min_score))

        rows = con.execute(
            f"""
            SELECT
                lower(e.token) AS tok,
                COUNT(*) AS c
            FROM neighbours n
            JOIN events e
                ON e.event_id = n.neighbour_event_id
            WHERE n.event_id IN ({placeholders})
              AND e.token IS NOT NULL
              AND length(e.token) > 0
              {score_clause}
            GROUP BY lower(e.token)
            """,
            params,
        ).fetchall()

        for tok, count in rows:
            if tok:
                counts[str(tok)] += int(count)

    return counts


def cluster_neighbour_profile(
    con: sqlite3.Connection,
    concept: str,
    pub_year: int,
    cluster_id: int,
    *,
    min_score: Optional[float] = None,
) -> Counter:
    event_ids = load_cluster_event_ids(
        con,
        concept,
        pub_year,
        cluster_id,
    )

    return neighbour_token_counts(
        con,
        event_ids,
        min_score=min_score,
    )


def _jaccard(
    source: Counter,
    target: Counter,
) -> float:
    source_tokens = set(source)
    target_tokens = set(target)

    if not source_tokens and not target_tokens:
        return 1.0

    if not source_tokens or not target_tokens:
        return 0.0

    intersection = len(
        source_tokens & target_tokens
    )

    union = len(
        source_tokens | target_tokens
    )

    return (
        intersection / union
        if union
        else 0.0
    )


def _cosine_counts(
    source: Counter,
    target: Counter,
) -> float:
    if not source or not target:
        return 0.0

    keys = set(source) | set(target)

    dot = 0.0
    source_norm = 0.0
    target_norm = 0.0

    for key in keys:
        source_value = float(
            source.get(key, 0)
        )
        target_value = float(
            target.get(key, 0)
        )

        dot += source_value * target_value
        source_norm += source_value * source_value
        target_norm += target_value * target_value

    denominator = (
        math.sqrt(source_norm)
        * math.sqrt(target_norm)
    )

    return (
        float(dot / denominator)
        if denominator > 0
        else 0.0
    )


def compute_neighbour_delta(
    source: Counter,
    target: Counter,
    *,
    top_n: int = 20,
) -> dict[str, Any]:
    """
    Compare two neighbour-token count profiles.

    gained:
        target count > source count

    lost:
        source count > target count

    stable:
        token occurs in both profiles

    Aggregate similarity metrics use the complete profiles. The explanatory
    gained/lost/stable lists are reduced to top_n entries for export.
    """
    all_tokens = set(source) | set(target)

    gained: list[dict[str, Any]] = []
    lost: list[dict[str, Any]] = []
    stable: list[dict[str, Any]] = []

    for token in all_tokens:
        source_count = int(
            source.get(token, 0)
        )
        target_count = int(
            target.get(token, 0)
        )

        delta = target_count - source_count

        row = {
            "token": token,
            "source_count": source_count,
            "target_count": target_count,
            "delta": delta,
        }

        if source_count > 0 and target_count > 0:
            stable.append(row)

        if delta > 0:
            gained.append(row)
        elif delta < 0:
            lost.append(row)

    gained.sort(
        key=lambda row: (
            -row["delta"],
            -row["target_count"],
            row["token"],
        )
    )

    lost.sort(
        key=lambda row: (
            row["delta"],
            -row["source_count"],
            row["token"],
        )
    )

    stable.sort(
        key=lambda row: (
            -min(
                row["source_count"],
                row["target_count"],
            ),
            -row["source_count"],
            row["token"],
        )
    )

    return {
        "jaccard": _jaccard(
            source,
            target,
        ),
        "cosine": _cosine_counts(
            source,
            target,
        ),
        "source_token_n": len(source),
        "target_token_n": len(target),
        "shared_token_n": len(
            set(source) & set(target)
        ),
        "gained": gained[:top_n],
        "lost": lost[:top_n],
        "stable": stable[:top_n],
    }


def load_temporal_edges(
    con: sqlite3.Connection,
    concept: str,
) -> list[tuple[int, int, int, int, str]]:
    rows = con.execute(
        """
        SELECT
            source_year,
            source_cluster,
            target_year,
            target_cluster,
            edge_type
        FROM temporal_cluster_edges
        WHERE concept=?
        ORDER BY
            source_year,
            source_cluster,
            target_year,
            target_cluster
        """,
        (concept,),
    ).fetchall()

    return [
        (
            int(source_year),
            int(source_cluster),
            int(target_year),
            int(target_cluster),
            str(edge_type),
        )
        for (
            source_year,
            source_cluster,
            target_year,
            target_cluster,
            edge_type,
        ) in rows
    ]


def build_neighbour_deltas_for_concept(
    con: sqlite3.Connection,
    concept: str,
    *,
    top_n: int = 20,
    min_score: Optional[float] = None,
    replace: bool = True,
) -> int:
    """
    Compute and store neighbour-token deltas for every temporal edge
    belonging to a concept.

    This is a derived Tier 4 calculation. Temporal edges and neighbour
    observations are treated as completed upstream inputs.
    """
    initialise_delta_tables(con)

    if replace:
        con.execute(
            """
            DELETE FROM temporal_neighbour_deltas
            WHERE concept=?
            """,
            (concept,),
        )

    edges = load_temporal_edges(
        con,
        concept,
    )

    if not edges:
        con.commit()
        return 0

    profile_cache: dict[
        tuple[int, int],
        Counter,
    ] = {}

    def profile(
        year: int,
        cluster: int,
    ) -> Counter:
        key = (year, cluster)

        if key not in profile_cache:
            profile_cache[key] = cluster_neighbour_profile(
                con,
                concept,
                year,
                cluster,
                min_score=min_score,
            )

        return profile_cache[key]

    rows_out: list[tuple] = []

    for (
        source_year,
        source_cluster,
        target_year,
        target_cluster,
        _edge_type,
    ) in edges:
        source = profile(
            source_year,
            source_cluster,
        )

        target = profile(
            target_year,
            target_cluster,
        )

        delta = compute_neighbour_delta(
            source,
            target,
            top_n=top_n,
        )

        rows_out.append(
            (
                concept,
                source_year,
                source_cluster,
                target_year,
                target_cluster,
                delta["jaccard"],
                delta["cosine"],
                delta["source_token_n"],
                delta["target_token_n"],
                delta["shared_token_n"],
                json.dumps(
                    delta["gained"],
                    ensure_ascii=False,
                ),
                json.dumps(
                    delta["lost"],
                    ensure_ascii=False,
                ),
                json.dumps(
                    delta["stable"],
                    ensure_ascii=False,
                ),
            )
        )

    con.executemany(
        """
        INSERT OR REPLACE INTO temporal_neighbour_deltas (
            concept,
            source_year,
            source_cluster,
            target_year,
            target_cluster,
            jaccard,
            cosine,
            source_token_n,
            target_token_n,
            shared_token_n,
            gained_json,
            lost_json,
            stable_json
        )
        VALUES (
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?
        )
        """,
        rows_out,
    )

    con.commit()

    return len(rows_out)


def build_neighbour_deltas_for_all(
    con: sqlite3.Connection,
    concepts: Optional[Iterable[str]] = None,
    **kwargs,
) -> dict[str, int]:
    """
    Build neighbour deltas for every requested concept.

    If concepts is omitted, concepts are discovered from temporal_cluster_edges.
    """
    if concepts is None:
        concepts = [
            row[0]
            for row in con.execute(
                """
                SELECT DISTINCT concept
                FROM temporal_cluster_edges
                ORDER BY concept
                """
            )
        ]

    results: dict[str, int] = {}

    for concept in concepts:
        results[concept] = build_neighbour_deltas_for_concept(
            con,
            concept,
            **kwargs,
        )

    return results


def load_deltas_for_concept(
    con: sqlite3.Connection,
    concept: str,
) -> list[dict[str, Any]]:
    initialise_delta_tables(con)

    rows = con.execute(
        """
        SELECT
            source_year,
            source_cluster,
            target_year,
            target_cluster,
            jaccard,
            cosine,
            source_token_n,
            target_token_n,
            shared_token_n,
            gained_json,
            lost_json,
            stable_json
        FROM temporal_neighbour_deltas
        WHERE concept=?
        ORDER BY
            source_year,
            source_cluster,
            target_year,
            target_cluster
        """,
        (concept,),
    ).fetchall()

    results = []

    for row in rows:
        (
            source_year,
            source_cluster,
            target_year,
            target_cluster,
            jaccard,
            cosine,
            source_token_n,
            target_token_n,
            shared_token_n,
            gained_json,
            lost_json,
            stable_json,
        ) = row

        results.append(
            {
                "source": f"{source_year}:{source_cluster}",
                "target": f"{target_year}:{target_cluster}",
                "source_year": int(source_year),
                "source_cluster": int(source_cluster),
                "target_year": int(target_year),
                "target_cluster": int(target_cluster),
                "jaccard": (
                    float(jaccard)
                    if jaccard is not None
                    else None
                ),
                "cosine": (
                    float(cosine)
                    if cosine is not None
                    else None
                ),
                "source_token_n": int(
                    source_token_n or 0
                ),
                "target_token_n": int(
                    target_token_n or 0
                ),
                "shared_token_n": int(
                    shared_token_n or 0
                ),
                "gained": json.loads(
                    gained_json or "[]"
                ),
                "lost": json.loads(
                    lost_json or "[]"
                ),
                "stable": json.loads(
                    stable_json or "[]"
                ),
            }
        )

    return results


def deltas_for_export(
    con: sqlite3.Connection,
    concept: str,
) -> dict[str, Any]:
    """
    Shape persisted neighbour deltas for inclusion in Tier 4 JSON.

    The by_edge mapping gives export_lineage O(1) lookup when annotating
    individual temporal links.
    """
    deltas = load_deltas_for_concept(
        con,
        concept,
    )

    by_edge = {
        f"{delta['source']}->{delta['target']}": delta
        for delta in deltas
    }

    return {
        "edges": deltas,
        "by_edge": by_edge,
        "n_edges": len(deltas),
    }
