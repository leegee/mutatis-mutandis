#!/usr/bin/env python
"""
temporal_neighbour_delta.py — Neighbour-profile continuity for temporal edges

Tier 3.1 links year-clusters by centroid cosine (CONTINUATION / SIGNIFICANT).
This module adds a parallel signal: how the *neighbour token field* of a
source year-cluster relates to the target year-cluster.

For each row in temporal_cluster_edges:

  1. Collect Tier 2 neighbours of all events in the source cluster
  2. Same for the target cluster
  3. Compare token multisets → Jaccard, cosine-on-count-vectors,
     gained / lost / stable token lists

Results are written to temporal_neighbour_deltas so Tier 4 (or any
explorer) can show why a lineage step is geometrically continuous but
contextually shifting, or the reverse.

Schema expectations (Tier 2 + 3.1)
---------------------------------
  concept_year_event_cluster(concept, pub_year, event_id, cluster_id)
  neighbours(event_id, neighbour_event_id, score, ...)
  events(event_id, token, ...)
  temporal_cluster_edges(concept, source_year, source_cluster,
                         target_year, target_cluster, similarity,
                         edge_type, confidence)

Usage
-----
    from temporal_neighbour_delta import (
        initialise_delta_tables,
        build_neighbour_deltas_for_concept,
        deltas_for_export,
    )

    initialise_delta_tables(con)
    build_neighbour_deltas_for_concept(con, "LAW")

CLI
---
    python temporal_neighbour_delta.py --db PATH --concept LAW
    python temporal_neighbour_delta.py --db PATH --all
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

DELTA_SCHEMA = """
CREATE TABLE IF NOT EXISTS temporal_neighbour_deltas (
    concept TEXT NOT NULL,
    source_year INTEGER NOT NULL,
    source_cluster INTEGER NOT NULL,
    target_year INTEGER NOT NULL,
    target_cluster INTEGER NOT NULL,

    -- Distribution similarity (0..1 style where defined)
    jaccard REAL,
    cosine REAL,
    source_token_n INTEGER,
    target_token_n INTEGER,
    shared_token_n INTEGER,

    -- JSON arrays of {token, source_count, target_count, delta}
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
    ON temporal_neighbour_deltas (concept, source_year, target_year);
"""


def initialise_delta_tables(con: sqlite3.Connection) -> None:
    con.executescript(DELTA_SCHEMA)
    con.commit()


def clear_deltas(con: sqlite3.Connection, concept: Optional[str] = None) -> None:
    if concept is None:
        con.execute("DELETE FROM temporal_neighbour_deltas")
    else:
        con.execute(
            "DELETE FROM temporal_neighbour_deltas WHERE concept=?",
            (concept,),
        )
    con.commit()


# ---------------------------------------------------------------------------
# Neighbour profiles per year-cluster
# ---------------------------------------------------------------------------

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
        WHERE concept=? AND pub_year=? AND cluster_id=?
        ORDER BY event_id
        """,
        (concept, pub_year, cluster_id),
    ).fetchall()
    return [int(r[0]) for r in rows]


def neighbour_token_counts(
    con: sqlite3.Connection,
    event_ids: Sequence[int],
    *,
    min_score: Optional[float] = None,
    max_depth: int = 1,
) -> Counter:
    """
    Aggregate neighbour tokens for a set of seed/member event_ids.

    Counts every neighbour row (token from events) whose score clears
    min_score and depth <= max_depth. Repeated neighbours across member
    events increase the count — recurrence is the signal.
    """
    if not event_ids:
        return Counter()

    counts: Counter = Counter()
    # Chunk IN lists to stay under SQLite variable limits.
    chunk = 500
    for i in range(0, len(event_ids), chunk):
        batch = list(event_ids[i : i + chunk])
        placeholders = ",".join("?" * len(batch))
        params: list[Any] = list(batch) + [max_depth]
        score_clause = ""
        if min_score is not None:
            score_clause = " AND n.score >= ? "
            params.append(float(min_score))

        rows = con.execute(
            f"""
            SELECT lower(e.token) AS tok, COUNT(*) AS c
            FROM neighbours n
            JOIN events e ON e.event_id = n.neighbour_event_id
            WHERE n.event_id IN ({placeholders})
              AND n.depth <= ?
              AND e.token IS NOT NULL
              AND length(e.token) > 0
              {score_clause}
            GROUP BY lower(e.token)
            """,
            params,
        ).fetchall()
        for tok, c in rows:
            if tok:
                counts[str(tok)] += int(c)
    return counts


def cluster_neighbour_profile(
    con: sqlite3.Connection,
    concept: str,
    pub_year: int,
    cluster_id: int,
    *,
    min_score: Optional[float] = None,
    max_depth: int = 1,
) -> Counter:
    eids = load_cluster_event_ids(con, concept, pub_year, cluster_id)
    return neighbour_token_counts(
        con, eids, min_score=min_score, max_depth=max_depth
    )


# ---------------------------------------------------------------------------
# Delta between two profiles
# ---------------------------------------------------------------------------

def _jaccard(a: Counter, b: Counter) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _cosine_counts(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = 0.0
    na = 0.0
    nb = 0.0
    for k in keys:
        va = float(a.get(k, 0))
        vb = float(b.get(k, 0))
        dot += va * vb
        na += va * va
        nb += vb * vb
    denom = math.sqrt(na) * math.sqrt(nb)
    return float(dot / denom) if denom > 0 else 0.0


def compute_neighbour_delta(
    source: Counter,
    target: Counter,
    *,
    top_n: int = 20,
) -> dict[str, Any]:
    """
    Compare two neighbour-token count profiles.

    gained  — tokens with target_count > source_count (sorted by delta desc)
    lost    — tokens with source_count > target_count
    stable  — tokens present on both sides, sorted by min(count) desc
    """
    all_tokens = set(source) | set(target)
    gained: list[dict[str, Any]] = []
    lost: list[dict[str, Any]] = []
    stable: list[dict[str, Any]] = []

    for tok in all_tokens:
        sc = int(source.get(tok, 0))
        tc = int(target.get(tok, 0))
        delta = tc - sc
        row = {
            "token": tok,
            "source_count": sc,
            "target_count": tc,
            "delta": delta,
        }
        if sc > 0 and tc > 0:
            stable.append(row)
        if delta > 0:
            gained.append(row)
        elif delta < 0:
            lost.append(row)

    gained.sort(key=lambda r: (-r["delta"], -r["target_count"], r["token"]))
    lost.sort(key=lambda r: (r["delta"], -r["source_count"], r["token"]))
    stable.sort(
        key=lambda r: (
            -min(r["source_count"], r["target_count"]),
            -r["source_count"],
            r["token"],
        )
    )

    return {
        "jaccard": _jaccard(source, target),
        "cosine": _cosine_counts(source, target),
        "source_token_n": len(source),
        "target_token_n": len(target),
        "shared_token_n": len(set(source) & set(target)),
        "gained": gained[:top_n],
        "lost": lost[:top_n],
        "stable": stable[:top_n],
    }


# ---------------------------------------------------------------------------
# Build + persist for a concept
# ---------------------------------------------------------------------------

def load_temporal_edges(
    con: sqlite3.Connection,
    concept: str,
) -> list[tuple[int, int, int, int, str]]:
    rows = con.execute(
        """
        SELECT source_year, source_cluster, target_year, target_cluster, edge_type
        FROM temporal_cluster_edges
        WHERE concept=?
        ORDER BY source_year, source_cluster, target_year, target_cluster
        """,
        (concept,),
    ).fetchall()
    return [
        (int(sy), int(sc), int(ty), int(tc), str(et))
        for sy, sc, ty, tc, et in rows
    ]


def build_neighbour_deltas_for_concept(
    con: sqlite3.Connection,
    concept: str,
    *,
    top_n: int = 20,
    min_score: Optional[float] = None,
    max_depth: int = 1,
    replace: bool = True,
) -> int:
    """
    Compute and store neighbour deltas for every temporal edge of concept.

    Returns number of delta rows written.
    """
    initialise_delta_tables(con)
    if replace:
        con.execute(
            "DELETE FROM temporal_neighbour_deltas WHERE concept=?",
            (concept,),
        )

    edges = load_temporal_edges(con, concept)
    if not edges:
        con.commit()
        return 0

    # Cache profiles: (year, cluster) -> Counter
    profile_cache: dict[tuple[int, int], Counter] = {}

    def profile(year: int, cluster: int) -> Counter:
        key = (year, cluster)
        if key not in profile_cache:
            profile_cache[key] = cluster_neighbour_profile(
                con,
                concept,
                year,
                cluster,
                min_score=min_score,
                max_depth=max_depth,
            )
        return profile_cache[key]

    rows_out: list[tuple] = []
    for sy, sc, ty, tc, _edge_type in edges:
        src = profile(sy, sc)
        tgt = profile(ty, tc)
        delta = compute_neighbour_delta(src, tgt, top_n=top_n)
        rows_out.append(
            (
                concept,
                sy,
                sc,
                ty,
                tc,
                delta["jaccard"],
                delta["cosine"],
                delta["source_token_n"],
                delta["target_token_n"],
                delta["shared_token_n"],
                json.dumps(delta["gained"], ensure_ascii=False),
                json.dumps(delta["lost"], ensure_ascii=False),
                json.dumps(delta["stable"], ensure_ascii=False),
            )
        )

    con.executemany(
        """
        INSERT OR REPLACE INTO temporal_neighbour_deltas (
            concept,
            source_year, source_cluster,
            target_year, target_cluster,
            jaccard, cosine,
            source_token_n, target_token_n, shared_token_n,
            gained_json, lost_json, stable_json
        )
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
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
    if concepts is None:
        concepts = [
            r[0]
            for r in con.execute(
                "SELECT DISTINCT concept FROM temporal_cluster_edges ORDER BY concept"
            )
        ]
    out: dict[str, int] = {}
    for concept in concepts:
        out[concept] = build_neighbour_deltas_for_concept(
            con, concept, **kwargs
        )
    return out


# ---------------------------------------------------------------------------
# Export helpers (Tier 4 JSON enrichment)
# ---------------------------------------------------------------------------

def load_deltas_for_concept(
    con: sqlite3.Connection,
    concept: str,
) -> list[dict[str, Any]]:
    initialise_delta_tables(con)
    rows = con.execute(
        """
        SELECT source_year, source_cluster, target_year, target_cluster,
               jaccard, cosine,
               source_token_n, target_token_n, shared_token_n,
               gained_json, lost_json, stable_json
        FROM temporal_neighbour_deltas
        WHERE concept=?
        ORDER BY source_year, source_cluster, target_year, target_cluster
        """,
        (concept,),
    ).fetchall()

    results = []
    for row in rows:
        (
            sy, sc, ty, tc,
            jaccard, cosine,
            sn, tn, shared,
            gained_j, lost_j, stable_j,
        ) = row
        results.append(
            {
                "source": f"{sy}:{sc}",
                "target": f"{ty}:{tc}",
                "source_year": int(sy),
                "source_cluster": int(sc),
                "target_year": int(ty),
                "target_cluster": int(tc),
                "jaccard": float(jaccard) if jaccard is not None else None,
                "cosine": float(cosine) if cosine is not None else None,
                "source_token_n": int(sn or 0),
                "target_token_n": int(tn or 0),
                "shared_token_n": int(shared or 0),
                "gained": json.loads(gained_j or "[]"),
                "lost": json.loads(lost_j or "[]"),
                "stable": json.loads(stable_j or "[]"),
            }
        )
    return results


def deltas_for_export(
    con: sqlite3.Connection,
    concept: str,
) -> dict[str, Any]:
    """
    Shape suitable for merging into Tier 4 lineage JSON:

        result["neighbour_deltas"] = deltas_for_export(con, concept)
    """
    deltas = load_deltas_for_concept(con, concept)
    by_edge = {
        f"{d['source']}->{d['target']}": d for d in deltas
    }
    return {
        "edges": deltas,
        "by_edge": by_edge,
        "n_edges": len(deltas),
    }


def attach_deltas_to_lineage_export(
    lineage_export: dict[str, Any],
    con: sqlite3.Connection,
    concept: Optional[str] = None,
) -> dict[str, Any]:
    """
    Non-destructive enrichment of a Tier 4 export_lineage() dict.
    Adds neighbour_deltas at top level and annotates each link when
    a matching delta exists.
    """
    concept = concept or lineage_export.get("concept")
    if not concept:
        return lineage_export

    payload = deltas_for_export(con, concept)
    lineage_export = dict(lineage_export)
    lineage_export["neighbour_deltas"] = payload

    links = lineage_export.get("links") or []
    enriched_links = []
    for link in links:
        key = f"{link.get('source')}->{link.get('target')}"
        delta = payload["by_edge"].get(key)
        link = dict(link)
        if delta is not None:
            link["neighbour_jaccard"] = delta["jaccard"]
            link["neighbour_cosine"] = delta["cosine"]
            link["neighbour_gained"] = delta["gained"][:10]
            link["neighbour_lost"] = delta["lost"][:10]
            link["neighbour_stable"] = delta["stable"][:10]
        enriched_links.append(link)
    lineage_export["links"] = enriched_links
    return lineage_export


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(
        description="Build neighbour-token deltas for temporal cluster edges"
    )
    p.add_argument(
        "--db",
        type=str,
        required=True,
        help="Path to Tier 2/3 SQLite DB (concept_year_event_cluster, "
        "temporal_cluster_edges, neighbours, events)",
    )
    p.add_argument("-c", "--concept", default=None, help="Single concept")
    p.add_argument(
        "--all",
        action="store_true",
        help="All concepts present in temporal_cluster_edges",
    )
    p.add_argument("--top-n", type=int, default=20, help="Tokens per gained/lost/stable list")
    p.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Optional minimum neighbour score filter",
    )
    p.add_argument("--max-depth", type=int, default=1)
    p.add_argument(
        "--export-json",
        type=str,
        default=None,
        help="Write deltas_for_export JSON for --concept to this path",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"DB not found: {db}")

    con = sqlite3.connect(str(db))
    try:
        initialise_delta_tables(con)
        if args.concept:
            n = build_neighbour_deltas_for_concept(
                con,
                args.concept.upper(),
                top_n=args.top_n,
                min_score=args.min_score,
                max_depth=args.max_depth,
            )
            print(f"{args.concept.upper()}: {n} delta edges")
            if args.export_json:
                payload = deltas_for_export(con, args.concept.upper())
                Path(args.export_json).write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(f"wrote {args.export_json}")
        elif args.all:
            counts = build_neighbour_deltas_for_all(
                con,
                top_n=args.top_n,
                min_score=args.min_score,
                max_depth=args.max_depth,
            )
            for concept, n in counts.items():
                print(f"{concept}: {n} delta edges")
            print(f"total concepts={len(counts)} edges={sum(counts.values())}")
        else:
            raise SystemExit("Specify --concept NAME or --all")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
