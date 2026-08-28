#!/usr/bin/env python

"""
tier4_0_lineage_graph.py

Constructs and exports a temporal semantic lineage graph from Tier 3.1
concept-year clusters.

The graph is intended to show how regions of semantic usage space persist,
split, merge, emerge, and disappear through time. Each node represents a
cluster of contextual observations for a concept in a particular publication
year.

A contextual observation is a single occurrence of a concept-bearing lexical
form in its source text, represented not merely by the token itself but by a
vector encoding the surrounding textual context in which that occurrence
appears.

These observations are produced from Tier 1 contextual embeddings: each
occurrence of a concept-bearing token in EEBO is located within its local
linguistic environment and encoded by MacBERTh. Thus two occurrences of the
same lexical form are separate observations when they appear in different
passages, documents, or rhetorical contexts.

Clustering these observations therefore groups recurring patterns of
meaning-in-use rather than occurrences of the same spelling alone. A node
represents a historically situated pattern of usage: a set of contextual
configurations in which a concept was employed during a particular period.

Nodes are connected when their centroid representations indicate a likely
continuation of the same semantic region in the following period.

The graph does not claim that individual words or concepts have a single
continuous historical identity. Instead, it models continuity and change in
the distribution of observed meanings: a cluster may persist because its
contextual usage remains close to its founding semantic position, branch into
multiple descendants as usage differentiates, merge with other semantic
regions, or terminate when no subsequent cluster provides a plausible
continuation.

Lineage assignment therefore combines two signals:

    1. structural continuity:
       whether a high-confidence temporal edge links two clusters;

    2. semantic persistence:
       whether the descendant cluster remains sufficiently close to the
       founding centroid of its lineage rather than merely following a chain
       of locally similar but progressively drifting clusters.

Neighbour-profile deltas are a separate derived Tier 4 signal. They are
computed from the completed Tier 2/3 database when this Tier 4 analysis runs,
then included in the exported lineage graph.

The graph should therefore be read as a map of changing semantic landscapes,
not as a deterministic model of lexical evolution.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import defaultdict
from sqlite3 import Connection

import networkx as nx
import numpy as np

from lib.corpus_config import CORPUS_TIER2_DB_PATH, GUI_PUBLIC_DIR
from lib.corpus_logging import logger
from lib.get_processed_concepts import get_processed_concepts
from lib.sqlite_vector_blob import blob_to_vector

from tier4.temporal_neighbour_delta import (
    build_neighbour_deltas_for_concept,
    deltas_for_export,
)


OUTPUT_DIR = GUI_PUBLIC_DIR / "lineage"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# A lineage may follow locally strong edges while gradually moving away from
# its founding semantic region. This threshold detects that cumulative drift.
DRIFT_THRESHOLD = 0.75

# Structural confidence required before an edge can continue an existing
# lineage.
CONFIDENCE_THRESHOLD = 0.95

# Keep exports inspectable without copying every observation into JSON.
EVENT_SAMPLE_SIZE = 8
NEIGHBOUR_SAMPLE_SIZE = 5

# Weak neighbour matches are excluded from both sampled contextual evidence
# and the Tier 4 neighbour-profile comparison.
MIN_NEIGHBOUR_SCORE = 0.02


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


def load_temporal_graph(con, concept):
    G = nx.DiGraph()

    nodes = con.execute(
        """
        SELECT concept, pub_year, cluster_id, point_count, centroid_vector
        FROM concept_year_cluster_info
        WHERE concept=? AND cluster_id >= 0
        ORDER BY pub_year, cluster_id
        """,
        (concept,),
    )

    for row in nodes:
        concept_row, year, cluster, size, centroid_vector = row

        node_id = f"{year}:{cluster}"

        G.add_node(
            node_id,
            concept=concept_row,
            year=int(year),
            cluster=int(cluster),
            size=int(size),
            vector=(
                blob_to_vector(centroid_vector)
                if centroid_vector is not None
                else None
            ),
        )

    edges = con.execute(
        """
        SELECT
            source_year,
            source_cluster,
            target_year,
            target_cluster,
            similarity,
            edge_type,
            confidence
        FROM temporal_cluster_edges
        WHERE concept=?
        """,
        (concept,),
    )

    for row in edges:
        (
            source_year,
            source_cluster,
            target_year,
            target_cluster,
            similarity,
            edge_type,
            confidence,
        ) = row

        source_id = f"{source_year}:{source_cluster}"
        target_id = f"{target_year}:{target_cluster}"

        # Edges referring to filtered-out clusters must not create implicit
        # NetworkX nodes without the attributes required by lineage analysis.
        if source_id not in G or target_id not in G:
            logger.warning(
                f"[tier4] skipping edge referencing missing node: "
                f"{source_id} -> {target_id}"
            )
            continue

        G.add_edge(
            source_id,
            target_id,
            similarity=float(similarity),
            edge_type=edge_type,
            confidence=(
                float(confidence)
                if confidence is not None
                else 0.0
            ),
        )

    return G


def aggregate_cluster_context(
    con,
    concept,
    year,
    cluster_id,
    limit=10,
):
    rows = con.execute(
        """
        SELECT e.token
        FROM concept_year_event_cluster c
        JOIN events e
             ON e.event_id = c.event_id
        WHERE c.concept=?
          AND c.pub_year=?
          AND c.cluster_id=?
        """,
        (
            concept,
            year,
            cluster_id,
        ),
    )

    counts = {}

    for (token,) in rows:
        if not token:
            continue

        token = token.lower()
        counts[token] = counts.get(token, 0) + 1

    return [
        {
            "token": token,
            "count": count,
        }
        for token, count in sorted(
            counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:limit]
    ]


def assign_lineages(G):
    """
    Assign lineage ids while measuring semantic persistence against each
    lineage's founding centroid rather than only its immediate parent.

    A strong sequence of local temporal edges can therefore still terminate
    and start a new lineage when cumulative semantic drift becomes too large.
    """

    lineage = {}
    merged_from = {}
    persistence_score = {}
    lineage_anchor = {}
    lineage_founder = {}
    next_lineage = 0

    nodes = sorted(
        G.nodes,
        key=lambda n: (
            G.nodes[n]["year"],
            G.nodes[n]["cluster"],
        ),
    )

    for node in nodes:
        incoming = list(G.predecessors(node))
        node_vector = G.nodes[node].get("vector")

        if not incoming:
            lineage[node] = next_lineage
            lineage_anchor[next_lineage] = node_vector
            lineage_founder[next_lineage] = node
            persistence_score[node] = 1.0
            next_lineage += 1
            continue

        # The highest-confidence parent supplies structural continuity;
        # remaining parents can be retained as merge sources.
        ranked = sorted(
            incoming,
            key=lambda parent: G.edges[parent, node]["confidence"],
            reverse=True,
        )

        parent = ranked[0]
        confidence = G.edges[parent, node]["confidence"]

        parent_lineage = lineage[parent]
        anchor_vector = lineage_anchor.get(parent_lineage)

        if node_vector is not None and anchor_vector is not None:
            persistence = cosine_similarity(
                node_vector,
                anchor_vector,
            )
        else:
            # Without vectors, structural confidence is the only available
            # continuity signal.
            persistence = confidence

        is_continuation = (
            confidence >= CONFIDENCE_THRESHOLD
            and persistence >= DRIFT_THRESHOLD
        )

        if not is_continuation:
            # The node either lacks sufficient structural support or has
            # drifted too far from its lineage origin, so it starts anew.
            lineage[node] = next_lineage
            lineage_anchor[next_lineage] = node_vector
            lineage_founder[next_lineage] = node
            persistence_score[node] = 1.0
            next_lineage += 1
        else:
            lineage[node] = parent_lineage
            persistence_score[node] = persistence

        # Preserve incoming edges from other lineages as merge evidence.
        other_lineages = sorted(
            {
                lineage[parent]
                for parent in ranked[1:]
                if parent in lineage
                and lineage[parent] != lineage[node]
            }
        )

        if other_lineages:
            merged_from[node] = other_lineages

    nx.set_node_attributes(G, lineage, "lineage")
    nx.set_node_attributes(G, merged_from, "merged_from")
    nx.set_node_attributes(G, persistence_score, "persistence")

    lineage_min_persistence = {}

    for node, lineage_id in lineage.items():
        score = persistence_score[node]

        lineage_min_persistence[lineage_id] = min(
            score,
            lineage_min_persistence.get(lineage_id, 1.0),
        )

    lineage_stability = {
        lineage_id: (
            min_score >= DRIFT_THRESHOLD
        )
        for lineage_id, min_score
        in lineage_min_persistence.items()
    }

    G.graph["lineage_min_persistence"] = lineage_min_persistence
    G.graph["lineage_stability"] = lineage_stability
    G.graph["lineage_founder"] = lineage_founder

    return G


def analyse_lineage(G):
    births = [
        node
        for node in G.nodes
        if G.in_degree(node) == 0
    ]

    deaths = [
        node
        for node in G.nodes
        if G.out_degree(node) == 0
    ]

    branching = [
        (node, G.out_degree(node))
        for node in G.nodes
        if G.out_degree(node) > 1
    ]

    merging = [
        (node, G.in_degree(node))
        for node in G.nodes
        if G.in_degree(node) > 1
    ]

    unstable = [
        lineage_id
        for lineage_id, stable
        in G.graph.get("lineage_stability", {}).items()
        if not stable
    ]

    logger.debug(
        f"[tier4] unstable lineages "
        f"(drifted below {DRIFT_THRESHOLD}):"
    )
    logger.debug(unstable)

    return {
        "births": births,
        "deaths": deaths,
        "branching": branching,
        "merging": merging,
        "unstable": unstable,
    }


def sample_cluster_events(
    con,
    concept,
    year,
    cluster_id,
    event_limit=EVENT_SAMPLE_SIZE,
    neighbour_limit=NEIGHBOUR_SAMPLE_SIZE,
    min_neighbour_score=MIN_NEIGHBOUR_SCORE,
):
    """
    Pull a deterministic sample of concrete observations from a cluster.

    Neighbours are grouped by lexical form so recurring contextual matches
    remain visible without allowing individual neighbour rows to dominate
    the exported detail.
    """

    rows = con.execute(
        """
        SELECT
            e.event_id,
            e.doc_id,
            e.token_idx,
            e.token,
            e.pub_year
        FROM concept_year_event_cluster c
        JOIN events e
             ON e.event_id = c.event_id
        WHERE c.concept=?
          AND c.pub_year=?
          AND c.cluster_id=?
        ORDER BY e.event_id
        """,
        (
            concept,
            year,
            cluster_id,
        ),
    ).fetchall()

    if not rows:
        return []

    # Membership should be unique, but defensive deduplication prevents
    # malformed historical data from duplicating exported observations.
    seen_events = set()
    unique_rows = []

    for row in rows:
        event_id = int(row[0])

        if event_id in seen_events:
            continue

        seen_events.add(event_id)
        unique_rows.append(row)

    rows = unique_rows

    # Even sampling gives a deterministic spread through the cluster rather
    # than systematically favouring the first event ids.
    if len(rows) > event_limit:
        indices = sorted(
            {
                int(round(i))
                for i in np.linspace(
                    0,
                    len(rows) - 1,
                    event_limit,
                )
            }
        )

        rows = [rows[i] for i in indices]

    samples = []

    for (
        event_id,
        doc_id,
        token_idx,
        token,
        ev_year,
    ) in rows:
        neighbour_rows = con.execute(
            """
            SELECT
                n.neighbour_event_id,
                n.token,
                n.doc_id,
                n.pub_year,
                n.token_idx,
                n.score
            FROM neighbours n
            WHERE n.event_id=?
              AND n.score >= ?
            ORDER BY n.score DESC
            LIMIT ?
            """,
            (
                event_id,
                min_neighbour_score,
                neighbour_limit * 5,
            ),
        ).fetchall()

        grouped = defaultdict(list)

        for (
            neighbour_event_id,
            neighbour_token,
            neighbour_doc_id,
            neighbour_year,
            neighbour_token_idx,
            score,
        ) in neighbour_rows:
            if not neighbour_token:
                continue

            neighbour_token = str(neighbour_token).lower()

            grouped[neighbour_token].append(
                {
                    "neighbour_event_id": int(neighbour_event_id),
                    "doc_id": neighbour_doc_id,
                    "pub_year": (
                        int(neighbour_year)
                        if neighbour_year is not None
                        else None
                    ),
                    "token_idx": (
                        int(neighbour_token_idx)
                        if neighbour_token_idx is not None
                        else None
                    ),
                    "score": float(score),
                }
            )

        neighbours = [
            {
                "token": neighbour_token,
                "count": len(neighbour_events),
                "max_score": max(
                    event["score"]
                    for event in neighbour_events
                ),
                "examples": neighbour_events[:3],
            }
            for (
                neighbour_token,
                neighbour_events,
            ) in grouped.items()
        ]

        # Recurrence is ranked ahead of isolated matches; similarity breaks
        # ties between equally recurrent contextual neighbours.
        neighbours.sort(
            key=lambda neighbour: (
                neighbour["count"],
                neighbour["max_score"],
            ),
            reverse=True,
        )

        samples.append(
            {
                "event_id": int(event_id),
                "doc_id": doc_id,
                "token_idx": (
                    int(token_idx)
                    if token_idx is not None
                    else None
                ),
                "token": token,
                "pub_year": int(ev_year),
                "neighbours": neighbours[:neighbour_limit],
            }
        )

    return samples


def export_lineage(
    con,
    concept,
    G,
    analysis=None,
):
    nodes = []

    # This only reads the Tier 4-derived delta table. Computation happens
    # once, before export, in analyse_concept_lineage().
    neighbour_deltas = deltas_for_export(
        con,
        concept,
    )

    rows = con.execute(
        """
        SELECT
            concept,
            pub_year,
            cluster_id,
            point_count,
            centroid_nx,
            centroid_ny,
            centroid_gnx,
            centroid_gny
        FROM concept_year_cluster_info
        WHERE concept=?
        ORDER BY pub_year, cluster_id
        """,
        (concept,),
    )

    lineage_stability = G.graph.get(
        "lineage_stability",
        {},
    )

    for row in rows:
        (
            concept_row,
            year,
            cluster,
            size,
            local_x,
            local_y,
            global_x,
            global_y,
        ) = row

        node_id = f"{year}:{cluster}"

        if node_id not in G:
            continue

        node_lineage = G.nodes[node_id].get("lineage")

        nodes.append(
            {
                "id": node_id,
                "year": year,
                "cluster": cluster,
                "size": size,
                "lineage": node_lineage,
                "merged_from": G.nodes[node_id].get(
                    "merged_from",
                    [],
                ),
                "persistence_score": G.nodes[node_id].get(
                    "persistence"
                ),
                "lineage_stable": lineage_stability.get(
                    node_lineage
                ),
                "local": {
                    "x": (
                        local_x
                        if local_x is not None
                        else 0
                    ),
                    "y": (
                        local_y
                        if local_y is not None
                        else 0
                    ),
                },
                "global": {
                    "x": (
                        global_x
                        if global_x is not None
                        else 0
                    ),
                    "y": (
                        global_y
                        if global_y is not None
                        else 0
                    ),
                },
                "event_sample": sample_cluster_events(
                    con,
                    concept,
                    year,
                    cluster,
                    min_neighbour_score=MIN_NEIGHBOUR_SCORE,
                ),
                "retrieval_profile": aggregate_cluster_context(
                    con,
                    concept,
                    year,
                    cluster,
                ),
            }
        )

    edges = []

    rows = con.execute(
        """
        SELECT
            source_year,
            source_cluster,
            target_year,
            target_cluster,
            similarity,
            confidence,
            edge_type
        FROM temporal_cluster_edges
        WHERE concept=?
        """,
        (concept,),
    )

    for row in rows:
        (
            source_year,
            source_cluster,
            target_year,
            target_cluster,
            similarity,
            confidence,
            edge_type,
        ) = row

        source = f"{source_year}:{source_cluster}"
        target = f"{target_year}:{target_cluster}"

        link = {
            "source": source,
            "target": target,
            "similarity": similarity,
            "confidence": confidence,
            "type": edge_type,
        }

        delta = neighbour_deltas["by_edge"].get(
            f"{source}->{target}"
        )

        if delta is not None:
            link["neighbour"] = {
                "jaccard": delta["jaccard"],
                "cosine": delta["cosine"],
                "gained": delta["gained"][:10],
                "lost": delta["lost"][:10],
                "stable": delta["stable"][:10],
            }

        edges.append(link)

    lineages_summary = [
        {
            "lineage": lineage_id,
            "min_persistence": min_score,
            "stable": lineage_stability.get(lineage_id),
        }
        for lineage_id, min_score
        in G.graph.get(
            "lineage_min_persistence",
            {},
        ).items()
    ]

    return {
        "generated": "tier4_0_lineage_graph",
        "concept": concept,
        "nodes": nodes,
        "links": edges,
        "neighbour_deltas": neighbour_deltas,
        "lineages": lineages_summary,
        "drift_threshold": DRIFT_THRESHOLD,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "neighbour_min_score": MIN_NEIGHBOUR_SCORE,
        "summary": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "lineages": len(lineages_summary),
            "stable_lineages": sum(
                bool(value)
                for value in lineage_stability.values()
            ),
            "births": (
                len(analysis["births"])
                if analysis
                else 0
            ),
            "deaths": (
                len(analysis["deaths"])
                if analysis
                else 0
            ),
            "branching": (
                len(analysis["branching"])
                if analysis
                else 0
            ),
            "merging": (
                len(analysis["merging"])
                if analysis
                else 0
            ),
            "unstable_lineages": (
                len(analysis["unstable"])
                if analysis
                else 0
            ),
        },
        "events": {
            "births": (
                analysis["births"]
                if analysis
                else []
            ),
            "deaths": (
                analysis["deaths"]
                if analysis
                else []
            ),
            "branching": (
                analysis["branching"]
                if analysis
                else []
            ),
            "merging": (
                analysis["merging"]
                if analysis
                else []
            ),
            "unstable": (
                analysis["unstable"]
                if analysis
                else []
            ),
        },
    }


def analyse_concept_lineage(
    con: Connection,
    concept: str,
    *,
    min_neighbour_score: float = MIN_NEIGHBOUR_SCORE,
) -> dict[str, object]:
    logger.info(f"[tier4] analysing {concept}")

    G = load_temporal_graph(
        con,
        concept,
    )

    G = assign_lineages(G)
    analysis = analyse_lineage(G)

    # Neighbour deltas are a derived Tier 4 analysis. The database is
    # treated as the completed Tier 2/3 input; export only reads this result.
    build_neighbour_deltas_for_concept(
        con,
        concept,
        min_score=min_neighbour_score,
        top_n=20,
    )

    return export_lineage(
        con,
        concept,
        G,
        analysis=analysis,
    )


def service(
    *,
    con: Connection,
    concept: str,
    write_json: bool = False,
    min_neighbour_score: float = MIN_NEIGHBOUR_SCORE,
) -> dict[str, object]:
    started = time.perf_counter()

    result = analyse_concept_lineage(
        con,
        concept,
        min_neighbour_score=min_neighbour_score,
    )

    elapsed = time.perf_counter() - started

    result["elapsed_seconds"] = round(
        elapsed,
        3,
    )

    if write_json:
        json_path = OUTPUT_DIR / f"{concept}_lineage.json"

        with open(
            json_path,
            "w",
            encoding="utf8",
        ) as f:
            json.dump(
                result,
                f,
                indent=2,
            )

        result["json_path"] = str(json_path)

        logger.info(
            f"[tier4-service] wrote {json_path}"
        )

    logger.info(
        f"[tier4-service] completed "
        f"{concept} in {elapsed:.2f}s"
    )

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--concept",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--min-score",
        default=MIN_NEIGHBOUR_SCORE,
        type=float,
        help="Minimum neighbour similarity used by Tier 4 neighbour-profile analysis",
    )

    args = parser.parse_args()

    con = sqlite3.connect(
        CORPUS_TIER2_DB_PATH
    )

    try:
        concepts = (
            [args.concept.upper()]
            if args.concept
            else get_processed_concepts(
                CORPUS_TIER2_DB_PATH
            )
        )

        for concept in concepts:
            result = service(
                con=con,
                concept=concept,
                write_json=args.json,
                min_neighbour_score=args.min_score,
            )

            logger.info(
                f"[tier4-main] "
                f"{result['concept']} complete"
            )

    finally:
        con.close()


if __name__ == "__main__":
    main()
