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
same lexical form (for example, "law") are separate observations if they
appear in different passages, documents, or rhetorical contexts. Conversely,
occurrences from different documents may occupy nearby regions of semantic
space when their surrounding contexts are similar.

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

The resulting graph is designed as an exploratory Digital Humanities
instrument: a way to inspect possible trajectories of meaning change in the
EEBO corpus, linking computationally identified semantic movements back to
concrete contextual observations and source documents.

The graph should therefore be read as a map of changing semantic landscapes,
not as a deterministic model of lexical evolution.
"""

from __future__ import annotations

import argparse
import time
import sqlite3
from sqlite3 import Connection
import json
from collections import defaultdict
import numpy as np
import networkx as nx

from lib.corpus_config import CORPUS_TIER2_DB_PATH, GUI_PUBLIC_DIR
from lib.sqlite_vector_blob import blob_to_vector
from lib.corpus_logging import logger
from lib.get_processed_concepts import get_processed_concepts

OUTPUT_DIR = GUI_PUBLIC_DIR / 'lineage'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Below this cosine similarity to its lineage's founding centroid, a node
# is considered to have drifted away from the original concept even if
# the edge that produced it had high structural confidence. This decouples
# "plausible successor cluster" (edge confidence) from "still the same
# concept" (semantic persistence).
DRIFT_THRESHOLD = 0.75

# Structural confidence floor for treating an edge as a continuation at all.
CONFIDENCE_THRESHOLD = 0.95

# How many concrete events (and, per event, how many neighbours) to embed
# in each cluster node's export so the JSON is inspectable without going
# back to the database.
EVENT_SAMPLE_SIZE = 8
NEIGHBOUR_SAMPLE_SIZE = 5


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    denom = (np.linalg.norm(a) * np.linalg.norm(b))

    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


def load_clusters(con, concept):
    rows = con.execute(
        """
        SELECT concept, pub_year, cluster_id, centroid_vector, point_count
        FROM concept_year_cluster_info
        WHERE concept=? AND cluster_id >= 0
        ORDER BY pub_year, cluster_id
        """,
        (concept,),
    )

    clusters = []

    for row in rows:
        clusters.append( {
            "concept": row[0],
            "year": int(row[1]),
            "cluster": int(row[2]),
            "vector": blob_to_vector(row[3]),
            "size": int(row[4]),
        } )

    return clusters


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
        concept, year, cluster, size, centroid_vector = row
        node_id = f"{year}:{cluster}"
        G.add_node(
            node_id,
            concept=concept,
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
        SELECT source_year, source_cluster, target_year, target_cluster, similarity, edge_type, confidence
        FROM temporal_cluster_edges
        WHERE concept=?
        """,
        (concept,),
    )

    for row in edges:
        ( sy, sc, ty, tc, similarity, edge_type, confidence, ) = row

        source_id = f"{sy}:{sc}"
        target_id = f"{ty}:{tc}"

        # Guard against edges referencing clusters filtered out of the
        # node query above (e.g. noise clusters with cluster_id == -1).
        # Without this, add_edge silently creates a bare node with no
        # "year"/"vector" attrs, which later crashes assign_lineages.
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
            confidence=float(
                confidence
            ) if confidence is not None else 0.0,
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
    Assigns a lineage id to every node, and additionally tracks semantic
    persistence: whether a node's centroid is still close to the centroid
    that founded its lineage, independent of edge-level confidence.

    Structural continuity (a confident edge exists) is necessary but not
    sufficient for semantic persistence -- a chain of individually
    plausible hops can still drift the meaning of a cluster over time.
    So each node is checked against its lineage's *origin* vector, not
    just its immediate parent's.
    """

    lineage = {}
    merged_from = {}
    persistence_score = {}
    lineage_anchor = {}     # lineage_id -> founding node's vector
    lineage_founder = {}    # lineage_id -> founding node id
    next_lineage = 0

    nodes = sorted(
        G.nodes,
        key=lambda n: (
            G.nodes[n]["year"],
            G.nodes[n]["cluster"],
        )
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

        # Rank all candidate parents by structural confidence so we can
        # both pick the best one and record the rest as merge sources.
        ranked = sorted(
            incoming,
            key=lambda p: G.edges[p, node]["confidence"],
            reverse=True,
        )

        parent = ranked[0]
        confidence = G.edges[parent, node]["confidence"]

        parent_lineage = lineage[parent]
        anchor_vector = lineage_anchor.get(parent_lineage)

        if node_vector is not None and anchor_vector is not None:
            persistence = cosine_similarity(node_vector, anchor_vector)
        else:
            # No vector available -- fall back to structural confidence
            # only, since we can't evaluate semantic drift directly.
            persistence = confidence

        is_continuation = (
            confidence >= CONFIDENCE_THRESHOLD
            and persistence >= DRIFT_THRESHOLD
        )

        if not is_continuation:
            # Either the structural link is weak, or the cluster has
            # drifted far enough from its lineage's origin that it no
            # longer represents the same concept. Either way: new
            # lineage, re-anchored at this node.
            lineage[node] = next_lineage
            lineage_anchor[next_lineage] = node_vector
            lineage_founder[next_lineage] = node
            persistence_score[node] = 1.0
            next_lineage += 1
        else:
            lineage[node] = parent_lineage
            persistence_score[node] = persistence

        # Any other incoming edge represents a merge: an earlier lineage
        # folding into this node's lineage. Record it even though it
        # didn't "win" the parent selection, so the merge isn't silently
        # discarded.
        other_lineages = sorted(
            {
                lineage[p]
                for p in ranked[1:]
                if p in lineage and lineage[p] != lineage[node]
            }
        )

        if other_lineages:
            merged_from[node] = other_lineages

    nx.set_node_attributes(G, lineage, "lineage")
    nx.set_node_attributes(G, merged_from, "merged_from")
    nx.set_node_attributes(G, persistence_score, "persistence")

    # Per-lineage summary: the weakest persistence score seen anywhere in
    # the lineage's chain. A lineage that never dips below DRIFT_THRESHOLD
    # is "stable"; one that does dip is flagged even though every
    # individual edge cleared the confidence bar.
    lineage_min_persistence = {}

    for node, lid in lineage.items():
        score = persistence_score[node]
        lineage_min_persistence[lid] = min(
            score,
            lineage_min_persistence.get(lid, 1.0),
        )

    lineage_stability = {
        lid: (min_score >= DRIFT_THRESHOLD)
        for lid, min_score in lineage_min_persistence.items()
    }

    G.graph["lineage_min_persistence"] = lineage_min_persistence
    G.graph["lineage_stability"] = lineage_stability
    G.graph["lineage_founder"] = lineage_founder

    return G


def analyse_lineage(G):
    births = [
        n
        for n in G.nodes
        if G.in_degree(n) == 0
    ]

    deaths = [
        n
        for n in G.nodes
        if G.out_degree(n) == 0
    ]

    branching = [
        (n, G.out_degree(n))
        for n in G.nodes
        if G.out_degree(n) > 1
    ]

    merging = [
        (n, G.in_degree(n))
        for n in G.nodes
        if G.in_degree(n) > 1
    ]

    unstable = [
        lid
        for lid, stable in G.graph.get("lineage_stability", {}).items()
        if not stable
    ]

    logger.debug(f"[tier4] unstable lineages (drifted below {DRIFT_THRESHOLD}):")
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
):
    """
    Pulls a small deterministic sample of concrete events belonging to
    this (concept, year, cluster).

    Events remain individual historical observations. Neighbours are
    aggregated by lexical form for display so repeated contextual matches
    do not overwhelm the detail panel.

    The underlying event ids and positions are retained inside examples.
    """

    rows = con.execute(
        """
        SELECT e.event_id, e.doc_id, e.token_idx, e.token, e.pub_year
        FROM concept_year_event_cluster c
        JOIN events e ON e.event_id = c.event_id
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

    # Protect against duplicate joins producing repeated observations.
    seen_events = set()
    unique_rows = []

    for row in rows:
        event_id = row[0]

        if event_id in seen_events:
            continue

        seen_events.add(event_id)
        unique_rows.append(row)

    rows = unique_rows

    if len(rows) > event_limit:
        indices = sorted(
            set(
                int(round(i))
                for i in np.linspace(
                    0,
                    len(rows) - 1,
                    event_limit,
                )
            )
        )

        rows = [rows[i] for i in indices]

    samples = []

    for event_id, doc_id, token_idx, token, ev_year in rows:

        neighbour_rows = con.execute(
            """
            SELECT
                n.neighbour_event_id,
                e.token,
                e.doc_id,
                e.pub_year,
                e.token_idx,
                n.score,
                n.depth
            FROM neighbours n
            JOIN events e
                ON e.event_id = n.neighbour_event_id
            WHERE n.event_id=?
            ORDER BY n.score DESC
            LIMIT ?
            """,
            (
                event_id,
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
            depth,
        ) in neighbour_rows:

            grouped[neighbour_token.lower()].append(
                {
                    "neighbour_event_id": neighbour_event_id,
                    "doc_id": neighbour_doc_id,
                    "pub_year": neighbour_year,
                    "token_idx": neighbour_token_idx,
                    "score": score,
                    "depth": depth,
                }
            )

        neighbours = [
            {
                "token": neighbour_token,
                "count": len(events),
                "max_score": max(
                    e["score"]
                    for e in events
                ),
                "examples": events[:3],
            }
            for neighbour_token, events in grouped.items()
        ]

        # Rank semantic neighbours by recurrence first, similarity second.
        neighbours.sort(
            key=lambda n: (
                n["count"],
                n["max_score"],
            ),
            reverse=True,
        )

        neighbours = neighbours[:neighbour_limit]

        samples.append(
            {
                "event_id": event_id,
                "doc_id": doc_id,
                "token_idx": token_idx,
                "token": token,
                "pub_year": ev_year,
                "neighbours": neighbours,
            }
        )

    return samples


def export_lineage(con, concept, G, analysis=None,):
    nodes = []

    rows = con.execute(
        """
        SELECT concept, pub_year, cluster_id, point_count, centroid_nx, centroid_ny, centroid_gnx, centroid_gny
        FROM concept_year_cluster_info
        WHERE concept=?
        ORDER BY pub_year, cluster_id
        """,
        (concept,)
    )

    lineage_min_persistence = G.graph.get("lineage_min_persistence", {})
    lineage_stability = G.graph.get("lineage_stability", {})

    for row in rows:
        ( concept, year, cluster, size, nx, ny, gnx, gny, ) = row
        node_id = f"{year}:{cluster}"

        if node_id not in G:
            # e.g. noise cluster, filtered out of the graph entirely
            continue

        node_lineage = G.nodes[node_id].get("lineage")

        nodes.append({
            "id": f"{year}:{cluster}",
            "year": year,
            "cluster": cluster,
            "size": size,
            "lineage": node_lineage,
            "merged_from": G.nodes[node_id].get("merged_from", []),
            "persistence_score": G.nodes[node_id].get("persistence"),
            "lineage_stable": lineage_stability.get(node_lineage),
            "local": {
                "x": nx if nx is not None else 0,
                "y": ny if ny is not None else 0,
            },
            "global": {
                "x": gnx if gnx is not None else 0,
                "y": gny if gny is not None else 0,
            },
            "event_sample": sample_cluster_events(
                con, concept, year, cluster,
            ),
            "context_profile": aggregate_cluster_context(
                con, concept, year, cluster,
            ),
        })


    edges = []

    rows = con.execute(
        """
        SELECT source_year, source_cluster, target_year, target_cluster, similarity, confidence, edge_type
        FROM temporal_cluster_edges
        WHERE concept=?
        """,
        (concept,)
    )


    for row in rows:
        (
            sy,
            sc,
            ty,
            tc,
            sim,
            confidence,
            edge_type,
        ) = row

        edges.append({
            "source": f"{sy}:{sc}",
            "target": f"{ty}:{tc}",
            "similarity": sim,
            "confidence": confidence,
            "type": edge_type,
        })

    lineages_summary = [
        {
            "lineage": lid,
            "min_persistence": min_score,
            "stable": lineage_stability.get(lid),
        }
        for lid, min_score in lineage_min_persistence.items()
    ]

    return {
        "generated": "tier4_0_lineage_graph",
        "concept": concept,
        "nodes": nodes,
        "links": edges,
        "lineages": lineages_summary,
        "drift_threshold": DRIFT_THRESHOLD,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "summary": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "lineages": len(lineage_min_persistence),
            "stable_lineages": sum(lineage_stability.values()),
            "births": len(analysis["births"]) if analysis else 0,
            "deaths": len(analysis["deaths"]) if analysis else 0,
            "branching": len(analysis["branching"]) if analysis else 0,
            "merging": len(analysis["merging"]) if analysis else 0,
            "unstable_lineages": len(analysis["unstable"]) if analysis else 0,
        },
        "events": {
            "births": analysis["births"] if analysis else [],
            "deaths": analysis["deaths"] if analysis else [],
            "branching": analysis["branching"] if analysis else [],
            "merging": analysis["merging"] if analysis else [],
            "unstable": analysis["unstable"] if analysis else [],
        },
    }


def analyse_concept_lineage(
    con: Connection,
    concept: str,
) -> dict[str, object]:
    logger.info( f"[tier4] analysing {concept}" )

    G = load_temporal_graph( con, concept, )

    G = assign_lineages(G)
    analysis = analyse_lineage(G)

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
) -> dict[str, object]:
    started = time.perf_counter()
    result = analyse_concept_lineage( con, concept, )
    elapsed = time.perf_counter() - started

    result["elapsed_seconds"] = round( elapsed, 3, )

    if write_json:
        json_path = ( OUTPUT_DIR / f"{concept}_lineage.json" )
        with open( json_path, "w", encoding="utf8", ) as f:
            json.dump( result, f, indent=2, )
        result["json_path"] = str(json_path)
        logger.info(f"[tier4-service] wrote {json_path}")

    logger.info( f"[tier4-service] completed {concept} in {elapsed:.2f}s" )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--concept", )
    parser.add_argument( "--json", action="store_true", default=True)
    args = parser.parse_args()

    con = sqlite3.connect( CORPUS_TIER2_DB_PATH )

    try:
        concepts = (
            [args.concept.upper()]
            if args.concept
            else get_processed_concepts(
                CORPUS_TIER2_DB_PATH
            )
        )

        for concept in concepts:
            result = service( con=con, concept=concept, write_json=args.json, )
            logger.info( f"[tier4-main] {result['concept']} complete" )

    finally:
        con.close()


if __name__ == "__main__":
    main()

