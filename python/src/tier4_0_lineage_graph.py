#!/usr/bin/env python

"""
tier4_0_lineage_graph.py
"""

from __future__ import annotations

import argparse
import sqlite3
import json

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from lib.eebo_config import CORPUS_TIER2_DB_PATH, GUI_PUBLIC_DIR
from lib.sqlite_vector_blob import blob_to_vector
from lib.eebo_logging import logger

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
        key=lambda n: G.nodes[n]["year"]
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


def draw_temporal_graph(G, output_png):
    years = sorted(
        {
            G.nodes[n]["year"]
            for n in G.nodes
        }
    )

    pos = {}

    for year in years:
        nodes = [
            n for n in G.nodes
            if G.nodes[n]["year"] == year
        ]

        for idx, node in enumerate(nodes):
            pos[node] = ( year, idx )

    fig, ax = plt.subplots(figsize=(20,12))


    # edges first
    for u,v,data in G.edges(data=True):
        x1,y1 = pos[u]
        x2,y2 = pos[v]

        ax.plot(
            [x1,x2],
            [y1,y2],
            linewidth=max( 0.5, data["similarity"] * 3 ),
            alpha=0.5,
        )


    # nodes -- colour by persistence so drift is visible at a glance
    xs=[]
    ys=[]
    sizes=[]
    colours=[]

    for n in G.nodes:
        x,y = pos[n]
        xs.append(x)
        ys.append(y)
        sizes.append( max( 50, G.nodes[n]["size"] ) )
        colours.append( G.nodes[n].get("persistence", 1.0) )

    scatter = ax.scatter(
        xs, ys, s=sizes, c=colours, cmap="RdYlGn", vmin=0.0, vmax=1.0,
    )
    plt.colorbar(scatter, ax=ax, label="semantic persistence")

    for n,(x,y) in pos.items():
        ax.text( x, y, n, fontsize=7, )

    ax.set_xlabel("Publication year")
    ax.set_ylabel("Cluster lineage")

    ax.grid(True)

    plt.savefig( output_png, dpi=300, bbox_inches="tight" )


def analyse_lineage(G):
    logger.info("nodes:", G.number_of_nodes())
    logger.info("edges:", G.number_of_edges())

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

    logger.info("births:", len(births))
    logger.info("deaths:", len(deaths))

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

    logger.info("branching:")
    for x in branching[:20]:
        logger.info(x)

    logger.info("merging:")
    for x in merging[:20]:
        logger.info(x)

    unstable = [
        lid
        for lid, stable in G.graph.get("lineage_stability", {}).items()
        if not stable
    ]

    logger.info(f"unstable lineages (drifted below {DRIFT_THRESHOLD}):")
    logger.info(unstable)


def sample_cluster_events(
    con,
    concept,
    year,
    cluster_id,
    event_limit=EVENT_SAMPLE_SIZE,
    neighbour_limit=NEIGHBOUR_SAMPLE_SIZE,
):
    """
    Pulls a small, deterministic sample of concrete events belonging to
    this (concept, year, cluster) node -- doc_id/token_idx pairs the
    reader can look up directly in the source text -- and, for each
    sampled event, its top neighbours.

    Sampling is spread evenly across event_id order (rather than the
    first N or a random draw) so the sample isn't biased toward
    whichever documents happened to be indexed first, and stays stable
    across re-runs.
    """

    rows = con.execute(
        """
        SELECT event_id, doc_id, token_idx, token, pub_year
        FROM events
        WHERE concept=? AND pub_year=? AND cluster_id=?
        ORDER BY event_id
        """,
        (concept, year, cluster_id),
    ).fetchall()

    if not rows:
        return []

    if len(rows) > event_limit:
        indices = sorted(
            set(
                int(round(i))
                for i in np.linspace(0, len(rows) - 1, event_limit)
            )
        )
        rows = [rows[i] for i in indices]

    samples = []

    for event_id, doc_id, token_idx, token, ev_year in rows:
        neighbour_rows = con.execute(
            """
            SELECT neighbour_event_id, token, doc_id, pub_year, token_idx, score, depth
            FROM neighbours
            WHERE event_id=?
            ORDER BY score DESC
            LIMIT ?
            """,
            (event_id, neighbour_limit),
        ).fetchall()

        neighbours = [
            {
                "neighbour_event_id": n_event_id,
                "token": n_token,
                "doc_id": n_doc_id,
                "pub_year": n_pub_year,
                "token_idx": n_token_idx,
                "score": n_score,
                "depth": depth,
            }
            for (
                n_event_id,
                n_token,
                n_doc_id,
                n_pub_year,
                n_token_idx,
                n_score,
                depth,
            ) in neighbour_rows
        ]

        samples.append({
            "event_id": event_id,
            "doc_id": doc_id,
            "token_idx": token_idx,
            "token": token,
            "pub_year": ev_year,
            "neighbours": neighbours,
        })

    return samples


def export_lineage(con, concept, G):
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
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--concept", required=True, )
    args = parser.parse_args()

    png_path     = OUTPUT_DIR / f"{args.concept.upper()}_lineage.png"
    json_path    = OUTPUT_DIR / f"{args.concept.upper()}_lineage.json"

    con = sqlite3.connect( CORPUS_TIER2_DB_PATH )

    logger.debug("[tier4] Load graph")
    G = load_temporal_graph(con, args.concept.upper())
    logger.debug("[tier4] Loaded graph")

    G = assign_lineages(G)
    logger.debug("[tier4] Assigned lineage")


    from collections import Counter

    lineage_counts = Counter(
        nx.get_node_attributes(G, "lineage").values()
    )

    print("Lineages:", len(lineage_counts))

    for lineage, count in lineage_counts.most_common(20):
        print(
            "lineage",
            lineage,
            "nodes",
            count,
            "min_persistence",
            G.graph["lineage_min_persistence"].get(lineage),
            "stable",
            G.graph["lineage_stability"].get(lineage),
        )

    scores = [
        data["confidence"]
        for _,_,data in G.edges(data=True)
    ]

    if scores:
        print(
            np.percentile(
                scores,
                [0,25,50,75,90,95,99,100]
            )
        )

    analyse_lineage(G)

    logger.info( f"{G.number_of_nodes()} nodes" )
    logger.info( f"{G.number_of_edges()} edges" )

    draw_temporal_graph( G, png_path )
    logger.info( f"Wrote {png_path}" )

    json_data = export_lineage(con, args.concept.upper(), G)
    con.close()

    json.dump(
        json_data,
        open(json_path,"w"),
        indent=2
    )
    logger.info( f"Wrote {json_path}" )


if __name__ == "__main__":
    main()