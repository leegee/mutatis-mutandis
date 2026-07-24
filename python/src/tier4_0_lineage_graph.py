#!/usr/bin/env python

"""
tier4_0_lineage_graph.py
"""

from __future__ import annotations

import argparse
import sqlite3

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity

from lib.eebo_config import CORPUS_TIER2_DB_PATH
from lib.sqlite_vector_blob import blob_to_vector
from lib.eebo_logging import logger


SIMILARITY_THRESHOLD = 0.80


def load_clusters(con, concept):
    rows = con.execute(
        """
        SELECT concept, pub_year, cluster_id, centroid_vector, point_count
        FROM concept_year_cluster_info
        WHERE concept=?
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
        SELECT concept, pub_year, cluster_id, point_count
        FROM concept_year_cluster_info
        WHERE concept=?
        ORDER BY pub_year, cluster_id
        """,
        (concept,),
    )

    for row in nodes:
        concept, year, cluster, size = row
        node_id = f"{year}:{cluster}"
        G.add_node(
            node_id,
            concept=concept,
            year=int(year),
            cluster=int(cluster),
            size=int(size),
        )


    edges = con.execute(
        """
        SELECT source_year, source_cluster, target_year, target_cluster, similarity, edge_type
        FROM temporal_cluster_edges
        WHERE concept=?
        """,
        (concept,),
    )

    for row in edges:
        (
            sy,
            sc,
            ty,
            tc,
            similarity,
            edge_type,
        ) = row

        G.add_edge(
            f"{sy}:{sc}",
            f"{ty}:{tc}",
            similarity=float(similarity),
            edge_type=edge_type,
        )

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


    # nodes
    xs=[]
    ys=[]
    sizes=[]

    for n in G.nodes:
        x,y = pos[n]
        xs.append(x)
        ys.append(y)
        sizes.append( max( 50, G.nodes[n]["size"] ) )

    ax.scatter( xs, ys, s=sizes, )

    for n,(x,y) in pos.items():
        ax.text( x, y, n, fontsize=7, )

    ax.set_xlabel("Publication year")
    ax.set_ylabel("Cluster lineage")

    ax.grid(True)

    plt.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight"
    )


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--concept", required=True, )
    parser.add_argument( "--graphml", default="lineage.graphml" )
    parser.add_argument( "--png", default="lineage.png", )
    args = parser.parse_args()

    con = sqlite3.connect( CORPUS_TIER2_DB_PATH )

    logger.debug("[tier4] Load graph")
    G = load_temporal_graph(con, args.concept.upper())
    logger.debug("[tier4] Loaded graph")

    logger.info( f"{G.number_of_nodes()} nodes" )
    logger.info( f"{G.number_of_edges()} edges" )

    nx.write_graphml( G, args.graphml, )

    draw_temporal_graph( G, args.png, )

    logger.info( f"Wrote {args.graphml}" )
    logger.info( f"Wrote {args.png}" )


if __name__ == "__main__":
    main()
