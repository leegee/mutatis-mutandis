#!/usr/bin/env python

# tier3_0_project_cluster.py

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np

import umap
import igraph as ig
import leidenalg
from sklearn.neighbors import NearestNeighbors

from lib.eebo_config import (
    CORPUS_TIER2_DB_PATH,
    ZARR_PATH,
    faiss_index_paths,
)

from lib.concept_resolve import resolve_concepts
from lib.zarr_event_lookup import ZarrEventLookup
from lib.eebo_faiss import EeboFaissIndex
from lib.eebo_logging import logger

MIN_IN_CLUSTER = 7

LOCAL_UMAP_PARAMS = {
    "n_neighbors": 15,
    "min_dist": 0.05,
    "metric": "cosine",
}

GLOBAL_UMAP_PARAMS = {
    "n_neighbors": 50,
    "min_dist": 0.1,
    "metric": "cosine",
}



def load_indices(
    years,
    masked=False,
):
    index_paths = {
        year:
            faiss_index_paths(
                masked=masked,
                year=year,
            )
        for year in years
    }

    index = {}

    for year, paths in index_paths.items():
        index[year] = {}

        for scale, path in paths.items():
            index[year][scale] = EeboFaissIndex.load(
                path
            )

    return index


def sqlite_connection( path: Path, ):
    con = sqlite3.connect(path)
    con.execute(
        "PRAGMA busy_timeout=5000"
    )
    return con



def load_event_rows(con, concept):
    """
    Load the empirical semantic field for a concept.

    concept_field_events is the authoritative relation:
        concept -> observed corpus events

    Seed events anchor the field.
    Neighbour events are retrieved semantic context.

    An event may belong to multiple fields.
    """
    return con.execute(
        """
        SELECT e.event_id, e.vector_id
        FROM concept_field_events f
        JOIN events e ON e.event_id = f.event_id
        WHERE f.concept = ?
        ORDER BY e.event_id
        """,
        (concept,),
    ).fetchall()


def load_vectors( lookup, event_rows, ):
    event_ids = [
        int(row[0])
        for row in event_rows
    ]

    if not event_ids:
        return (
            [],
            np.empty(
                (0, 0),
                dtype=np.float32,
            ),
        )

    vectors = lookup.get_concatenated_embeddings(
        event_ids
    )

    return (
        event_ids,
        vectors,
    )


def project( vectors, params, ):
    if len(vectors) < MIN_IN_CLUSTER:
        return np.zeros( ( len(vectors), 2, ), dtype=np.float32, )
    reducer = umap.UMAP( random_state=42, **params, )
    return reducer.fit_transform( vectors )


def build_knn_graph( vectors, n_neighbors=15, ):
    if len(vectors) < 3:
        return []

    n_neighbors = min(
        n_neighbors,
        len(vectors) - 1,
    )

    if len(vectors) <= n_neighbors:
        n_neighbors = len(vectors) - 1

    if n_neighbors < 2:
        return []

    nn = NearestNeighbors(
        n_neighbors=n_neighbors + 1,
        metric="cosine",
    )

    nn.fit(vectors)

    _, indices = nn.kneighbors(
        vectors
    )

    edges = []

    for idx, neighbours in enumerate(indices):
        for neighbour in neighbours[1:]:
            edges.append(
                (
                    idx,
                    int(neighbour),
                )
            )

    return edges


def leiden_cluster( vectors, ):
    if len(vectors) < MIN_IN_CLUSTER:
        return np.full( len(vectors), -1, dtype=np.int32, )

    edges = build_knn_graph( vectors )

    graph = ig.Graph(
        edges=edges,
        directed=False,
    )

    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        seed=42,
        resolution_parameter=0.8,
    )

    labels = np.full(
        len(vectors),
        -1,
        dtype=np.int32,
    )

    for cluster_id, members in enumerate(partition):
        for member in members:
            labels[member] = cluster_id

    return labels


def write_geometry(
    con,
    event_ids,
    local_coords,
    global_coords,
    clusters,
):
    rows = []

    for idx, event_id in enumerate(event_ids):
        rows.append(
            (
                float(local_coords[idx][0]),
                float(local_coords[idx][1]),
                float(global_coords[idx][0]),
                float(global_coords[idx][1]),
                int(clusters[idx]),
                (
                    "noise"
                    if clusters[idx] == -1
                    else None
                ),
                event_id,
            )
        )

    con.executemany(
        """
        UPDATE events SET
            nx=?,
            ny=?,
            gnx=?,
            gny=?,
            cluster_id=?,
            cluster_label=?
        WHERE event_id=?
        """,
        rows,
    )


def write_cluster_info(con, concept, local_coords, global_coords, clusters):
    con.execute(
        "DELETE FROM concept_cluster_info WHERE concept = ?",
        (concept,),
    )

    cluster_ids = sorted(set(int(x) for x in clusters))

    data = []
    for cluster_id in cluster_ids:
        mask = (clusters == cluster_id)
        if not np.any(mask):
            continue

        data.append((
            concept,
            cluster_id,
            "noise" if cluster_id == -1 else None,
            float(local_coords[mask, 0].mean()),
            float(local_coords[mask, 1].mean()),
            float(global_coords[mask, 0].mean()),
            float(global_coords[mask, 1].mean()),
            int(mask.sum()),
            None,                    # description
        ))

    con.executemany(
        """
        INSERT INTO concept_cluster_info (
            concept, cluster_id, cluster_label,
            centroid_nx, centroid_ny, centroid_gnx, centroid_gny,
            point_count, description
        ) VALUES (?,?,?,?,?,?,?,?,?)
        """,
        data,
    )


def build_global_projection(
    lookup,
    all_field_event_ids,
):
    logger.info( f"[tier3] global projection events={len(all_field_event_ids):,}" )
    vectors = lookup.get_concatenated_embeddings( all_field_event_ids )
    coords = project( vectors, GLOBAL_UMAP_PARAMS, )
    return {
        int(event_id): coords[idx]
        for idx, event_id in enumerate(all_field_event_ids)
    }


def process_concept( con, lookup, concept, global_coords, ):
    logger.info( f"[tier3] processing {concept}" )

    rows = load_event_rows( con, concept, )

    if not rows:
        logger.warning( f"[tier3] {concept}: no events" )
        return

    event_ids, vectors = load_vectors( lookup, rows, )
    if len(event_ids) == 0:
        return

    logger.info( f"[tier3] {concept}: field events={len(event_ids):,}" )

    local_coords = project( vectors, LOCAL_UMAP_PARAMS, )
    clusters = leiden_cluster( vectors, )

    global_xy = np.asarray(
        [
            global_coords[eid]
            for eid in event_ids
        ],
        dtype=np.float32,
    )

    write_geometry( con, event_ids, local_coords, global_xy, clusters, )
    write_cluster_info( con, concept, local_coords, global_xy, clusters, )
    con.commit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--concept", default=None, )
    parser.add_argument( "--mask", action="store_true", )
    args = parser.parse_args()

    con = sqlite_connection( CORPUS_TIER2_DB_PATH )

    resolved_concepts = [
        concept
        for concept, _ in resolve_concepts( concept=args.concept )
    ]

    if not resolved_concepts:
        logger.warning( "[tier3] no concepts resolved" )
        return

    global_concepts = [
        concept
        for concept, _ in resolve_concepts(concept=None)
    ]

    logger.info( f"[tier3] concepts={len(resolved_concepts)}" )
    logger.info( f"[tier3] global concepts={len(global_concepts)}" )

    # Global lookup intentionally has no form restriction.
    # Global geometry must remain comparable between concepts.
    lookup = ZarrEventLookup( ZARR_PATH )

    years = sorted(
        set(
            int(y)
            for y in lookup.pub_year
            if y > 0
        )
    )

    index = load_indices( years, masked=args.mask, )
    lookup.attach_index( index )

    all_field_event_ids = []

    for concept in global_concepts:
        rows = load_event_rows( con, concept, )
        all_field_event_ids.extend(
            int(row[0])
            for row in rows
        )

    if not all_field_event_ids:
        logger.warning( "[tier3] no events found" )
        return

    # Defensive ordering guarantees:
    #   - deterministic UMAP input order
    #   - stable SQLite updates
    #   - reproducible plots
    all_field_event_ids = sorted( set(all_field_event_ids) )
    logger.info( f"[tier3] global projection events={len(all_field_event_ids):,}" )

    global_coords = build_global_projection( lookup, all_field_event_ids, )

    for concept in resolved_concepts:
        process_concept(
            con,
            lookup,
            concept,
            global_coords,
        )

    con.close()
    logger.info( f"[tier3] Done." )

if __name__ == "__main__":
    main()
