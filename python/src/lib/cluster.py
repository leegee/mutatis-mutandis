# lib.clustering

from __future__ import annotations

import numpy as np
import umap
import igraph as ig
import leidenalg
from sklearn.neighbors import NearestNeighbors

from lib.eebo_logging import logger
from lib.eebo_config import (
    CORPUS_TIER2_DB_PATH,
    ZARR_PATH,
    faiss_index_paths,
)


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

    vectors = lookup.get_concatenated_embeddings( event_ids )

    return ( event_ids, vectors, )


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
            edges.append( (
                idx,
                int(neighbour),
            ) )

    return edges


def leiden_cluster( vectors, resolution_parameter=0.8, n_neighbors=15 ):
    if len(vectors) < MIN_IN_CLUSTER:
        return np.full( len(vectors), -1, dtype=np.int32, )

    edges = build_knn_graph( vectors, n_neighbors=n_neighbors )

    graph = ig.Graph(
        edges=edges,
        directed=False,
    )

    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        seed=42,
        resolution_parameter=resolution_parameter,
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


def compute_cluster_centroids(
    vectors,
    local_coords,
    global_coords,
    clusters,
):
    records = []

    for cluster_id in sorted(set(clusters)):
        mask = clusters == cluster_id

        if not np.any(mask):
            continue

        records.append( {
            "cluster_id": int(cluster_id),
            "centroid_vector": vectors[mask] .mean(axis=0) .astype(np.float32),
            "centroid_nx": float(local_coords[mask,0].mean()),
            "centroid_ny": float(local_coords[mask,1].mean()),
            "centroid_gnx": float(global_coords[mask,0].mean()),
            "centroid_gny": float(global_coords[mask,1].mean()),
            "point_count": int(mask.sum()),
        } )

    return records
