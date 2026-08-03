#!/usr/bin/env python

# tier3_0_project_cluster.py

from __future__ import annotations
from typing import Tuple

import argparse
import sqlite3
from sqlite3 import Connection
from pathlib import Path
import time

import numpy as np
from numpy.typing import NDArray

from lib.eebo_config import (
    CORPUS_TIER2_DB_PATH,
    ZARR_PATH,
    faiss_index_paths,
)

from lib.concept_resolve import resolve_concepts
from lib.zarr_event_lookup import ZarrEventLookup
from lib.eebo_faiss import EeboFaissIndex
from lib.eebo_logging import logger
from lib.sqlite_vector_blob import vector_to_blob
from lib.cluster import (
    LOCAL_UMAP_PARAMS,
    load_event_rows,
    load_vectors,
    project,
    leiden_cluster,
    build_global_projection,
)


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


def write_cluster_info(
    con,
    concept,
    vectors,
    local_coords,
    global_coords,
    clusters,
):
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

        centroid_vector = (
            vectors[mask]
            .mean(axis=0)
            .astype(np.float32)
        )

        data.append(
            (
                concept,
                cluster_id,
                "noise" if cluster_id == -1 else None,

                float(local_coords[mask, 0].mean()),
                float(local_coords[mask, 1].mean()),

                float(global_coords[mask, 0].mean()),
                float(global_coords[mask, 1].mean()),

                vector_to_blob(centroid_vector),

                int(mask.sum()),

                None,   # description
            )
        )

    con.executemany(
        """
        INSERT INTO concept_cluster_info (
            concept,
            cluster_id,
            cluster_label,

            centroid_nx,
            centroid_ny,

            centroid_gnx,
            centroid_gny,

            centroid_vector,

            point_count,

            description
        )
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        data,
    )


def cluster_concept(
    con: Connection,
    lookup: ZarrEventLookup,
    concept: str,
    global_coords: dict[int, NDArray[np.float32]],
    resolution_parameter: float,
    n_neighbors: int,
) -> dict[str, object]:

    logger.info( f"[tier3] processing {concept}" )
    rows = load_event_rows( con, concept, )

    if not rows:
        logger.warning( f"[tier3] {concept}: no events" )
        return {
            "concept": concept,
            "status": "no-op",
            "reason": "No events",
        }

    event_ids, vectors = load_vectors( lookup, rows, )
    if len(event_ids) == 0:
        return {
            "concept": concept,
            "status": "no-op",
            "reason": "No events",
        }


    logger.info( f"[tier3] {concept}: field events={len(event_ids):,}" )

    local_coords = project( vectors, LOCAL_UMAP_PARAMS, )
    clusters = leiden_cluster(
        vectors,
        resolution_parameter=resolution_parameter,
        n_neighbors=n_neighbors,
    )

    global_xy = np.asarray(
        [
            global_coords[eid]
            for eid in event_ids
        ],
        dtype=np.float32,
    )

    write_geometry( con, event_ids, local_coords, global_xy, clusters, )
    write_cluster_info( con, concept, vectors, local_coords, global_xy, clusters, )
    con.commit()
    return {
        "concept": concept,
        "status": "complete","c"
        "events": len(event_ids),
        "clusters": len({
            int(c)
            for c in clusters
            if c != -1
        } ),
        "noise_points": int( np.sum(clusters == -1) )
    }


def build_tier3_resources(
    *,
    masked=False,
) -> tuple[
    Connection,
    ZarrEventLookup,
    dict[int, np.ndarray],
]:
    con = sqlite_connection( CORPUS_TIER2_DB_PATH )
    lookup = ZarrEventLookup( ZARR_PATH )

    years = sorted(
        set(
            int(y)
            for y in lookup.pub_year
            if y > 0
        )
    )

    index = load_indices(
        years,
        masked=masked,
    )

    lookup.attach_index(index)

    global_concepts = [
        concept
        for concept, _ in resolve_concepts(concept=None)
    ]

    all_field_event_ids = []

    for concept in global_concepts:
        rows = load_event_rows(
            con,
            concept,
        )
        all_field_event_ids.extend(
            int(row[0])
            for row in rows
        )

    global_coords = build_global_projection(
        lookup,
        sorted(set(all_field_event_ids)),
    )
    return (
        con,
        lookup,
        global_coords,
    )


def service(
    *,
    con: Connection,
    lookup: ZarrEventLookup,
    global_coords: dict[int, NDArray[np.float32]],
    concept: str,
    resolution_parameter: float = 0.8,
    n_neighbors: int = 15,
) -> dict[str, object]:
    """
    Cluster one concept semantic field.

    Resources are expected to be pre-built and reused:
        - SQLite connection
        - Zarr event lookup
        - global projection coordinates

    Returns summary metadata describing the clustering operation.
    """
    started = time.perf_counter()
    logger.info( f"[tier3-service] processing {concept}" )

    report = cluster_concept(
        con,
        lookup,
        concept,
        global_coords,
        resolution_parameter,
        n_neighbors,
    )

    elapsed = time.perf_counter() - started
    logger.info( f"[tier3-service] completed {concept} in {elapsed:.2f}s" )

    return {
        **report,
        "resolution_parameter": resolution_parameter,
        "n_neighbors": n_neighbors,
        "elapsed_seconds": round(elapsed, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--concept", default=None, )
    parser.add_argument( "--mask", action="store_true", )
    parser.add_argument( "-r", "--resolution", type=float, default=0.8, help="Leiden resolution parameter (default: 0.8)", )
    parser.add_argument( "-n", "--neighbors", type=int, default=15, help="kNN graph neighbours (default: 15)", )
    args = parser.parse_args()

    try:
        con, lookup, global_coords = build_tier3_resources( masked=args.mask, )

        resolved_concepts = [
            concept
            for concept, _ in resolve_concepts( concept=args.concept )
        ]

        if not resolved_concepts:
            logger.warning( "[tier3-main] no concepts resolved" )
            return

        logger.info( f"[tier3-main] concepts={len(resolved_concepts)}" )

        for concept in resolved_concepts:
            result = service(
                con                  = con,
                lookup               = lookup,
                global_coords        = global_coords,
                concept              = concept,
                resolution_parameter = args.resolution,
                n_neighbors          = args.neighbors,
            )
            logger.info( f"[tier3-main] completed {result['concept']}" )

    finally:
        con.close()

    logger.info( "[tier3-main] Done." )


if __name__ == "__main__":
    main()
