#!/usr/bin/env python
"""
tier3/tier3_0_project_cluster.py
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from lib.cluster import (
    LOCAL_UMAP_PARAMS,
    build_global_projection,
    load_event_rows,
    local_project_and_cluster,
)
from lib.concept_resolve import resolve_concepts
from lib.corpus_config import (
    CORPUS_TIER2_DB_PATH,
    EVENTSTORE_T1_PATH,
)
from lib.corpus_logging import logger
from lib.sqlite_vector_blob import vector_to_blob
from tier1.observation_store_api import open_observation_lookup

YEAR_BUCKET = 10


def sqlite_connection(path: Path):
    con = sqlite3.connect(path)
    con.execute("PRAGMA busy_timeout=5000")
    return con


def write_geometry_sqlite(
    con,
    event_ids,
    local_coords,
    global_coords,
    clusters,
):
    rows = []

    for idx, event_id in enumerate(event_ids):
        gx = global_coords[idx][0]
        gy = global_coords[idx][1]

        rows.append(
            (
                float(local_coords[idx][0]),
                float(local_coords[idx][1]),
                (
                    float(gx)
                    if np.isfinite(gx)
                    else None
                ),
                (
                    float(gy)
                    if np.isfinite(gy)
                    else None
                ),
                int(clusters[idx]),
                (
                    "noise"
                    if int(clusters[idx]) == -1
                    else None
                ),
                int(event_id),
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


def write_cluster_info_sqlite(
    con,
    concept,
    cluster_centroid_vectors,
    local_coords,
    global_coords,
    clusters,
):
    """
    cluster_centroid_vectors is a mapping of cluster_id to its
    streaming-computed mean embedding.

    The complete field embedding matrix is deliberately not required.
    """
    con.execute(
        """
        DELETE FROM concept_cluster_info
        WHERE concept = ?
        """,
        (concept,),
    )

    data = []

    for cluster_id in sorted(
        set(
            int(x)
            for x in clusters
        )
    ):
        mask = clusters == cluster_id

        if not np.any(mask):
            continue

        centroid_vector = cluster_centroid_vectors.get(
            cluster_id
        )

        if centroid_vector is None:
            continue

        gnx = None
        gny = None

        if global_coords is not None:
            gx = global_coords[mask, 0]
            gy = global_coords[mask, 1]

            finite = np.isfinite(gx) & np.isfinite(gy)

            if np.any(finite):
                gnx = float(gx[finite].mean())
                gny = float(gy[finite].mean())

        data.append(
            (
                concept,
                int(cluster_id),
                (
                    "noise"
                    if cluster_id == -1
                    else None
                ),
                float(local_coords[mask, 0].mean()),
                float(local_coords[mask, 1].mean()),
                gnx,
                gny,
                vector_to_blob(centroid_vector),
                int(mask.sum()),
                None,
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
    *,
    load_rows,
    write_geometry,
    write_cluster_info,
    commit,
    lookup,
    concept: str,
    global_coords: dict[
        int,
        NDArray[np.float32],
    ],
    resolution_parameter: float,
    n_neighbors: int,
) -> dict[str, object]:
    logger.info( f"[tier3] processing {concept}" )

    rows = load_rows( concept )
    if not rows:
        logger.warning( f"[tier3] {concept}: no events" )
        return {
            "concept": concept,
            "status": "no-op",
            "reason": "No events",
        }

    event_ids = [
        int(row[0])
        for row in rows
    ]

    strata = [
        (
            int(row[2]) // YEAR_BUCKET
            if len(row) > 2
            and row[2] is not None
            else int(
                lookup.pub_year[
                    lookup.get_pos( int(row[0]) )
                ]
            ) // YEAR_BUCKET
        )
        for row in rows
    ]

    if len(event_ids) == 0:
        return {
            "concept": concept,
            "status": "no-op",
            "reason": "No events",
        }

    logger.info( f"[tier3] {concept}: field events={len(event_ids):,}" )

    result = local_project_and_cluster(
        lookup,
        event_ids,
        strata=strata,
        umap_params=LOCAL_UMAP_PARAMS,
        resolution_parameter=resolution_parameter,
        n_neighbors=n_neighbors,
    )

    event_ids = result["event_ids"]
    local_coords = result["local_coords"]
    clusters = result["clusters"]
    cluster_centroid_vectors = (
        result["cluster_centroid_vectors"]
    )
    fit_info = result["fit_info"]

    if fit_info["sampled"]:
        logger.info( f"[tier3] {concept}: sampled fit ({fit_info['fit_n']:,}/ {fit_info['n']:,} events, {fit_info['outlier_n']:,} guaranteed outliers)" )

    # global_xy = np.asarray(
    #     [
    #         global_coords[event_id]
    #         for event_id in event_ids
    #     ],
    #     dtype=np.float32,
    # )

    if global_coords is None:
        global_xy = np.full(
            (len(event_ids), 2),
            np.nan,
            dtype=np.float32,
        )
    else:
        global_xy = np.asarray(
            [
                global_coords[event_id]
                for event_id in event_ids
            ],
            dtype=np.float32,
        )

    write_geometry( event_ids, local_coords, global_xy, clusters, )
    write_cluster_info( concept, cluster_centroid_vectors, local_coords, global_xy, clusters, )

    commit()

    return {
        "concept": concept,
        "status": "complete",
        "events": len(event_ids),
        "clusters": len(
            {
                int(cluster)
                for cluster in clusters
                if cluster != -1
            }
        ),
        "noise_points": int(
            np.sum(
                clusters == -1
            )
        ),
        "sampled": fit_info["sampled"],
        "fit_events": fit_info["fit_n"],
        "outlier_events": fit_info["outlier_n"],
    }


def build_tier3_resources(
    *,
    store_path=None,
    db_path=None,
):
    """
    Build shared Tier 3 resources.

    Tier 1 Parquet is the source of truth for observation identity,
    provenance, publication year, and embeddings.

    Tier 3 does not depend on Zarr, FAISS, DiskANN, or Postgres for
    observation access.

    SQLite remains the disposable Tier 3 result store because the
    downstream schema and consumers still expect the existing
    events and concept_cluster_info tables.
    """
    store_path = Path( store_path or EVENTSTORE_T1_PATH )
    db_path    = Path( db_path or CORPUS_TIER2_DB_PATH )
    lookup     = open_observation_lookup( store_path )
    con        = sqlite_connection( db_path )

    concepts = [
        concept
        for concept, _
        in resolve_concepts(
            concept=None
        )
    ]

    present = {
        row[0]
        for row in con.execute( "SELECT concept FROM concepts" )
    }

    concepts = [
        concept
        for concept in concepts
        if concept in present
    ] or sorted(present)

    def load_rows_for_concept( concept ):
        return load_event_rows( con, concept )

    # Global projection is disabled because its cost is currently
    # disproportionate to its value. Do not scan the event field merely
    # to construct inputs for a projection that is not being built.
    global_coords = None

    return {
        "backend": "parquet+lance",
        "lookup": lookup,
        "con": con,
        "global_coords": global_coords,
        "concepts": concepts,
        "load_rows": load_rows_for_concept,
        "write_geometry": (
            lambda event_ids,
            local_coords,
            global_coords,
            clusters:
            write_geometry_sqlite(
                con,
                event_ids,
                local_coords,
                global_coords,
                clusters,
            )
        ),
        "write_cluster_info": (
            lambda concept,
            cluster_centroid_vectors,
            local_coords,
            global_coords,
            clusters:
            write_cluster_info_sqlite(
                con,
                concept,
                cluster_centroid_vectors,
                local_coords,
                global_coords,
                clusters,
            )
        ),
        "commit": con.commit,
    }


def service(
    *,
    resources: dict,
    concept: str,
    resolution_parameter: float = 0.8,
    n_neighbors: int = 15,
) -> dict[str, object]:
    started = time.perf_counter()

    logger.info( f"[tier3-service] processing {concept}" )

    report = cluster_concept(
        load_rows=resources["load_rows"],
        write_geometry=resources["write_geometry"],
        write_cluster_info=resources[ "write_cluster_info" ],
        commit=resources["commit"],
        lookup=resources["lookup"],
        concept=concept,
        global_coords=resources[ "global_coords" ],
        resolution_parameter=resolution_parameter,
        n_neighbors=n_neighbors,
    )

    elapsed = ( time.perf_counter() - started )

    logger.info( f"[tier3-service] completed " f"{concept} in {elapsed:.2f}s" )

    return {
        **report,
        "resolution_parameter": (
            resolution_parameter
        ),
        "n_neighbors": n_neighbors,
        "elapsed_seconds": round(
            elapsed,
            3,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--concept", default=None, )
    parser.add_argument( "--store", type=Path, default=EVENTSTORE_T1_PATH, help=( "Tier 1 Parquet observation-store root" ), )
    parser.add_argument( "--db", type=Path, default=CORPUS_TIER2_DB_PATH, help=( "Tier 2 SQLite result database" ), )
    parser.add_argument( "-r", "--resolution", type=float, default=0.8, help=( "Leiden resolution parameter (default: 0.8)" ), )
    parser.add_argument( "-n", "--neighbors", type=int, default=15, help=( "kNN graph neighbours (default: 15)" ), )
    args = parser.parse_args()

    resources = build_tier3_resources( store_path=args.store, db_path=args.db )

    try:
        if args.concept:
            concepts = [ args.concept.upper() ]
        else:
            concepts = resources[ "concepts" ]

        if not concepts:
            logger.warning( "[tier3-main] no concepts resolved" )
            return

        logger.info( f"[tier3-main] backend={resources['backend']} concepts={len(concepts)}" )

        for concept in concepts:
            result = service(
                resources=resources,
                concept=concept,
                resolution_parameter=( args.resolution ),
                n_neighbors=args.neighbors,
            )

            logger.info( f"[tier3-main] completed {result.get('concept')}" )

    finally:
        con = resources.get("con")

        if con is not None:
            con.close()

        lookup = resources.get("lookup")

        lookup_con = getattr( lookup, "_con", None )

        if lookup_con is not None:
            lookup_con.close()

    logger.info( "[tier3-main] Done." )


if __name__ == "__main__":
    main()
