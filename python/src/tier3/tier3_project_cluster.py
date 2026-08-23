#!/usr/bin/env python

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
from lib.corpus_db import get_connection
from lib.corpus_logging import logger
from lib.sqlite_vector_blob import vector_to_blob
from lib.zarr_event_lookup import ZarrEventLookup

from tier2.persistence import (
    dump_pg_stage_to_sqlite,
    list_stage_concepts_pg,
    load_event_rows_pg,
    write_cluster_info_pg,
    write_geometry_pg,
)

YEAR_BUCKET = 10


def sqlite_connection(path: Path):
    con = sqlite3.connect(path)
    con.execute(
        "PRAGMA busy_timeout=5000"
    )
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
        rows.append(
            (
                float(local_coords[idx][0]),
                float(local_coords[idx][1]),
                float(global_coords[idx][0]),
                float(global_coords[idx][1]),
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
        mask = (
            clusters == cluster_id
        )

        if not np.any(mask):
            continue

        centroid_vector = (
            cluster_centroid_vectors.get(
                cluster_id
            )
        )

        if centroid_vector is None:
            continue

        data.append(
            (
                concept,
                int(cluster_id),
                (
                    "noise"
                    if cluster_id == -1
                    else None
                ),
                float(
                    local_coords[
                        mask,
                        0,
                    ].mean()
                ),
                float(
                    local_coords[
                        mask,
                        1,
                    ].mean()
                ),
                float(
                    global_coords[
                        mask,
                        0,
                    ].mean()
                ),
                float(
                    global_coords[
                        mask,
                        1,
                    ].mean()
                ),
                vector_to_blob(
                    centroid_vector
                ),
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
    lookup: ZarrEventLookup,
    concept: str,
    global_coords: dict[
        int,
        NDArray[np.float32],
    ],
    resolution_parameter: float,
    n_neighbors: int,
) -> dict[str, object]:
    logger.info(
        f"[tier3] processing {concept}"
    )

    rows = load_rows(
        concept
    )

    if not rows:
        logger.warning(
            f"[tier3] {concept}: no events"
        )

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
                    lookup.get_pos(
                        int(row[0])
                    )
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

    logger.info(
        f"[tier3] {concept}: "
        f"field events={len(event_ids):,}"
    )

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
        logger.info(
            f"[tier3] {concept}: sampled fit "
            f"({fit_info['fit_n']:,}/"
            f"{fit_info['n']:,} events, "
            f"{fit_info['outlier_n']:,} "
            f"guaranteed outliers)"
        )

    global_xy = np.asarray(
        [
            global_coords[event_id]
            for event_id in event_ids
        ],
        dtype=np.float32,
    )

    write_geometry(
        event_ids,
        local_coords,
        global_xy,
        clusters,
    )

    write_cluster_info(
        concept,
        cluster_centroid_vectors,
        local_coords,
        global_xy,
        clusters,
    )

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
    use_pg: bool,
    db_path=None,
):
    """
    Build shared Tier 3 resources.

    Tier 3 obtains semantic vectors exclusively through the Tier 1
    observation lookup.

    The physical vector backend is intentionally hidden behind that
    lookup. Tier 3 must not depend on FAISS, DiskANN, or any ANN index.
    """
    lookup = ZarrEventLookup(
        EVENTSTORE_T1_PATH
    )

    if use_pg:
        pg = get_connection()

        concepts = list_stage_concepts_pg(
            pg
        )

        load_rows = (
            lambda concept:
            load_event_rows_pg(
                pg,
                concept,
            )
        )

        all_field_event_ids = []
        strata = []

        for concept in concepts:
            rows = load_event_rows_pg(
                pg,
                concept,
            )

            for row in rows:
                event_id = int(
                    row[0]
                )

                if (
                    len(row) > 2
                    and row[2] is not None
                ):
                    year = int(row[2])
                else:
                    year = int(
                        lookup.pub_year[
                            lookup.get_pos(
                                event_id
                            )
                        ]
                    )

                all_field_event_ids.append(
                    event_id
                )

                strata.append(
                    (
                        concept,
                        year // YEAR_BUCKET,
                    )
                )

        seen = {}

        for event_id, stratum in zip(
            all_field_event_ids,
            strata,
        ):
            if event_id not in seen:
                seen[event_id] = stratum

        uniq_ids = list(
            seen.keys()
        )

        uniq_strata = [
            seen[event_id]
            for event_id in uniq_ids
        ]

        global_coords = build_global_projection(
            lookup,
            uniq_ids,
            strata=uniq_strata,
        )

        def write_geom(
            event_ids,
            local_coords,
            global_coords,
            clusters,
        ):
            write_geometry_pg(
                pg,
                event_ids,
                local_coords,
                global_coords,
                clusters,
            )

        def write_cinfo(
            concept,
            cluster_centroid_vectors,
            local_coords,
            global_coords,
            clusters,
        ):
            write_cluster_info_pg(
                pg,
                concept,
                cluster_centroid_vectors,
                local_coords,
                global_coords,
                clusters,
                vector_to_blob,
            )

        return {
            "backend": "pg",
            "pg": pg,
            "con": None,
            "lookup": lookup,
            "global_coords": global_coords,
            "concepts": concepts,
            "load_rows": load_rows,
            "write_geometry": write_geom,
            "write_cluster_info": write_cinfo,
            "commit": pg.commit,
        }

    path = (
        db_path
        or CORPUS_TIER2_DB_PATH
    )

    con = sqlite_connection(
        path
    )

    concepts = [
        concept
        for concept, _
        in resolve_concepts(
            concept=None
        )
    ]

    present = {
        row[0]
        for row in con.execute(
            "SELECT concept FROM concepts"
        )
    }

    concepts = [
        concept
        for concept in concepts
        if concept in present
    ] or sorted(present)

    all_field_event_ids = []
    strata = []

    for concept in concepts:
        rows = load_event_rows(
            con,
            concept,
        )

        for row in rows:
            event_id = int(
                row[0]
            )

            if (
                len(row) > 2
                and row[2] is not None
            ):
                year = int(row[2])
            else:
                year = int(
                    lookup.pub_year[
                        lookup.get_pos(
                            event_id
                        )
                    ]
                )

            all_field_event_ids.append(
                event_id
            )

            strata.append(
                (
                    concept,
                    year // YEAR_BUCKET,
                )
            )

    seen = {}

    for event_id, stratum in zip(
        all_field_event_ids,
        strata,
    ):
        if event_id not in seen:
            seen[event_id] = stratum

    uniq_ids = list(
        seen.keys()
    )

    uniq_strata = [
        seen[event_id]
        for event_id in uniq_ids
    ]

    global_coords = build_global_projection(
        lookup,
        uniq_ids,
        strata=uniq_strata,
    )

    return {
        "backend": "sqlite",
        "pg": None,
        "con": con,
        "lookup": lookup,
        "global_coords": global_coords,
        "concepts": concepts,
        "load_rows": (
            lambda concept:
            load_event_rows(
                con,
                concept,
            )
        ),
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

    logger.info(
        f"[tier3-service] processing {concept}"
    )

    report = cluster_concept(
        load_rows=resources["load_rows"],
        write_geometry=resources["write_geometry"],
        write_cluster_info=resources[
            "write_cluster_info"
        ],
        commit=resources["commit"],
        lookup=resources["lookup"],
        concept=concept,
        global_coords=resources[
            "global_coords"
        ],
        resolution_parameter=resolution_parameter,
        n_neighbors=n_neighbors,
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    logger.info(
        f"[tier3-service] completed "
        f"{concept} in {elapsed:.2f}s"
    )

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

    parser.add_argument(
        "--concept",
        default=None,
    )

    parser.add_argument(
        "--pg",
        action="store_true",
        help=(
            "Read/write Postgres tier2_stage "
            "(leave stage alive from Tier 2)"
        ),
    )

    parser.add_argument(
        "--publish-sqlite",
        default=None,
        help=(
            "After clustering, dump stage "
            "to this SQLite path (PG mode only)"
        ),
    )

    parser.add_argument(
        "-r",
        "--resolution",
        type=float,
        default=0.8,
        help=(
            "Leiden resolution parameter "
            "(default: 0.8)"
        ),
    )

    parser.add_argument(
        "-n",
        "--neighbors",
        type=int,
        default=15,
        help=(
            "kNN graph neighbours "
            "(default: 15)"
        ),
    )

    args = parser.parse_args()

    resources = build_tier3_resources(
        use_pg=args.pg
    )

    try:
        if args.concept:
            concepts = [
                args.concept.upper()
            ]
        else:
            concepts = resources[
                "concepts"
            ]

        if not concepts:
            logger.warning(
                "[tier3-main] "
                "no concepts resolved"
            )
            return

        logger.info(
            f"[tier3-main] "
            f"backend={resources['backend']} "
            f"concepts={len(concepts)}"
        )

        for concept in concepts:
            result = service(
                resources=resources,
                concept=concept,
                resolution_parameter=(
                    args.resolution
                ),
                n_neighbors=args.neighbors,
            )

            logger.info(
                "[tier3-main] completed "
                f"{result.get('concept')}"
            )

        if (
            args.pg
            and args.publish_sqlite
        ):
            resources["commit"]()

            dump_pg_stage_to_sqlite(
                resources["pg"],
                args.publish_sqlite,
                clear=True,
            )

            logger.info(
                "[tier3-main] published "
                f"SQLite → {args.publish_sqlite}"
            )

    finally:
        if resources.get("con") is not None:
            resources["con"].close()

        pg = resources.get("pg")

        if (
            pg is not None
            and hasattr(pg, "close")
        ):
            pg.close()

    logger.info(
        "[tier3-main] Done."
    )


if __name__ == "__main__":
    main()
    