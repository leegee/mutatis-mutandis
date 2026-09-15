#!/usr/bin/env python
"""
tier3/tier3_0_project_cluster.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from lib.cluster import (
    LOCAL_UMAP_PARAMS,
    local_project_and_cluster,
)
from lib.concept_resolve import resolve_concepts
from lib.corpus_config import (
    CORPUS_MAX_YEAR,
    CORPUS_MIN_YEAR,
    LANCE_INDEXES_DIR,
)
from lib.corpus_db import get_connection
from lib.corpus_logging import logger
from lib.sqlite_vector_blob import vector_to_blob
from retrieval.lance_observation_index_store import (
    LanceObservationIndexStore,
)
from retrieval.models import SearchSpace

YEAR_BUCKET = 10
CLUSTER_SCALE = "local"


def load_concepts(con):
    with con.cursor() as cur:
        cur.execute(
            """
            SELECT concept
            FROM tier2.concepts
            ORDER BY concept
            """
        )
        present = {
            str(row[0])
            for row in cur.fetchall()
        }

    resolved = [
        concept
        for concept, _
        in resolve_concepts(concept=None)
    ]

    concepts = [
        concept
        for concept in resolved
        if concept in present
    ]

    return concepts or sorted(present)


def load_event_rows(con, concept: str) -> list[tuple[int, int]]:
    """
    Tier 2 event_field is the complete persisted field population.

    It contains both seed events and retrieved neighbour events, with one
    row per concept/event pair.
    """
    with con.cursor() as cur:
        cur.execute(
            """
            SELECT
                ef.event_id,
                e.pub_year
            FROM tier2.event_field ef
            JOIN events e
              ON e.event_id = ef.event_id
            WHERE ef.concept = %s
            ORDER BY ef.event_id
            """,
            (concept,),
        )

        rows = cur.fetchall()

    return [
        (int(event_id), int(pub_year))
        for event_id, pub_year in rows
        if pub_year is not None
    ]


def write_geometry_postgres(
    con,
    concept: str,
    event_ids,
    local_coords,
    global_coords,
    clusters,
) -> None:
    rows = []

    for idx, event_id in enumerate(event_ids):
        gx = float(global_coords[idx][0])
        gy = float(global_coords[idx][1])

        rows.append(
            (
                concept,
                int(event_id),
                float(local_coords[idx][0]),
                float(local_coords[idx][1]),
                gx if np.isfinite(gx) else None,
                gy if np.isfinite(gy) else None,
                int(clusters[idx]),
                "noise" if int(clusters[idx]) == -1 else None,
            )
        )

    if not rows:
        return

    with con.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO tier3.event_geometry (
                concept,
                event_id,
                nx,
                ny,
                gnx,
                gny,
                cluster_id,
                cluster_label
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (concept, event_id)
            DO UPDATE SET
                nx = EXCLUDED.nx,
                ny = EXCLUDED.ny,
                gnx = EXCLUDED.gnx,
                gny = EXCLUDED.gny,
                cluster_id = EXCLUDED.cluster_id,
                cluster_label = EXCLUDED.cluster_label
            """,
            rows,
        )


def write_cluster_info_postgres(
    con,
    concept: str,
    cluster_centroid_vectors,
    local_coords,
    global_coords,
    clusters,
) -> None:
    """
    Persist cluster summaries for one concept.

    Centroid vectors remain in the database so cluster identity can later be
    compared in embedding space without depending on a particular 2-D
    projection.
    """
    cluster_ids = sorted(
        set(int(x) for x in clusters)
    )

    rows = []

    for cluster_id in cluster_ids:
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

        gx = global_coords[mask, 0]
        gy = global_coords[mask, 1]

        finite = np.isfinite(gx) & np.isfinite(gy)

        if np.any(finite):
            gnx = float(gx[finite].mean())
            gny = float(gy[finite].mean())

        rows.append(
            (
                concept,
                cluster_id,
                "noise" if cluster_id == -1 else None,
                float(local_coords[mask, 0].mean()),
                float(local_coords[mask, 1].mean()),
                gnx,
                gny,
                vector_to_blob(centroid_vector),
                int(mask.sum()),
                None,
            )
        )

    with con.cursor() as cur:
        cur.execute(
            """
            DELETE FROM tier3.concept_cluster_info
            WHERE concept = %s
            """,
            (concept,),
        )

        if rows:
            cur.executemany(
                """
                INSERT INTO tier3.concept_cluster_info (
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
                VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s
                )
                """,
                rows,
            )


def cluster_concept(
    *,
    con,
    index,
    concept: str,
    resolution_parameter: float,
    n_neighbors: int,
) -> dict[str, object]:
    logger.info(
        "[tier3] processing %s",
        concept,
    )

    rows = load_event_rows(
        con,
        concept,
    )

    if not rows:
        logger.warning(
            "[tier3] %s: no events",
            concept,
        )
        return {
            "concept": concept,
            "status": "no-op",
            "reason": "No events",
        }

    event_ids = [
        event_id
        for event_id, _
        in rows
    ]

    pub_years = {
        event_id: pub_year
        for event_id, pub_year
        in rows
    }

    strata = [
        pub_years[event_id] // YEAR_BUCKET
        for event_id in event_ids
    ]

    logger.info(
        "[tier3] %s: field events=%s",
        concept,
        f"{len(event_ids):,}",
    )

    result = local_project_and_cluster(
        index,
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
            "[tier3] %s: sampled fit (%s/%s events, %s "
            "guaranteed outliers)",
            concept,
            f"{fit_info['fit_n']:,}",
            f"{fit_info['n']:,}",
            f"{fit_info['outlier_n']:,}",
        )

    global_xy = np.full(
        (len(event_ids), 2),
        np.nan,
        dtype=np.float32,
    )

    with con.transaction():
        con.cursor().execute(
            """
            DELETE FROM tier3.event_geometry
            WHERE concept = %s
            """,
            (concept,),
        )

        write_geometry_postgres(
            con,
            concept,
            event_ids,
            local_coords,
            global_xy,
            clusters,
        )

        write_cluster_info_postgres(
            con,
            concept,
            cluster_centroid_vectors,
            local_coords,
            global_xy,
            clusters,
        )

    return {
        "concept": concept,
        "status": "complete",
        "events": len(event_ids),
        "clusters": len(
            {
                int(cluster)
                for cluster in clusters
                if int(cluster) != -1
            }
        ),
        "noise_points": int(
            np.sum(clusters == -1)
        ),
        "sampled": fit_info["sampled"],
        "fit_events": fit_info["fit_n"],
        "outlier_events": fit_info["outlier_n"],
    }


def build_tier3_resources(
    *,
    lance_root: str | Path = LANCE_INDEXES_DIR,
):
    con = get_connection()

    lance_store = LanceObservationIndexStore(
        lance_root,
        available_years=range(
            CORPUS_MIN_YEAR,
            CORPUS_MAX_YEAR + 1,
        ),
        available_scales=(CLUSTER_SCALE,),
    )

    indexes = lance_store.get(
        SearchSpace(
            years=(
                CORPUS_MIN_YEAR,
                CORPUS_MAX_YEAR,
            ),
            scale=(CLUSTER_SCALE,),
        )
    )

    if CLUSTER_SCALE not in indexes:
        raise RuntimeError(
            f"Missing Lance scale: {CLUSTER_SCALE}"
        )

    concepts = load_concepts(con)

    return {
        "backend": "postgres+lance",
        "con": con,
        "index": indexes[CLUSTER_SCALE],
        "concepts": concepts,
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
        "[tier3-service] processing %s",
        concept,
    )

    result = cluster_concept(
        con=resources["con"],
        index=resources["index"],
        concept=concept,
        resolution_parameter=resolution_parameter,
        n_neighbors=n_neighbors,
    )

    elapsed = time.perf_counter() - started

    logger.info(
        "[tier3-service] completed %s in %.2fs",
        concept,
        elapsed,
    )

    return {
        **result,
        "resolution_parameter": resolution_parameter,
        "n_neighbors": n_neighbors,
        "elapsed_seconds": round(elapsed, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--concept",
        default=None,
    )

    parser.add_argument(
        "--lance-root",
        type=Path,
        default=LANCE_INDEXES_DIR,
    )

    parser.add_argument(
        "-r",
        "--resolution",
        type=float,
        default=0.8,
        help="Leiden resolution parameter (default: 0.8)",
    )

    parser.add_argument(
        "-n",
        "--neighbors",
        type=int,
        default=15,
        help="kNN graph neighbours (default: 15)",
    )

    args = parser.parse_args()

    resources = build_tier3_resources(
        lance_root=args.lance_root,
    )

    try:
        if args.concept:
            concepts = [args.concept.upper()]
        else:
            concepts = resources["concepts"]

        if not concepts:
            logger.warning(
                "[tier3-main] no concepts resolved"
            )
            return

        logger.info(
            "[tier3-main] backend=%s concepts=%s",
            resources["backend"],
            len(concepts),
        )

        for concept in concepts:
            result = service(
                resources=resources,
                concept=concept,
                resolution_parameter=args.resolution,
                n_neighbors=args.neighbors,
            )

            logger.info(
                "[tier3-main] %s: events=%s clusters=%s "
                "noise=%s sampled=%s",
                result.get("concept"),
                f"{result.get('events', 0):,}",
                result.get("clusters"),
                f"{result.get('noise_points', 0):,}",
                result.get("sampled"),
            )

    finally:
        resources["con"].close()

    logger.info("[tier3-main] Done.")


if __name__ == "__main__":
    main()