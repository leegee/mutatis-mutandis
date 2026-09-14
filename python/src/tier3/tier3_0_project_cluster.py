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
    load_event_rows,
    local_project_and_cluster,
)
from lib.concept_resolve import resolve_concepts
from lib.corpus_config import (
    CORPUS_MAX_YEAR,
    CORPUS_MIN_YEAR,
    CORPUS_TIER2_DB_PATH,
    CORPUS_TIER3_DB_PATH,
    LANCE_INDEXES_DIR,
)
from lib.corpus_db import analysis_db_connection, get_connection
from lib.corpus_logging import logger
from lib.sqlite_vector_blob import vector_to_blob
from retrieval.lance_observation_index_store import LanceObservationIndexStore
from retrieval.models import SearchSpace

YEAR_BUCKET = 10

_TIER3_SCHEMA = """
CREATE TABLE IF NOT EXISTS event_geometry (
    concept       TEXT NOT NULL,
    event_id      INTEGER NOT NULL,
    nx            REAL NOT NULL,
    ny            REAL NOT NULL,
    gnx           REAL,
    gny           REAL,
    cluster_id    INTEGER NOT NULL,
    cluster_label TEXT,
    PRIMARY KEY (concept, event_id)
);

CREATE TABLE IF NOT EXISTS concept_cluster_info (
    concept          TEXT NOT NULL,
    cluster_id       INTEGER NOT NULL,
    cluster_label    TEXT,
    centroid_nx      REAL,
    centroid_ny      REAL,
    centroid_gnx     REAL,
    centroid_gny     REAL,
    centroid_vector  BLOB NOT NULL,
    point_count      INTEGER NOT NULL,
    description      TEXT,
    PRIMARY KEY (concept, cluster_id)
);
"""


def sqlite_connection(path: Path):
    con = analysis_db_connection(path)
    con.execute("PRAGMA busy_timeout=5000")
    return con


def tier3_connection(path: Path):
    con = sqlite_connection(path)
    con.executescript(_TIER3_SCHEMA)
    con.commit()
    return con


def write_geometry_sqlite(
    con,
    concept,
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
                concept,
                int(event_id),
                float(local_coords[idx][0]),
                float(local_coords[idx][1]),
                float(gx) if np.isfinite(gx) else None,
                float(gy) if np.isfinite(gy) else None,
                int(clusters[idx]),
                "noise" if int(clusters[idx]) == -1 else None,
            )
        )

    con.executemany(
        """
        INSERT INTO event_geometry (
            concept,
            event_id,
            nx,
            ny,
            gnx,
            gny,
            cluster_id,
            cluster_label
        )
        VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT (concept, event_id)
        DO UPDATE SET
            nx = excluded.nx,
            ny = excluded.ny,
            gnx = excluded.gnx,
            gny = excluded.gny,
            cluster_id = excluded.cluster_id,
            cluster_label = excluded.cluster_label
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
    con.execute(
        """
        DELETE FROM concept_cluster_info
        WHERE concept = ?
        """,
        (concept,),
    )

    data = []

    for cluster_id in sorted(
        set(int(x) for x in clusters)
    ):
        mask = clusters == cluster_id

        if not np.any(mask):
            continue

        centroid_vector = cluster_centroid_vectors.get(cluster_id)

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


def load_pub_years(pg_con, event_ids):
    if not event_ids:
        return {}

    with pg_con.cursor() as cur:
        cur.execute(
            """
            SELECT event_id, pub_year
            FROM events
            WHERE event_id = ANY(%s)
            """,
            (event_ids,),
        )

        rows = cur.fetchall()

    pub_years = {
        int(event_id): pub_year
        for event_id, pub_year in rows
    }

    missing = [
        event_id
        for event_id in event_ids
        if event_id not in pub_years
    ]

    if missing:
        raise RuntimeError(
            f"Tier 3: {len(missing)} field events are missing "
            f"from PostgreSQL events; first IDs: {missing[:10]}"
        )

    missing_years = [
        event_id
        for event_id in event_ids
        if pub_years[event_id] is None
    ]

    if missing_years:
        raise RuntimeError(
            f"Tier 3: {len(missing_years)} events have no publication "
            f"year; first IDs: {missing_years[:10]}"
        )

    return {
        event_id: int(pub_years[event_id])
        for event_id in event_ids
    }


def cluster_concept(
    *,
    load_rows,
    pub_years,
    write_geometry,
    write_cluster_info,
    commit,
    index,
    concept: str,
    global_coords: dict[int, NDArray[np.float32]] | None,
    resolution_parameter: float,
    n_neighbors: int,
) -> dict[str, object]:
    logger.info(f"[tier3] processing {concept}")

    rows = load_rows(concept)

    if not rows:
        logger.warning(f"[tier3] {concept}: no events")
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
        pub_years[event_id] // YEAR_BUCKET
        for event_id in event_ids
    ]

    logger.info(
        f"[tier3] {concept}: field events={len(event_ids):,}"
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
    cluster_centroid_vectors = result["cluster_centroid_vectors"]
    fit_info = result["fit_info"]

    if fit_info["sampled"]:
        logger.info(
            f"[tier3] {concept}: sampled fit "
            f"({fit_info['fit_n']:,}/{fit_info['n']:,} events, "
            f"{fit_info['outlier_n']:,} guaranteed outliers)"
        )

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

    write_geometry(
        concept,
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
            np.sum(clusters == -1)
        ),
        "sampled": fit_info["sampled"],
        "fit_events": fit_info["fit_n"],
        "outlier_events": fit_info["outlier_n"],
    }


def build_tier3_resources(
    *,
    tier2_db_path=None,
    tier3_db_path=None,
):
    """
    Build the shared Tier 3 resources.

    Tier 2 supplies field membership.
    PostgreSQL supplies publication years.
    Lance supplies embeddings.
    Tier 3 SQLite stores the derived geometry and clusters.
    """
    tier2_db_path = Path(
        tier2_db_path or CORPUS_TIER2_DB_PATH
    )
    tier3_db_path = Path(
        tier3_db_path or CORPUS_TIER3_DB_PATH
    )

    lance_store = LanceObservationIndexStore(
        LANCE_INDEXES_DIR,
        available_scales=("local",),
    )

    indexes = lance_store.get(
        SearchSpace(
            years=(
                CORPUS_MIN_YEAR,
                CORPUS_MAX_YEAR,
            ),
            scale=("local",),
        )
    )

    index = indexes["local"]

    tier2_con = sqlite_connection(tier2_db_path)
    tier3_con = tier3_connection(tier3_db_path)
    pg_con = get_connection()

    concepts = [
        concept
        for concept, _
        in resolve_concepts(concept=None)
    ]

    present = {
        row[0]
        for row in tier2_con.execute(
            "SELECT concept FROM concepts"
        )
    }

    concepts = [
        concept
        for concept in concepts
        if concept in present
    ] or sorted(present)

    def load_rows_for_concept(concept):
        return load_event_rows(
            tier2_con,
            concept,
        )

    global_coords = None

    return {
        "backend": "tier2-sqlite+postgres+lance",
        "index": index,
        "tier2_con": tier2_con,
        "tier3_con": tier3_con,
        "pg_con": pg_con,
        "global_coords": global_coords,
        "concepts": concepts,
        "load_rows": load_rows_for_concept,
        "pub_years": {},
        "write_geometry": (
            lambda concept,
            event_ids,
            local_coords,
            global_coords,
            clusters:
            write_geometry_sqlite(
                tier3_con,
                concept,
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
                tier3_con,
                concept,
                cluster_centroid_vectors,
                local_coords,
                global_coords,
                clusters,
            )
        ),
        "commit": tier3_con.commit,
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

    rows = resources["load_rows"](concept)

    event_ids = [
        int(row[0])
        for row in rows
    ]

    pub_years = load_pub_years(
        resources["pg_con"],
        event_ids,
    )

    report = cluster_concept(
        load_rows=resources["load_rows"],
        pub_years=pub_years,
        write_geometry=resources["write_geometry"],
        write_cluster_info=resources["write_cluster_info"],
        commit=resources["commit"],
        index=resources["index"],
        concept=concept,
        global_coords=resources["global_coords"],
        resolution_parameter=resolution_parameter,
        n_neighbors=n_neighbors,
    )

    elapsed = time.perf_counter() - started

    logger.info(
        f"[tier3-service] completed "
        f"{concept} in {elapsed:.2f}s"
    )

    return {
        **report,
        "resolution_parameter": resolution_parameter,
        "n_neighbors": n_neighbors,
        "elapsed_seconds": round(elapsed, 3),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--concept",
        default=None,
    )

    parser.add_argument(
        "--tier2-db",
        type=Path,
        default=CORPUS_TIER2_DB_PATH,
        help="Tier 2 SQLite input database",
    )

    parser.add_argument(
        "--tier3-db",
        type=Path,
        default=CORPUS_TIER3_DB_PATH,
        help="Tier 3 SQLite output database",
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
        tier2_db_path=args.tier2_db,
        tier3_db_path=args.tier3_db,
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
            f"[tier3-main] backend={resources['backend']} "
            f"concepts={len(concepts)}"
        )

        for concept in concepts:
            result = service(
                resources=resources,
                concept=concept,
                resolution_parameter=args.resolution,
                n_neighbors=args.neighbors,
            )

            logger.info(
                f"[tier3-main] completed "
                f"{result.get('concept')}"
            )

    finally:
        resources["tier2_con"].close()
        resources["tier3_con"].close()
        resources["pg_con"].close()

    logger.info("[tier3-main] Done.")


if __name__ == "__main__":
    main()
