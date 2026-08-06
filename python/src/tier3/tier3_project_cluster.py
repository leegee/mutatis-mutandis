#!/usr/bin/env python
"""
tier3_project_cluster

Cluster Tier-2 semantic fields (local UMAP + Leiden) and write geometry
back into the live store.

Preferred store: Postgres tier2_stage.* left alive by Tier 2
  (--stage-pg --no-publish-sqlite).

Fallback: SQLite Tier-2 DB (legacy / published artifact).

Vectors come from ZarrEventLookup.get_concatenated_embeddings — FAISS
indexes are not required for clustering.
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from lib.eebo_config import (
    CORPUS_TIER2_DB_PATH,
    ZARR_PATH,
    faiss_index_paths,
    discover_index_years,
)
from lib.concept_resolve import resolve_concepts
from lib.zarr_event_lookup import ZarrEventLookup
from lib.eebo_logging import logger
from lib.sqlite_vector_blob import vector_to_blob
from lib.eebo_db import get_connection
from lib.cluster import (
    LOCAL_UMAP_PARAMS,
    load_event_rows,
    local_project_and_cluster,
    build_global_projection,
)

from tier2.persistence import (
    load_event_rows_pg,
    list_stage_concepts_pg,
    write_geometry_pg,
    write_cluster_info_pg,
    dump_pg_stage_to_sqlite,
)
from tier2.resources import LazyYearIndices

# Diachronic stratum width for global UMAP fit sample:
# (concept, year // YEAR_BUCKET). 10 ≈ decade, 25 ≈ quarter-century.
YEAR_BUCKET = 10


def sqlite_connection(path: Path):
    con = sqlite3.connect(path)
    con.execute("PRAGMA busy_timeout=5000")
    return con


def write_geometry_sqlite(con, event_ids, local_coords, global_coords, clusters):
    rows = []
    for idx, event_id in enumerate(event_ids):
        rows.append(
            (
                float(local_coords[idx][0]),
                float(local_coords[idx][1]),
                float(global_coords[idx][0]),
                float(global_coords[idx][1]),
                int(clusters[idx]),
                "noise" if int(clusters[idx]) == -1 else None,
                int(event_id),
            )
        )
    con.executemany(
        """
        UPDATE events SET
            nx=?, ny=?, gnx=?, gny=?,
            cluster_id=?, cluster_label=?
        WHERE event_id=?
        """,
        rows,
    )


def write_cluster_info_sqlite(
    con, concept, cluster_centroid_vectors, local_coords, global_coords, clusters
):
    """
    `cluster_centroid_vectors` is {cluster_id: (dim,) float32}, computed
    streaming over the full field (see
    lib.cluster.compute_cluster_centroid_vectors_streaming) — NOT the
    full (n, dim) vectors matrix. This avoids needing the whole field's
    embeddings resident just to compute a per-cluster mean; local_coords/
    global_coords/clusters are still full-length (n,) arrays, which is
    cheap (no dim multiplier).
    """
    con.execute(
        "DELETE FROM concept_cluster_info WHERE concept = ?",
        (concept,),
    )
    data = []
    for cluster_id in sorted(set(int(x) for x in clusters)):
        mask = clusters == cluster_id
        if not np.any(mask):
            continue
        centroid_vector = cluster_centroid_vectors.get(cluster_id)
        if centroid_vector is None:
            continue
        data.append(
            (
                concept,
                int(cluster_id),
                "noise" if cluster_id == -1 else None,
                float(local_coords[mask, 0].mean()),
                float(local_coords[mask, 1].mean()),
                float(global_coords[mask, 0].mean()),
                float(global_coords[mask, 1].mean()),
                vector_to_blob(centroid_vector),
                int(mask.sum()),
                None,
            )
        )
    con.executemany(
        """
        INSERT INTO concept_cluster_info (
            concept, cluster_id, cluster_label,
            centroid_nx, centroid_ny,
            centroid_gnx, centroid_gny,
            centroid_vector, point_count, description
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
    global_coords: dict[int, NDArray[np.float32]],
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

    # event_ids + strata are cheap — no embeddings needed yet.
    # local_project_and_cluster loads vectors itself, bounded, and only
    # for the events it actually needs at each stage (coarse-centroid
    # scoring, the fit sample, and the transform batches).
    event_ids = [int(r[0]) for r in rows]
    strata = [
        int(r[2]) // YEAR_BUCKET if len(r) > 2 and r[2] is not None
        else int(lookup.pub_year[lookup.get_pos(int(r[0])) ]) // YEAR_BUCKET
        for r in rows
    ]

    if len(event_ids) == 0:
        return {
            "concept": concept,
            "status": "no-op",
            "reason": "No events",
        }

    logger.info(f"[tier3] {concept}: field events={len(event_ids):,}")

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
    cluster_centroid_vectors = result["cluster_centroid_vectors"]
    fit_info = result["fit_info"]

    if fit_info["sampled"]:
        logger.info(
            f"[tier3] {concept}: sampled fit "
            f"({fit_info['fit_n']:,}/{fit_info['n']:,} events, "
            f"{fit_info['outlier_n']:,} guaranteed outliers)"
        )

    global_xy = np.asarray(
        [global_coords[eid] for eid in event_ids],
        dtype=np.float32,
    )

    write_geometry(event_ids, local_coords, global_xy, clusters)
    write_cluster_info(concept, cluster_centroid_vectors, local_coords, global_xy, clusters)
    commit()

    return {
        "concept": concept,
        "status": "complete",
        "events": len(event_ids),
        "clusters": len({int(c) for c in clusters if c != -1}),
        "noise_points": int(np.sum(clusters == -1)),
        "sampled": fit_info["sampled"],
        "fit_events": fit_info["fit_n"],
        "outlier_events": fit_info["outlier_n"],
    }


def _attach_lazy_indices(lookup, masked=False, workers=1):
    """
    get_concatenated_embeddings requires attach_index(). Use LazyYearIndices
    so years load on first touch instead of all up front.
    """
    years = discover_index_years(masked)
    if not years:
        raise RuntimeError("No FAISS indices found")
    index_paths = {
        year: faiss_index_paths(masked=masked, year=year)
        for year in years
    }
    indexes = LazyYearIndices(index_paths, workers=workers)
    lookup.attach_index(indexes)
    return indexes


def build_tier3_resources(*, use_pg: bool, db_path=None, masked=False):
    """
    Build shared resources.

    use_pg=True  — read concepts/fields from Postgres tier2_stage
    use_pg=False — read from SQLite at db_path (default CORPUS_TIER2_DB_PATH)

    Embeddings require FAISS via lookup.attach_index(); we attach a
    LazyYearIndices so only years actually touched are loaded.

    *** BREAKING CHANGE for the PG backend — action required ***
    write_cluster_info's contract changed: it now receives
    `cluster_centroid_vectors` — a {cluster_id: (dim,) float32} dict
    computed streaming over the whole field — instead of the full
    (n, dim) `vectors` matrix + internal `vectors[mask].mean(axis=0)`.
    This is what let the LOCAL projection stop materialising a large
    concept's entire embedding matrix (see lib.cluster.
    local_project_and_cluster / compute_cluster_centroid_vectors_streaming).

    write_cluster_info_sqlite in this file has been updated to match
    (see above). tier2/persistence.py's write_cluster_info_pg has NOT
    — I don't have that file's source, so I can't safely edit it here.
    It needs the equivalent change: replace its `vectors[mask].mean(
    axis=0)` per cluster with a direct lookup into the now-dict-shaped
    `cluster_centroid_vectors` argument, exactly mirroring the sqlite
    version's diff above. Until that's done, --pg mode will pass a
    dict where write_cluster_info_pg expects an ndarray and will error
    (or silently misbehave, depending on what it does with `vectors[mask]`
    on a dict) — please share that file and I'll patch it directly.
    """
    lookup = ZarrEventLookup(ZARR_PATH)
    indexes = _attach_lazy_indices(lookup, masked=masked)

    if use_pg:
        pg = get_connection()
        concepts = list_stage_concepts_pg(pg)
        load_rows = lambda c: load_event_rows_pg(pg, c)

        all_field_event_ids = []
        strata = []
        for concept in concepts:
            rows = load_event_rows_pg(pg, concept)
            for r in rows:
                eid = int(r[0])
                # Prefer year from the row when present; otherwise Zarr lookup.
                if len(r) > 2 and r[2] is not None:
                    year = int(r[2])
                else:
                    year = int(lookup.pub_year[lookup.get_pos(eid)])
                all_field_event_ids.append(eid)
                strata.append((concept, year // YEAR_BUCKET))

        # Deduplicate event_ids; first (concept, bucket) wins.
        seen = {}
        for eid, s in zip(all_field_event_ids, strata):
            if eid not in seen:
                seen[eid] = s
        uniq_ids = list(seen.keys())
        uniq_strata = [seen[e] for e in uniq_ids]

        global_coords = build_global_projection(
            lookup, uniq_ids, strata=uniq_strata
        )

        def write_geom(eids, local_c, global_c, clusters):
            write_geometry_pg(pg, eids, local_c, global_c, clusters)

        def write_cinfo(concept, cluster_centroid_vectors, local_c, global_c, clusters):
            # NOTE: write_cluster_info_pg's contract has changed here —
            # see the big comment above build_tier3_resources for why,
            # and the required matching patch in tier2/persistence.py.
            write_cluster_info_pg(
                pg, concept, cluster_centroid_vectors, local_c, global_c, clusters, vector_to_blob
            )

        commit = (lambda: pg.commit()) if hasattr(pg, "commit") else (lambda: None)

        return {
            "backend": "pg",
            "pg": pg,
            "con": None,
            "lookup": lookup,
            "indexes": indexes,
            "global_coords": global_coords,
            "concepts": concepts,
            "load_rows": load_rows,
            "write_geometry": write_geom,
            "write_cluster_info": write_cinfo,
            "commit": commit,
        }

    path = db_path or CORPUS_TIER2_DB_PATH
    con = sqlite_connection(path)
    concepts = [
        concept for concept, _ in resolve_concepts(concept=None)
    ]
    present = {
        r[0]
        for r in con.execute("SELECT concept FROM concepts")
    }
    concepts = [c for c in concepts if c in present] or sorted(present)

    all_field_event_ids = []
    strata = []
    for concept in concepts:
        rows = load_event_rows(con, concept)
        for r in rows:
            eid = int(r[0])
            if len(r) > 2 and r[2] is not None:
                year = int(r[2])
            else:
                year = int(lookup.pub_year[lookup.get_pos(eid)])
            all_field_event_ids.append(eid)
            strata.append((concept, year // YEAR_BUCKET))

    seen = {}
    for eid, s in zip(all_field_event_ids, strata):
        if eid not in seen:
            seen[eid] = s
    uniq_ids = list(seen.keys())
    uniq_strata = [seen[e] for e in uniq_ids]

    global_coords = build_global_projection(
        lookup, uniq_ids, strata=uniq_strata
    )

    return {
        "backend": "sqlite",
        "pg": None,
        "con": con,
        "lookup": lookup,
        "indexes": indexes,
        "global_coords": global_coords,
        "concepts": concepts,
        "load_rows": lambda c: load_event_rows(con, c),
        "write_geometry": lambda eids, lc, gc, cl: write_geometry_sqlite(
            con, eids, lc, gc, cl
        ),
        "write_cluster_info": lambda concept, vectors, lc, gc, cl: write_cluster_info_sqlite(
            con, concept, vectors, lc, gc, cl
        ),
        "commit": lambda: con.commit(),
    }


def service(
    *,
    resources: dict,
    concept: str,
    resolution_parameter: float = 0.8,
    n_neighbors: int = 15,
) -> dict[str, object]:
    started = time.perf_counter()
    logger.info(f"[tier3-service] processing {concept}")

    report = cluster_concept(
        load_rows=resources["load_rows"],
        write_geometry=resources["write_geometry"],
        write_cluster_info=resources["write_cluster_info"],
        commit=resources["commit"],
        lookup=resources["lookup"],
        concept=concept,
        global_coords=resources["global_coords"],
        resolution_parameter=resolution_parameter,
        n_neighbors=n_neighbors,
    )

    elapsed = time.perf_counter() - started
    logger.info(f"[tier3-service] completed {concept} in {elapsed:.2f}s")

    return {
        **report,
        "resolution_parameter": resolution_parameter,
        "n_neighbors": n_neighbors,
        "elapsed_seconds": round(elapsed, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", default=None)
    parser.add_argument(
        "--pg",
        action="store_true",
        help="Read/write Postgres tier2_stage (leave stage alive from Tier 2)",
    )
    parser.add_argument(
        "--publish-sqlite",
        default=None,
        help="After clustering, dump stage → this SQLite path (PG mode only)",
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

    resources = build_tier3_resources(use_pg=args.pg)
    try:
        if args.concept:
            concepts = [args.concept.upper()]
        else:
            concepts = resources["concepts"]

        if not concepts:
            logger.warning("[tier3-main] no concepts resolved")
            return

        logger.info(
            f"[tier3-main] backend={resources['backend']} concepts={len(concepts)}"
        )

        for concept in concepts:
            result = service(
                resources=resources,
                concept=concept,
                resolution_parameter=args.resolution,
                n_neighbors=args.neighbors,
            )
            logger.info(f"[tier3-main] completed {result.get('concept')}")

        if args.pg and args.publish_sqlite:
            resources["commit"]()
            dump_pg_stage_to_sqlite(
                resources["pg"], args.publish_sqlite, clear=True
            )
            logger.info(f"[tier3-main] published SQLite → {args.publish_sqlite}")
    finally:
        if resources.get("con") is not None:
            resources["con"].close()
        pg = resources.get("pg")
        if pg is not None and hasattr(pg, "close"):
            pg.close()

    logger.info("[tier3-main] Done.")


if __name__ == "__main__":
    main()
