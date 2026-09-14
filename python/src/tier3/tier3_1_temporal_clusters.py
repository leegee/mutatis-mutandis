#!/usr/bin/env python

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sqlite3
import time
from functools import partial
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from lib.cluster import (
    LOCAL_UMAP_PARAMS,
    compute_cluster_centroids,
    leiden_cluster,
    project,
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
from retrieval.models import SearchSpace

from retrieval.lance_observation_index import LanceObservationIndex
from retrieval.lance_observation_index_store import LanceObservationIndexStore


CLUSTER_SCALE = "local"


YEAR_CLUSTER_SCHEMA = """
CREATE TABLE IF NOT EXISTS concept_year_cluster_info (
    concept TEXT NOT NULL,
    pub_year INTEGER NOT NULL,
    cluster_id INTEGER NOT NULL,
    cluster_label TEXT,
    centroid_nx REAL,
    centroid_ny REAL,
    centroid_gnx REAL,
    centroid_gny REAL,
    centroid_vector BLOB,
    point_count INTEGER,
    description TEXT,
    relative_mass REAL,
    PRIMARY KEY (
        concept,
        pub_year,
        cluster_id
    )
);

CREATE INDEX IF NOT EXISTS idx_year_clusters
ON concept_year_cluster_info (
    concept,
    pub_year
);

CREATE TABLE IF NOT EXISTS concept_year_event_cluster (
    concept TEXT NOT NULL,
    pub_year INTEGER NOT NULL,
    event_id INTEGER NOT NULL,
    cluster_id INTEGER NOT NULL,
    PRIMARY KEY (
        concept,
        pub_year,
        event_id
    )
);

CREATE INDEX IF NOT EXISTS idx_year_event_cluster_lookup
ON concept_year_event_cluster (
    concept,
    pub_year,
    cluster_id
);

CREATE TABLE IF NOT EXISTS temporal_cluster_edges (
    concept TEXT,
    source_year INTEGER,
    source_cluster INTEGER,
    target_year INTEGER,
    target_cluster INTEGER,
    similarity REAL,
    edge_type TEXT,
    confidence REAL,
    PRIMARY KEY (
        concept,
        source_year,
        source_cluster,
        target_year,
        target_cluster,
        edge_type
    )
);

CREATE INDEX IF NOT EXISTS idx_temporal_edges_source
ON temporal_cluster_edges (
    concept,
    source_year,
    source_cluster
);

CREATE INDEX IF NOT EXISTS idx_temporal_edges_target
ON temporal_cluster_edges (
    concept,
    target_year,
    target_cluster
);

CREATE INDEX IF NOT EXISTS idx_temporal_edges_similarity
ON temporal_cluster_edges (
    concept,
    similarity
);

CREATE INDEX IF NOT EXISTS idx_temporal_edges_year_transition
ON temporal_cluster_edges (
    concept,
    source_year,
    target_year
);
"""


def sqlite_connection(
    path: Path,
    busy_timeout_ms: int = 30000,
):
    con = analysis_db_connection(path)

    con.execute(
        f"PRAGMA busy_timeout={busy_timeout_ms}"
    )
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA wal_autocheckpoint=1000")
    con.execute("PRAGMA locking_mode=NORMAL")

    return con


def initialise_temporal_tables(con):
    con.executescript(YEAR_CLUSTER_SCHEMA)
    con.commit()


def clear_temporal_clusters(con):
    logger.info("[tier3.1] clearing temporal cluster output")

    con.execute(
        "DROP TABLE IF EXISTS concept_year_cluster_info"
    )
    con.execute(
        "DROP TABLE IF EXISTS concept_year_event_cluster"
    )
    con.execute(
        "DROP TABLE IF EXISTS temporal_cluster_edges"
    )

    con.commit()

    initialise_temporal_tables(con)


def delete_temporal_edges(con, concept):
    con.execute(
        """
        DELETE FROM temporal_cluster_edges
        WHERE concept=?
        """,
        (concept,),
    )


def delete_concept_clusters(con, concept):
    con.execute(
        """
        DELETE FROM concept_year_cluster_info
        WHERE concept=?
        """,
        (concept,),
    )

    con.execute(
        """
        DELETE FROM concept_year_event_cluster
        WHERE concept=?
        """,
        (concept,),
    )


def with_sqlite_retry(
    fn,
    retries=10,
    delay=0.5,
):
    for attempt in range(retries):
        try:
            return fn()

        except sqlite3.OperationalError as exc:
            if "database is locked" not in str(exc):
                raise

            if attempt == retries - 1:
                raise

            wait = delay * (2 ** attempt)

            logger.warning(
                "[tier3.1] database locked, retry %d/%d after %.1fs",
                attempt + 1,
                retries,
                wait,
            )

            time.sleep(wait)


def load_concept_event_rows(
    tier2_con,
    pg_con,
    concept,
):
    """
    Load the empirical semantic field from Tier 2 and resolve publication
    years from PostgreSQL.

    Tier 2 owns field membership. PostgreSQL owns corpus event metadata.
    """

    event_rows = tier2_con.execute(
        """
        SELECT event_id
        FROM event_field
        WHERE concept=?
        ORDER BY event_id
        """,
        (concept,),
    ).fetchall()

    if not event_rows:
        raise RuntimeError(
            f"[tier3.1] no Tier 2 field events found for concept={concept!r}"
        )

    event_ids = [
        int(row[0])
        for row in event_rows
    ]

    metadata = pg_con.execute(
        """
        SELECT event_id, pub_year
        FROM events
        WHERE event_id = ANY(%s)
        """,
        (event_ids,),
    ).fetchall()

    metadata_by_id = {
        int(event_id): pub_year
        for event_id, pub_year in metadata
    }

    missing = [
        event_id
        for event_id in event_ids
        if event_id not in metadata_by_id
    ]

    if missing:
        raise RuntimeError(
            f"[tier3.1] {len(missing)} event(s) in Tier 2 "
            f"event_field are absent from PostgreSQL"
        )

    missing_year = [
        event_id
        for event_id in event_ids
        if metadata_by_id[event_id] is None
    ]

    if missing_year:
        raise RuntimeError(
            f"[tier3.1] {len(missing_year)} event(s) have no pub_year"
        )

    by_year = {}

    for event_id in event_ids:
        pub_year = int(metadata_by_id[event_id])

        by_year.setdefault(
            pub_year,
            [],
        ).append(event_id)

    logger.info(
        "[tier3.1] %s: %d events across %d years",
        concept,
        len(event_ids),
        len(by_year),
    )

    return by_year


def load_event_vectors(
    index,
    event_ids,
):
    """
    Reconstruct embeddings from Lance using stable event IDs.

    The returned vector order must exactly match event_ids. This invariant
    is required because cluster assignments are persisted against event IDs.
    """

    if not event_ids:
        return (
            [],
            np.empty(
                (0, 0),
                dtype=np.float32,
            ),
        )

    vectors = index.reconstruct_many(
        event_ids
    )

    vectors = np.asarray(
        vectors,
        dtype=np.float32,
    )

    if vectors.ndim != 2:
        raise RuntimeError(
            f"Expected 2-D embeddings, got shape={vectors.shape}"
        )

    if len(vectors) != len(event_ids):
        raise RuntimeError(
            "Embedding/event alignment mismatch: "
            f"{len(event_ids)} event IDs but {len(vectors)} vectors"
        )

    if not np.isfinite(vectors).all():
        raise RuntimeError(
            "Lance returned non-finite embedding values"
        )

    return (
        list(event_ids),
        vectors,
    )


def write_year_cluster_info(
    con,
    concept,
    pub_year,
    cluster_records,
):
    rows = []

    for cluster in cluster_records:
        rows.append(
            (
                concept,
                pub_year,
                cluster["cluster_id"],
                (
                    "noise"
                    if cluster["cluster_id"] == -1
                    else None
                ),
                cluster["centroid_nx"],
                cluster["centroid_ny"],
                cluster["centroid_gnx"],
                cluster["centroid_gny"],
                vector_to_blob(
                    cluster["centroid_vector"]
                ),
                cluster["point_count"],
                cluster["relative_mass"],
                None,
            )
        )

    con.executemany(
        """
        INSERT INTO concept_year_cluster_info (
            concept,
            pub_year,
            cluster_id,
            cluster_label,
            centroid_nx,
            centroid_ny,
            centroid_gnx,
            centroid_gny,
            centroid_vector,
            point_count,
            relative_mass,
            description
        )
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )


def write_year_event_cluster_map(
    con,
    concept,
    pub_year,
    event_ids,
    clusters,
):
    rows = [
        (
            concept,
            pub_year,
            int(event_id),
            int(cluster_id),
        )
        for event_id, cluster_id in zip(
            event_ids,
            clusters,
        )
    ]

    con.executemany(
        """
        INSERT OR REPLACE INTO concept_year_event_cluster (
            concept,
            pub_year,
            event_id,
            cluster_id
        )
        VALUES (?,?,?,?)
        """,
        rows,
    )


def process_concept_year(
    con,
    index,
    concept,
    pub_year,
    event_ids,
    resolution_parameter,
    n_neighbors,
):
    logger.info(
        "[tier3.1] %s %s: %d events",
        concept,
        pub_year,
        len(event_ids),
    )

    if not event_ids:
        return

    event_ids, vectors = load_event_vectors(
        index,
        event_ids,
    )

    if len(event_ids) == 0:
        return

    local_coords = project(
        vectors,
        LOCAL_UMAP_PARAMS,
    )

    clusters = leiden_cluster(
        vectors,
        resolution_parameter=resolution_parameter,
        n_neighbors=n_neighbors,
    )

    # Tier 3.1 has no corpus-wide coordinate system while global UMAP is
    # disabled. The local coordinates therefore occupy both coordinate
    # columns for schema compatibility; they must not be interpreted as
    # comparable across publication years.
    global_xy = np.asarray(
        local_coords,
        dtype=np.float32,
    )

    cluster_records = compute_cluster_centroids(
        vectors,
        local_coords,
        global_xy,
        clusters,
    )

    total = sum(
        cluster["point_count"]
        for cluster in cluster_records
    )

    for cluster in cluster_records:
        cluster["relative_mass"] = (
            cluster["point_count"] / total
            if total > 0
            else 0.0
        )

    write_year_cluster_info(
        con,
        concept,
        pub_year,
        cluster_records,
    )

    write_year_event_cluster_map(
        con,
        concept,
        pub_year,
        event_ids,
        clusters,
    )


def process_concept(
    tier2_con,
    pg_con,
    tier3_con,
    index,
    concept,
    resolution_parameter,
    n_neighbors,
):
    by_year = load_concept_event_rows(
        tier2_con,
        pg_con,
        concept,
    )

    if not by_year:
        logger.warning(
            "[tier3.1] no events for concept=%s",
            concept,
        )
        return

    delete_concept_clusters(
        tier3_con,
        concept,
    )

    for pub_year, event_ids in by_year.items():
        process_concept_year(
            tier3_con,
            index,
            concept,
            pub_year,
            event_ids,
            resolution_parameter,
            n_neighbors,
        )

    tier3_con.commit()


def load_year_clusters(
    con,
    concept,
    pub_year,
):
    rows = con.execute(
        """
        SELECT
            cluster_id,
            centroid_vector
        FROM concept_year_cluster_info
        WHERE
            concept=?
            AND pub_year=?
            AND cluster_id >= 0
        """,
        (
            concept,
            pub_year,
        ),
    ).fetchall()

    result = []

    for cluster_id, blob in rows:
        vector = np.frombuffer(
            blob,
            dtype=np.float32,
        ).copy()

        norm = np.linalg.norm(vector)

        if norm > 0:
            vector = vector / norm

        result.append(
            (
                int(cluster_id),
                vector,
            )
        )

    return result


def build_temporal_edges(
    con,
    concept,
    similarity_threshold=0.95,
):
    logger.info(
        "[tier3.1] building temporal edges %s",
        concept,
    )

    delete_temporal_edges(
        con,
        concept,
    )

    years = [
        row[0]
        for row in con.execute(
            """
            SELECT DISTINCT pub_year
            FROM concept_year_cluster_info
            WHERE
                concept=?
                AND cluster_id >= 0
            ORDER BY pub_year
            """,
            (concept,),
        )
    ]

    edges = []

    year_clusters_cache = {}

    def get_year_clusters(year):
        cached = year_clusters_cache.get(year)

        if cached is None:
            cached = load_year_clusters(
                con,
                concept,
                year,
            )
            year_clusters_cache[year] = cached

        return cached

    for source_year, target_year in zip(
        years,
        years[1:],
    ):
        source_clusters = get_year_clusters(
            source_year,
        )

        target_clusters = get_year_clusters(
            target_year,
        )

        if not source_clusters or not target_clusters:
            continue

        source_ids = [
            cluster[0]
            for cluster in source_clusters
        ]

        source_vectors = np.vstack(
            [
                cluster[1]
                for cluster in source_clusters
            ]
        )

        target_ids = [
            cluster[0]
            for cluster in target_clusters
        ]

        target_vectors = np.vstack(
            [
                cluster[1]
                for cluster in target_clusters
            ]
        )

        similarity = cosine_similarity(
            source_vectors,
            target_vectors,
        )

        for i, source_cluster in enumerate(
            source_ids
        ):
            row = similarity[i]

            best_j = int(
                np.argmax(row)
            )

            best_score = float(
                row[best_j]
            )

            if len(row) > 1:
                sorted_scores = np.sort(row)
                second_score = float(
                    sorted_scores[-2]
                )
            else:
                second_score = 0.0

            margin = (
                best_score
                - second_score
            )

            # Margin is retained for diagnostics. Confidence currently
            # remains defined on the raw cosine-similarity scale.
            _ = margin

            confidence = best_score

            if best_score >= similarity_threshold:
                edges.append(
                    (
                        concept,
                        source_year,
                        source_cluster,
                        target_year,
                        target_ids[best_j],
                        best_score,
                        "CONTINUATION",
                        confidence,
                    )
                )

            for j, target_cluster in enumerate(
                target_ids
            ):
                if j == best_j:
                    continue

                score = float(
                    row[j]
                )

                if score < similarity_threshold:
                    continue

                edges.append(
                    (
                        concept,
                        source_year,
                        source_cluster,
                        target_year,
                        target_cluster,
                        score,
                        "SIGNIFICANT",
                        score,
                    )
                )

    con.executemany(
        """
        INSERT OR REPLACE INTO temporal_cluster_edges (
            concept,
            source_year,
            source_cluster,
            target_year,
            target_cluster,
            similarity,
            edge_type,
            confidence
        )
        VALUES (?,?,?,?,?,?,?,?)
        """,
        edges,
    )

    logger.info(
        "[tier3.1] edges created: %d",
        len(edges),
    )


_WORKER_TIER2_CON = None
_WORKER_TIER3_CON = None
_WORKER_PG_CON = None
_WORKER_INDEX = None


def _pin_single_threaded_math_libs():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    try:
        import numba

        numba.set_num_threads(1)

    except Exception:
        pass


def _init_worker(
    tier2_db_path,
    tier3_db_path,
    lance_root,
    busy_timeout_ms,
):
    global _WORKER_TIER2_CON, _WORKER_TIER3_CON, _WORKER_PG_CON, _WORKER_INDEX

    _pin_single_threaded_math_libs()

    _WORKER_TIER2_CON = sqlite_connection(
        tier2_db_path,
        busy_timeout_ms=busy_timeout_ms,
    )

    _WORKER_TIER3_CON = sqlite_connection(
        tier3_db_path,
        busy_timeout_ms=busy_timeout_ms,
    )

    initialise_temporal_tables(
        _WORKER_TIER3_CON
    )

    _WORKER_PG_CON = get_connection()

    lance_store = LanceObservationIndexStore(
        lance_root,
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

    _WORKER_INDEX = indexes["local"]


def _process_concept_worker(
    concept,
    similarity_threshold,
    resolution_parameter,
    n_neighbors,
):
    global _WORKER_TIER2_CON, _WORKER_TIER3_CON, _WORKER_PG_CON, _WORKER_INDEX

    try:

        def write_concept():
            try:
                _WORKER_TIER3_CON.execute(
                    "BEGIN IMMEDIATE"
                )

                process_concept(
                    _WORKER_TIER2_CON,
                    _WORKER_PG_CON,
                    _WORKER_TIER3_CON,
                    _WORKER_INDEX,
                    concept,
                    resolution_parameter,
                    n_neighbors,
                )

                build_temporal_edges(
                    _WORKER_TIER3_CON,
                    concept,
                    similarity_threshold,
                )

                _WORKER_TIER3_CON.commit()

            except Exception:
                if _WORKER_TIER3_CON.in_transaction:
                    _WORKER_TIER3_CON.rollback()

                raise

        with_sqlite_retry(
            write_concept
        )

        return (
            concept,
            None,
        )

    except Exception as exc:
        logger.exception(
            "[tier3.1] concept=%s failed in worker",
            concept,
        )

        return (
            concept,
            repr(exc),
        )


def run_parallel(
    tier2_con,
    tier3_con,
    pg_con,
    concepts,
    workers,
    tier2_db_path,
    tier3_db_path,
    lance_root,
    similarity_threshold,
    resolution_parameter,
    n_neighbors,
):
    global _WORKER_TIER2_CON, _WORKER_TIER3_CON, _WORKER_PG_CON, _WORKER_INDEX

    tier2_con.close()
    tier3_con.close()
    pg_con.close()

    ctx = mp.get_context(
        "fork"
        if "fork" in mp.get_all_start_methods()
        else "spawn"
    )

    failures = []

    with ctx.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(
            tier2_db_path,
            tier3_db_path,
            lance_root,
            30000,
        ),
    ) as pool:

        worker = partial(
            _process_concept_worker,
            similarity_threshold=similarity_threshold,
            resolution_parameter=resolution_parameter,
            n_neighbors=n_neighbors,
        )

        for concept, err in pool.imap_unordered(
            worker,
            concepts,
        ):
            if err is None:
                logger.info(
                    "[tier3.1] done: %s",
                    concept,
                )
            else:
                logger.error(
                    "[tier3.1] FAILED: %s: %s",
                    concept,
                    err,
                )
                failures.append(concept)

    if failures:
        raise SystemExit(
            f"[tier3.1] {len(failures)} concept(s) failed: "
            f"{failures}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--concept",
        default=None,
    )

    parser.add_argument(
        "-t",
        "--similarity-threshold",
        type=float,
        default=0.85,
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

    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete all Tier 3.1 temporal cluster output.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concepts to process in parallel.",
    )

    args = parser.parse_args()

    logger.info(
        "[tier3.1] options: %s",
        vars(args),
    )

    logger.info(
        "[tier3.1] cluster embedding scale: %s",
        CLUSTER_SCALE,
    )

    if CLUSTER_SCALE != "local":
        raise RuntimeError(
            "Tier 3.1 must use the local Lance scale"
        )

    tier2_con = sqlite_connection(
        CORPUS_TIER2_DB_PATH
    )

    tier3_con = sqlite_connection(
        CORPUS_TIER3_DB_PATH
    )

    pg_con = get_connection()

    initialise_temporal_tables(
        tier3_con
    )

    if args.clear:
        clear_temporal_clusters(
            tier3_con
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

    logger.info(
        "[tier3.1] opened local Lance observation index"
    )

    concepts = [
        concept
        for concept, _
        in resolve_concepts(
            concept=args.concept
        )
    ]

    logger.info(
        "[tier3.1] processing %d concept(s)",
        len(concepts),
    )

    if args.workers > 1:
        run_parallel(
            tier2_con,
            tier3_con,
            pg_con,
            concepts,
            args.workers,
            CORPUS_TIER2_DB_PATH,
            CORPUS_TIER3_DB_PATH,
            LANCE_INDEXES_DIR,
            args.similarity_threshold,
            args.resolution,
            args.neighbors,
        )

    else:
        try:
            for concept in concepts:
                process_concept(
                    tier2_con,
                    pg_con,
                    tier3_con,
                    index,
                    concept,
                    args.resolution,
                    args.neighbors,
                )

                build_temporal_edges(
                    tier3_con,
                    concept,
                    args.similarity_threshold,
                )

                tier3_con.commit()

        finally:
            tier2_con.close()
            tier3_con.close()
            pg_con.close()

    logger.info(
        "[tier3.1] Done."
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
