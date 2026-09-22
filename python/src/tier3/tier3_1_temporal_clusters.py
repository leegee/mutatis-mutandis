#!/usr/bin/env python

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
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
    LANCE_INDEXES_DIR,
)
from lib.corpus_db import get_connection
from lib.corpus_logging import logger
from lib.vector_blob import vector_to_bytes
from retrieval.lance_observation_index import LanceObservationIndex
from retrieval.lance_observation_index_store import (
    LanceObservationIndexStore,
)
from retrieval.models import SearchSpace


CLUSTER_SCALE = "medium"


def initialise_temporal_tables(con) -> None:
    """
    Create Tier 3.1 persistence tables if they do not already exist.

    PostgreSQL owns temporal-cluster persistence. Lance remains authoritative
    for embedding vectors used during clustering.
    """
    with con.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tier3.concept_year_cluster_info (
                concept TEXT NOT NULL,
                pub_year INTEGER NOT NULL,
                cluster_id INTEGER NOT NULL,
                cluster_label TEXT,
                centroid_nx DOUBLE PRECISION,
                centroid_ny DOUBLE PRECISION,
                centroid_gnx DOUBLE PRECISION,
                centroid_gny DOUBLE PRECISION,
                centroid_vector BYTEA,
                point_count INTEGER,
                description TEXT,
                relative_mass DOUBLE PRECISION,
                PRIMARY KEY (
                    concept,
                    pub_year,
                    cluster_id
                )
            )
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_year_clusters
            ON tier3.concept_year_cluster_info (
                concept,
                pub_year
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tier3.concept_year_event_cluster (
                concept TEXT NOT NULL,
                pub_year INTEGER NOT NULL,
                event_id BIGINT NOT NULL,
                cluster_id INTEGER NOT NULL,
                PRIMARY KEY (
                    concept,
                    pub_year,
                    event_id
                )
            )
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_year_event_cluster_lookup
            ON tier3.concept_year_event_cluster (
                concept,
                pub_year,
                cluster_id
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tier3.temporal_cluster_edges (
                concept TEXT NOT NULL,
                source_year INTEGER NOT NULL,
                source_cluster INTEGER NOT NULL,
                target_year INTEGER NOT NULL,
                target_cluster INTEGER NOT NULL,
                similarity DOUBLE PRECISION,
                edge_type TEXT,
                confidence DOUBLE PRECISION,
                PRIMARY KEY (
                    concept,
                    source_year,
                    source_cluster,
                    target_year,
                    target_cluster,
                    edge_type
                )
            )
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_temporal_edges_source
            ON tier3.temporal_cluster_edges (
                concept,
                source_year,
                source_cluster
            )
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_temporal_edges_target
            ON tier3.temporal_cluster_edges (
                concept,
                target_year,
                target_cluster
            )
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_temporal_edges_similarity
            ON tier3.temporal_cluster_edges (
                concept,
                similarity
            )
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_temporal_edges_year_transition
            ON tier3.temporal_cluster_edges (
                concept,
                source_year,
                target_year
            )
            """
        )

    con.commit()


def clear_temporal_clusters(con) -> None:
    logger.info(
        "[tier3.1] clearing temporal cluster output"
    )

    with con.cursor() as cur:
        cur.execute(
            "DROP TABLE IF EXISTS tier3.concept_year_cluster_info"
        )
        cur.execute(
            "DROP TABLE IF EXISTS tier3.concept_year_event_cluster"
        )
        cur.execute(
            "DROP TABLE IF EXISTS tier3.temporal_cluster_edges"
        )

    con.commit()
    initialise_temporal_tables(con)


def delete_temporal_edges(
    con,
    concept: str,
) -> None:
    with con.cursor() as cur:
        cur.execute(
            """
            DELETE FROM tier3.temporal_cluster_edges
            WHERE concept = %s
            """,
            (concept,),
        )


def delete_concept_clusters(
    con,
    concept: str,
) -> None:
    with con.cursor() as cur:
        cur.execute(
            """
            DELETE FROM tier3.concept_year_cluster_info
            WHERE concept = %s
            """,
            (concept,),
        )

        cur.execute(
            """
            DELETE FROM tier3.concept_year_event_cluster
            WHERE concept = %s
            """,
            (concept,),
        )


def load_concept_event_rows(
    con,
    concept: str,
) -> dict[int, list[int]]:
    """
    Load the complete persisted Tier 2 semantic field and group event IDs
    by authoritative publication year.
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
              AND e.pub_year IS NOT NULL
            ORDER BY e.pub_year, ef.event_id
            """,
            (concept,),
        )

        rows = cur.fetchall()

    if not rows:
        raise RuntimeError(
            f"[tier3.1] no Tier 2 field events found "
            f"for concept={concept!r}"
        )

    by_year: dict[int, list[int]] = {}

    for event_id, pub_year in rows:
        by_year.setdefault(
            int(pub_year),
            [],
        ).append(int(event_id))

    logger.info(
        "[tier3.1] %s: %d events across %d years",
        concept,
        sum(len(ids) for ids in by_year.values()),
        len(by_year),
    )

    return by_year


def load_event_vectors(
    index: LanceObservationIndex,
    event_ids: list[int],
) -> tuple[list[int], np.ndarray]:
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
    concept: str,
    pub_year: int,
    cluster_records,
) -> None:
    rows = []

    for cluster in cluster_records:
        rows.append(
            (
                concept,
                pub_year,
                int(cluster["cluster_id"]),
                (
                    "noise"
                    if cluster["cluster_id"] == -1
                    else None
                ),
                cluster["centroid_nx"],
                cluster["centroid_ny"],
                cluster["centroid_gnx"],
                cluster["centroid_gny"],
                vector_to_bytes(
                    cluster["centroid_vector"]
                ),
                cluster["point_count"],
                None,
                cluster["relative_mass"],
            )
        )

    if not rows:
        return

    with con.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO tier3.concept_year_cluster_info (
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
                description,
                relative_mass
            )
            VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s
            )
            """,
            rows,
        )


def write_year_event_cluster_map(
    con,
    concept: str,
    pub_year: int,
    event_ids: list[int],
    clusters,
) -> None:
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

    if not rows:
        return

    with con.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO tier3.concept_year_event_cluster (
                concept,
                pub_year,
                event_id,
                cluster_id
            )
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (
                concept,
                pub_year,
                event_id
            )
            DO UPDATE SET
                cluster_id = EXCLUDED.cluster_id
            """,
            rows,
        )


def process_concept_year(
    con,
    index: LanceObservationIndex,
    concept: str,
    pub_year: int,
    event_ids: list[int],
    resolution_parameter: float,
    n_neighbors: int,
) -> None:
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

    # Global UMAP is currently disabled. Global coordinates therefore remain
    # NULL rather than reusing local coordinates that are not cross-year
    # comparable.
    global_xy = np.full(
        (len(event_ids), 2),
        np.nan,
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
    con,
    index: LanceObservationIndex,
    concept: str,
    resolution_parameter: float,
    n_neighbors: int,
) -> None:
    by_year = load_concept_event_rows(
        con,
        concept,
    )

    if not by_year:
        logger.warning(
            "[tier3.1] no events for concept=%s",
            concept,
        )
        return

    delete_concept_clusters(
        con,
        concept,
    )

    for pub_year, event_ids in by_year.items():
        process_concept_year(
            con,
            index,
            concept,
            pub_year,
            event_ids,
            resolution_parameter,
            n_neighbors,
        )


def load_year_clusters(
    con,
    concept: str,
    pub_year: int,
) -> list[tuple[int, np.ndarray]]:
    with con.cursor() as cur:
        cur.execute(
            """
            SELECT
                cluster_id,
                centroid_vector
            FROM tier3.concept_year_cluster_info
            WHERE
                concept = %s
                AND pub_year = %s
                AND cluster_id >= 0
            ORDER BY cluster_id
            """,
            (
                concept,
                pub_year,
            ),
        )

        rows = cur.fetchall()

    result = []

    for cluster_id, blob in rows:
        if blob is None:
            continue

        vector = np.frombuffer(
            bytes(blob),
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
    concept: str,
    similarity_threshold: float = 0.95,
) -> None:
    logger.info(
        "[tier3.1] building temporal edges %s",
        concept,
    )

    delete_temporal_edges(
        con,
        concept,
    )

    with con.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT pub_year
            FROM tier3.concept_year_cluster_info
            WHERE
                concept = %s
                AND cluster_id >= 0
            ORDER BY pub_year
            """,
            (concept,),
        )

        years = [
            int(row[0])
            for row in cur.fetchall()
        ]

    edges = []
    year_clusters_cache: dict[
        int,
        list[tuple[int, np.ndarray]],
    ] = {}

    def get_year_clusters(
        year: int,
    ) -> list[tuple[int, np.ndarray]]:
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

            # Retain the margin calculation for future confidence
            # diagnostics without treating it as the current confidence
            # definition.
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

    if not edges:
        logger.info(
            "[tier3.1] edges created: 0",
        )
        return

    with con.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO tier3.temporal_cluster_edges (
                concept,
                source_year,
                source_cluster,
                target_year,
                target_cluster,
                similarity,
                edge_type,
                confidence
            )
            VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s
            )
            ON CONFLICT (
                concept,
                source_year,
                source_cluster,
                target_year,
                target_cluster,
                edge_type
            )
            DO UPDATE SET
                similarity = EXCLUDED.similarity,
                confidence = EXCLUDED.confidence
            """,
            edges,
        )

    logger.info(
        "[tier3.1] edges created: %d",
        len(edges),
    )


def build_tier3_1_index(
    lance_root: str | Path = LANCE_INDEXES_DIR,
) -> LanceObservationIndex:
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

    return indexes[CLUSTER_SCALE]


_WORKER_CON = None
_WORKER_INDEX = None


def _pin_single_threaded_math_libs() -> None:
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
    lance_root,
):
    global _WORKER_CON, _WORKER_INDEX

    _pin_single_threaded_math_libs()

    _WORKER_CON = get_connection()
    initialise_temporal_tables(
        _WORKER_CON
    )

    _WORKER_INDEX = build_tier3_1_index(
        lance_root,
    )


def _process_concept_worker(
    concept,
    similarity_threshold,
    resolution_parameter,
    n_neighbors,
):
    global _WORKER_CON, _WORKER_INDEX

    try:
        with _WORKER_CON.transaction():
            process_concept(
                _WORKER_CON,
                _WORKER_INDEX,
                concept,
                resolution_parameter,
                n_neighbors,
            )

            build_temporal_edges(
                _WORKER_CON,
                concept,
                similarity_threshold,
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
    con,
    concepts,
    workers,
    lance_root,
    similarity_threshold,
    resolution_parameter,
    n_neighbors,
):
    con.close()

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
            lance_root,
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


def main() -> None:
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

    parser.add_argument(
        "--lance-root",
        type=Path,
        default=LANCE_INDEXES_DIR,
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

    con = get_connection()

    try:
        initialise_temporal_tables(
            con
        )

        if args.clear:
            clear_temporal_clusters(
                con
            )

        index = build_tier3_1_index(
            args.lance_root,
        )

        logger.info(
            "[tier3.1] opened %s Lance observation index",
            CLUSTER_SCALE,
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

        if not concepts:
            logger.warning(
                "[tier3.1] no concepts resolved"
            )
            return

        if args.workers > 1:
            run_parallel(
                con,
                concepts,
                args.workers,
                args.lance_root,
                args.similarity_threshold,
                args.resolution,
                args.neighbors,
            )

        else:
            for concept in concepts:
                try:
                    with con.transaction():
                        process_concept(
                            con,
                            index,
                            concept,
                            args.resolution,
                            args.neighbors,
                        )

                        build_temporal_edges(
                            con,
                            concept,
                            args.similarity_threshold,
                        )

                except Exception:
                    logger.exception(
                        "[tier3.1] concept=%s failed",
                        concept,
                    )
                    raise

    finally:
        con.close()

    logger.info(
        "[tier3.1] Done."
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
