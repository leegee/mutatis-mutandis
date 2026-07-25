#!/usr/bin/env python

# tier3_1_temporal_clusters.py

from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import os
import time
import sqlite3
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import hashlib

from lib.eebo_config import (
    CORPUS_TIER2_DB_PATH,
    ZARR_PATH,
    faiss_index_paths,
    TMP_DIR,
)

from lib.concept_resolve import resolve_concepts
from lib.eebo_faiss import EeboFaissIndex
from lib.eebo_logging import logger
from lib.zarr_event_lookup import ZarrEventLookup
from lib.sqlite_vector_blob import vector_to_blob

from lib.cluster import (
    LOCAL_UMAP_PARAMS,
    build_global_projection,
    leiden_cluster,
    load_vectors,
    project,
    compute_cluster_centroids,
)


GLOBAL_PROJECTION_CACHE = TMP_DIR / "tier3_global_projection.npz"


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

CREATE INDEX IF NOT EXISTS idx_year_clusters ON concept_year_cluster_info ( concept, pub_year );

-- Per-event cluster assignment for a (concept, pub_year) Leiden run.
-- process_concept_year() already computes this (event_ids, clusters)
-- before collapsing it into centroid rows via compute_cluster_centroids();
-- previously that per-event mapping was discarded once the centroids
-- were written. Without it, nothing can answer "which events actually
-- belong to this node" -- e.g. tier4's event sampling was joining
-- against events.cluster_id, which is a *different* clustering
-- (tier3_0's whole-concept Leiden run) with an unrelated ID space.
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
    ON concept_year_event_cluster ( concept, pub_year, cluster_id );

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
        target_cluster
    )
);

CREATE INDEX IF NOT EXISTS idx_temporal_edges_source          ON temporal_cluster_edges ( concept, source_year, source_cluster );
CREATE INDEX IF NOT EXISTS idx_temporal_edges_target          ON temporal_cluster_edges ( concept, target_year, target_cluster );
CREATE INDEX IF NOT EXISTS idx_temporal_edges_similarity      ON temporal_cluster_edges ( concept, similarity );
CREATE INDEX IF NOT EXISTS idx_temporal_edges_year_transition ON temporal_cluster_edges ( concept, source_year, target_year );
"""


def initialise_temporal_tables(con):
    """
    Ensure Tier 3.1 output tables exist.
    """
    con.executescript(YEAR_CLUSTER_SCHEMA)
    con.commit()


def clear_temporal_clusters(con):
    """
    Remove all Tier 3.1 output. Leaves Tier 2 data untouched.
    """
    logger.info("[tier3.1] clearing concept_year_cluster_info")
    con.execute( "DROP TABLE IF EXISTS concept_year_cluster_info" )
    con.execute( "DROP TABLE IF EXISTS concept_year_event_cluster" )
    con.execute( "DROP TABLE IF EXISTS temporal_cluster_edges" )
    con.commit()
    initialise_temporal_tables(con)


def delete_temporal_edges( con, concept, ):
    con.execute(" DELETE FROM temporal_cluster_edges WHERE concept=?", (concept,) )


def delete_concept_clusters( con, concept, ):
    """
    Remove all Tier 3.1 cluster rows for a concept, across every year --
    both the aggregate centroid rows and the per-event membership map.

    PERF: previously this was called once per (concept, year) inside
    write_year_cluster_info. Since we now batch-load and process all of
    a concept's years together in one pass, it's equivalent -- and far
    fewer statements -- to clear the whole concept once up front and
    then do plain INSERTs per year.
    """
    con.execute( "DELETE FROM concept_year_cluster_info WHERE concept=?", ( concept, ), )
    con.execute( "DELETE FROM concept_year_event_cluster WHERE concept=?", ( concept, ), )


def sqlite_connection(path: Path, busy_timeout_ms: int = 30000):
    con = sqlite3.connect(path)
    con.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
    # PERF: WAL lets readers and the writer coexist without blocking,
    # and NORMAL synchronous still fsyncs at WAL checkpoints (safe against
    # app crashes) but skips the fsync-per-commit that DELETE-mode/FULL
    # synchronous does. Big win given how many commits this script used
    # to issue. WAL is also what makes it safe for multiple worker
    # processes (see --workers) to hold their own connection and commit
    # concurrently -- writers still serialize at commit time, but readers
    # are never blocked, and busy_timeout absorbs the brief serialization
    # wait instead of raising "database is locked".
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA wal_autocheckpoint=1000")
    con.execute("PRAGMA locking_mode=NORMAL")
    return con


def event_id_hash(event_ids):
    h = hashlib.sha256()
    for eid in event_ids:
        h.update(str(eid).encode())
    return h.hexdigest()



def with_sqlite_retry(fn, retries=10, delay=0.5):
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


def load_indices(years, masked=False):
    index_paths = {
        year: faiss_index_paths(
            masked=masked,
            year=year,
        )
        for year in years
    }

    index = {}

    for year, paths in index_paths.items():
        index[year] = {}

        for scale, path in paths.items():
            index[year][scale] = EeboFaissIndex.load(path)

    return index


def load_concept_event_rows(
    con,
    concept,
):
    """
    PERF: replaces the old per-year query (load_year_event_rows), which
    re-ran the concept_field_events/events join once for every distinct
    pub_year. This pulls every row for the concept in a single query,
    already ordered by pub_year, and callers group it in Python with
    itertools.groupby -- one round trip and one join execution per
    concept instead of one per (concept, year).
    """
    rows = con.execute(
        """
        SELECT e.pub_year, e.event_id, e.vector_id
        FROM concept_field_events f
        JOIN events e ON e.event_id=f.event_id
        WHERE f.concept=?
        ORDER BY e.pub_year, e.event_id
        """,
        (concept,),
    ).fetchall()

    by_year = {}
    for pub_year, group in itertools.groupby(rows, key=lambda r: r[0]):
        by_year[pub_year] = [(event_id, vector_id) for (_, event_id, vector_id) in group]

    return by_year


def write_year_cluster_info(
    con,
    concept,
    pub_year,
    cluster_records,
):
    # NOTE: the per-year delete that used to live here has moved to
    # delete_concept_clusters(), called once per concept before the
    # year loop -- see process_concept().
    rows = []
    for c in cluster_records:
        rows.append( (
            concept,
            pub_year,
            c["cluster_id"],
            "noise" if c["cluster_id"] == -1 else None,
            c["centroid_nx"],
            c["centroid_ny"],
            c["centroid_gnx"],
            c["centroid_gny"],
            vector_to_blob( c["centroid_vector"] ),
            c["point_count"],
            c["relative_mass"],
            None,
        ) )

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
    """
    Persists the per-event -> cluster_id assignment that leiden_cluster()
    produces for this (concept, pub_year). This is exactly what
    compute_cluster_centroids() consumes to build the aggregate rows in
    concept_year_cluster_info -- it was already sitting in memory in
    process_concept_year(), just never written anywhere. Downstream
    consumers (e.g. tier4's per-node event sampling) need this table to
    answer "which concrete events are in this node" at all; joining
    against events.cluster_id is wrong here, since that column belongs
    to tier3_0's separate, whole-concept clustering pass and uses an
    unrelated cluster_id numbering.
    """
    rows = [
        ( concept, pub_year, int(event_id), int(cluster_id), )
        for event_id, cluster_id in zip( event_ids, clusters, )
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


def process_concept_year( con, lookup, concept, pub_year, rows, global_coords, ):
    logger.info( f"[tier3.1] {concept} {pub_year}" )

    if not rows:
        return

    event_ids, vectors = load_vectors( lookup, rows, )

    if len(event_ids) == 0:
        return

    local_coords = project( vectors, LOCAL_UMAP_PARAMS, )
    clusters = leiden_cluster( vectors, )

    global_xy = np.asarray(
        [
            global_coords[eid]
            for eid in event_ids
        ],
        dtype=np.float32,
    )

    cluster_records = compute_cluster_centroids( vectors, local_coords, global_xy, clusters, )

    total = sum(
        c["point_count"]
        for c in cluster_records
    )

    for c in cluster_records:
        c["relative_mass"] = (
            c["point_count"] / total
            if total > 0
            else 0.0
        )

    write_year_cluster_info( con, concept, pub_year, cluster_records, )
    write_year_event_cluster_map( con, concept, pub_year, event_ids, clusters, )


def process_concept( con, lookup, concept, global_coords, ):
    """
    Process every year for a single concept, then commit once.

    PERF: previously process_concept_year() queried the DB and committed
    once per (concept, year). This now issues a single batched query for
    the whole concept (load_concept_event_rows), clears old rows once,
    writes each year's clusters, and commits a single time at the end --
    cutting both DB round trips and fsync-triggering commits roughly by
    the average number of years per concept.
    """
    by_year = load_concept_event_rows( con, concept, )

    if not by_year:
        return

    delete_concept_clusters( con, concept, )

    for pub_year, rows in by_year.items():
        process_concept_year( con, lookup, concept, pub_year, rows, global_coords, )

    con.commit()


def load_year_clusters(
    con,
    concept,
    pub_year,
):
    rows = con.execute(
        """
        SELECT cluster_id, centroid_vector
        FROM concept_year_cluster_info
        WHERE concept=? AND pub_year=? AND cluster_id >= 0
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

        result.append( (
            int(cluster_id),
            vector,
        ) )

    return result


def build_temporal_edges( con, concept, similarity_threshold=0.95, ):
    logger.info( f"[tier3.1] building temporal edges {concept}" )
    delete_temporal_edges( con, concept )

    years = [
        row[0]
        for row in con.execute(
            """
            SELECT DISTINCT pub_year
            FROM concept_year_cluster_info
            WHERE concept=?
            ORDER BY pub_year
            """,
            (concept,),
        )
    ]

    edges = []

    for source_year, target_year in zip( years, years[1:], ):
        source_clusters = load_year_clusters( con, concept, source_year, )
        target_clusters = load_year_clusters( con, concept, target_year, )

        if not source_clusters or not target_clusters:
            continue


        source_ids = [
            x[0]
            for x in source_clusters
        ]

        source_vectors = np.vstack( [
            x[1]
            for x in source_clusters
        ] )


        target_ids = [
            x[0]
            for x in target_clusters
        ]

        target_vectors = np.vstack( [
            x[1]
            for x in target_clusters
        ] )


        similarity = cosine_similarity( source_vectors, target_vectors, )

        # Lineage per cluster
        for i, source_cluster in enumerate(source_ids):
            row = similarity[i]
            best_j = int(np.argmax(row))
            best_score = float(row[best_j])
            sorted_scores = np.sort(row)
            second_score = (
                float(sorted_scores[-2])
                if len(sorted_scores) > 1
                else 0.0
            )
            confidence = best_score - second_score
            if best_score >= similarity_threshold:
                edges.append( (
                    concept,
                    source_year,
                    source_cluster,
                    target_year,
                    target_ids[best_j],
                    best_score,
                    "CONTINUATION",
                    confidence
                ) )

        # Splits/merges:
        for i, source_cluster in enumerate(source_ids):
            for j, target_cluster in enumerate(target_ids):
                score = float(similarity[i, j])
                if score >= similarity_threshold:
                    edges.append( (
                        concept,
                        source_year,
                        source_cluster,
                        target_year,
                        target_cluster,
                        score,
                        "SIGNIFICANT",
                        score
                    ) )

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

    # con.commit()
    logger.info( f"[tier3.1] edges created: {len(edges)}" )


def load_or_build_global_projection( lookup, all_event_ids, cache_path, ):
    if cache_path is not None and cache_path.exists():
        logger.info( f"[tier3.1] loading cached global projection from {cache_path}" )
        cached = np.load( cache_path, allow_pickle=False, )
        cached_ids = cached["event_ids"]
        cached_xy = cached["xy"]

        cached_fingerprint = cached["fingerprint"].item()
        if cached_fingerprint != event_id_hash(cached["event_ids"]):
            raise RuntimeError( "Global projection cache fingerprint mismatch" )
        if cached_fingerprint == event_id_hash(all_event_ids):
            return {
                int(eid): xy
                for eid, xy in zip(cached_ids, cached_xy)
            }
        logger.info( "[tier3.1] cached global projection is stale (event set changed) -- rebuilding" )

    global_coords = build_global_projection( lookup, all_event_ids, )

    if cache_path is not None:
        ids_arr = np.asarray(
            all_event_ids,
            dtype=np.int64,
        )

        xy_arr = np.asarray(
            [
                global_coords[eid]
                for eid in all_event_ids
            ],
            dtype=np.float32,
        )

        np.savez(
            cache_path,
            event_ids=ids_arr,
            xy=xy_arr,
            fingerprint=event_id_hash(all_event_ids),
        )

        logger.info( f"[tier3.1] cached global projection to {cache_path}" )
    return global_coords


# ---------------------------------------------------------------------------
# Parallel (--workers > 1) support
# ---------------------------------------------------------------------------
#
# We parallelise across CONCEPTS, not years within a concept -- each concept
# is fully independent work (its own query, its own UMAP/Leiden calls, its
# own DB rows), which makes it the natural unit of parallelism.
#
# NOTE: lib/cluster.py's project() always calls umap.UMAP(random_state=42,
# ...). umap-learn forces n_jobs=1 internally whenever random_state is set
# (required for reproducibility), so every UMAP fit here -- including the
# one build_global_projection() runs in the parent before we fork -- is
# already constrained to a single thread by the library itself. That
# significantly reduces the odds of the hazard above actually manifesting,
# though we can't fully rule out the underlying numba threading layer still
# initializing its pool machinery even at n_jobs=1. Treat the cache-based
# mitigation below as cheap insurance rather than evidence the risk is high.
#
# The safe pattern is therefore:
#   1. First run (any time the event set changes): `--workers 1
#      --global-projection-cache PATH` to compute and cache global_coords.
#   2. Subsequent runs: `--workers N --global-projection-cache PATH` --
#      the cache hit means load_or_build_global_projection() only does
#      plain numpy I/O in the parent, so there's no known-multithreaded
#      library state alive at fork time.

_WORKER_LOOKUP = None
_WORKER_GLOBAL_COORDS = None
_WORKER_CON = None


def _pin_single_threaded_math_libs():
    """
    Each worker is itself a full process doing its own FAISS reconstruct/
    search calls and its own per-year UMAP/Leiden calls. If those libraries
    also try to multithread internally, N worker processes each spinning up
    M threads oversubscribes the machine and can make --workers slower than
    --workers 1. Pin every worker to single-threaded math; the speedup comes
    from running many concepts concurrently across processes, not from
    intra-call threading.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    try:
        import faiss
        faiss.omp_set_num_threads(1)
    except Exception:
        pass

    try:
        import numba
        numba.set_num_threads(1)
    except Exception:
        pass


def _init_worker(
    db_path,
    zarr_path,
    years,
    masked,
    cache_path,
    busy_timeout_ms,
):
    global _WORKER_CON, _WORKER_LOOKUP, _WORKER_GLOBAL_COORDS

    _pin_single_threaded_math_libs()

    _WORKER_CON = sqlite_connection(
        db_path,
        busy_timeout_ms=busy_timeout_ms,
    )
    _WORKER_CON.execute("PRAGMA busy_timeout=30000")

    _WORKER_LOOKUP = ZarrEventLookup(zarr_path)

    _WORKER_LOOKUP.attach_index(
        load_indices(
            years,
            masked=masked,
        )
    )

    cached = np.load(
        cache_path,
        allow_pickle=False,
    )

    _WORKER_GLOBAL_COORDS = {
        int(eid): xy
        for eid, xy in zip(
            cached["event_ids"],
            cached["xy"],
        )
    }


def _process_concept_worker(concept):
    global _WORKER_LOOKUP, _WORKER_GLOBAL_COORDS, _WORKER_CON

    try:
        def write_concept():
            try:
                _WORKER_CON.execute("BEGIN IMMEDIATE")

                process_concept(
                    _WORKER_CON,
                    _WORKER_LOOKUP,
                    concept,
                    _WORKER_GLOBAL_COORDS,
                )

                build_temporal_edges(
                    _WORKER_CON,
                    concept,
                )

                _WORKER_CON.commit()

            except Exception:
                if _WORKER_CON.in_transaction:
                    _WORKER_CON.rollback()
                raise

        with_sqlite_retry(write_concept)

        return (concept, None)

    except Exception as exc:
        logger.exception(
            f"[tier3.1] concept={concept} failed in worker"
        )
        return (concept, repr(exc))


def run_parallel( con, concepts, workers, db_path, years, masked, ):
    global _WORKER_LOOKUP, _WORKER_GLOBAL_COORDS

    _WORKER_LOOKUP = None
    _WORKER_GLOBAL_COORDS = None
    _WORKER_CON = None
    _WORKER_DB_PATH = None
    _WORKER_ZARR_PATH = None
    _WORKER_INDEX_YEARS = None
    _WORKER_MASKED = False

    # Do not need the parent's connection
    con.close()

    ctx = mp.get_context(
        "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    )

    failures = []
    with ctx.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(
            db_path,
            ZARR_PATH,
            years,
            masked,
            GLOBAL_PROJECTION_CACHE,
            30000,
        ),
    ) as pool:
        for concept, err in pool.imap_unordered(_process_concept_worker, concepts):
            if err is None:
                logger.info( f"[tier3.1] done: {concept}" )
            else:
                logger.error( f"[tier3.1] FAILED: {concept}: {err}" )
                failures.append(concept)

    if failures:
        raise SystemExit(
            f"[tier3.1] {len(failures)} concept(s) failed: {failures}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument( "--concept", default=None, )
    parser.add_argument( "--mask", action="store_true", )
    parser.add_argument( "--clear", action="store_true", help="Delete all temporal cluster output before processing.", )
    parser.add_argument( "--workers", type=int, default=1, help="Number of concepts to process in parallel", )

    args = parser.parse_args()

    lookup = ZarrEventLookup( ZARR_PATH )

    con = sqlite_connection( CORPUS_TIER2_DB_PATH )
    initialise_temporal_tables(con)
    if args.clear:
        clear_temporal_clusters(con)

    years = sorted( {
        int(y)
        for y in lookup.pub_year
        if y > 0
    } )


    all_rows = con.execute( "SELECT event_id FROM concept_field_events" )

    all_event_ids = sorted( {
        int(r[0])
        for r in all_rows
    } )

    global_coords = load_or_build_global_projection(
        lookup,
        all_event_ids,
        GLOBAL_PROJECTION_CACHE,
    )

    concepts = [ c for c, _ in resolve_concepts( concept=args.concept ) ]

    if args.workers > 1:
        run_parallel( con, concepts, args.workers, CORPUS_TIER2_DB_PATH, years, args.mask, )
    else:
        lookup.attach_index( load_indices( years, masked=args.mask, ) )
        for concept in concepts:
            process_concept( con, lookup, concept, global_coords, )

            # This must run once per concept -- previously it was
            # dedented to run only once after the loop, using whichever
            # `concept` was left over from the final iteration.
            build_temporal_edges( con, concept )
        con.close()
    logger.info( "[tier3.1] Done." )


if __name__ == "__main__":
    mp.freeze_support()
    main()
