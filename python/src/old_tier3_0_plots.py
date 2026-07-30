#!/usr/bin/env python

"""
tier3_0_plots.py - the nightmare plotter that needs to be rewritten

Outputs:
    Sqlite3 DB shared with fronto-end

Each point carries six coordinates:

    Local projection  (independent UMAP fit on this file's points only)
        x,  y   — raw local UMAP output
        nx, ny  — normalised within this file's own padded bounds  [0, 1]

    Global projection  (single joint UMAP fit across ALL points in the run)
        gx,  gy  — raw global UMAP output
        gnx, gny — normalised within the shared global padded bounds  [0, 1]

concept_neighbours and bfs_global files additionally carry:
    depth  — 0 = concept seed, 1 = direct neighbour, 2 = second-order neighbour

concept_clusters files additionally carry:
    cluster_id      — integer cluster id from HDBSCAN or Leiden (-1 = noise)
    cluster_label   — letter label A, B, C... (null for noise)

Note: all event_id values are serialised as JSON strings to avoid JS/TS
BigInt issues with integers exceeding Number.MAX_SAFE_INTEGER (2^53 - 1).
"""

import argparse
import json
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import umap
import hdbscan
import pacmap
import sqlite3
import leidenalg
import igraph as ig

from lib.eebo_config import (
    PLOT_DIR, faiss_index_paths,
    ZARR_PATH, MASKED_ZARR_PATH,
    CORPUS_TIER2_DB_PATH, CORPUS_TIER2_MASKED_DB_PATH,
    discover_index_years,
)
from lib.eebo_faiss import EeboFaissIndex, multiscale_search
from lib.concept_resolve import resolve_concepts
from lib.eebo_logging import logger, setEmit
from lib.corpus_db import get_connection
from lib.embedding_cache import EmbeddingCache
from tier2_0_concept_events import ZarrEventLookup

from tier2_0_concept_events import ZarrEventLookup, sqlite3_connection

CONCEPT_DIR       = PLOT_DIR / "concept"
CONCEPT_NEIGH_DIR = PLOT_DIR / "concept_neighbours"
BFS_DIR           = PLOT_DIR / "bfs_global"
CLUSTER_DIR       = PLOT_DIR / "concept_clusters"

K = 25


# Move to a lib, this is also in T2
def load_all_year_indices(masked: bool, max_workers: int = 8) -> dict[int, dict[str, EeboFaissIndex]]:
    years = discover_index_years(masked)
    if not years:
        raise RuntimeError(
            f"No FAISS indices found for mode={'masked' if masked else 'unmasked'}. "
            f"Run build_indices.py first."
        )

    jobs = [
        (year, scale, path)
        for year in years
        for scale, path in faiss_index_paths(masked, year=year).items()
    ]

    index: dict[int, dict[str, EeboFaissIndex]] = {year: {} for year in years}

    logger.info(f"[tier3] loading {len(jobs)} FAISS indices ({max_workers} workers)")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_job = {
            pool.submit(EeboFaissIndex.load, path): (year, scale)
            for year, scale, path in jobs
        }
        for future in as_completed(future_to_job):
            year, scale = future_to_job[future]
            index[year][scale] = future.result()

    logger.info(f"[tier3] finished loading {len(jobs)} indices across {len(years)} years")
    return index

def backfill_missing_events_from_zarr(db_path, lookup, event_ids):
    sqlite_conn = sqlite3_connection(db_path)
    try:
        existing = {
            row[0]
            for row in sqlite_conn.execute("SELECT event_id FROM events")
        }
        missing_ids = list(set(event_ids) - existing)
        if not missing_ids:
            logger.info("[tier3] no missing SQLite events")
            return

        logger.info(f"[tier3] backfilling {len(missing_ids):,} missing events")

        missing_doc_ids = set()
        for eid in missing_ids:
            event = lookup.get_event(eid)
            if event is not None:
                missing_doc_ids.add(event["doc_id"])

        pg_conn = get_connection()
        try:
            with pg_conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT  doc_id, pub_year FROM pamphlet_corpus WHERE doc_id = ANY(%s)
                    """,
                    (list(missing_doc_ids),)
                )
                pub_year_map = {row[0]: row[1] for row in cur.fetchall()}
        finally:
            pg_conn.close()

        rows = []
        for eid in missing_ids:
            event = lookup.get_event(eid)
            if event is None:
                logger.warning(f"[tier3] missing Zarr event_id={eid}")
                continue
            rows.append((
                int(eid),
                "__derived__",
                int(event.get("vector_id", -1)),
                event.get("token"),
                event.get("doc_id"),
                pub_year_map.get(event.get("doc_id")),
                int(event.get("token_idx", -1)),
                int(event.get("window_id", -1)),
                int(event.get("window_token_pos", -1)),
            ))

        sqlite_conn.executemany(
            """
            INSERT OR IGNORE INTO events (
                event_id, concept, vector_id, token, doc_id, pub_year,
                token_idx, window_id, window_token_pos
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        sqlite_conn.commit()
        logger.info(f"[tier3] inserted {len(rows):,} rows")
    finally:
        sqlite_conn.close()


def build_depth_layers(
    index,    # dict[str, EeboFaissIndex] — local/medium/broad
    lookup,   # CHANGED: was emb_cache
    seed_ids,
    k: int = 25,
    max_depth: int = 2,
    max_nodes: int | None = None,
) -> dict[int, list[str]]:
    """
    BFS expansion via ANN search from seed_ids to max_depth hops, using
    fused (local/medium/broad -> RRF) neighbours at each hop instead of a
    single ensemble-embedding search.
    """
    seen = set(map(str, seed_ids))
    layers: dict[int, list[str]] = {0: list(seen)}
    frontier = list(seen)

    for depth in range(1, max_depth + 1):
        if not frontier:
            break
        if max_nodes is not None and sum(len(v) for v in layers.values()) >= max_nodes:
            break

        positions = np.array(
            [lookup.get_pos(int(eid)) for eid in frontier], dtype=np.int64
        )
        fused = multiscale_search(index, lookup, positions, top_n=k)

        ring: set[str] = set()
        for fused_neighbours in fused:
            for entry in fused_neighbours:
                sid = str(entry["event_id"])
                if sid not in seen:
                    ring.add(sid)

        seen |= ring
        layers[depth] = list(ring)
        frontier = list(ring)

    return layers


def depth_layers_to_flat(layers: dict[int, list[str]]) -> tuple[list[str], dict[str, int]]:
    """
    Flatten {depth: [ids]} into (all_ids, {id: depth}).
    Preserves depth-ascending order: seeds first, then ring 1, ring 2, ...
    """
    all_ids: list[str] = []
    depth_map: dict[str, int] = {}
    for depth in sorted(layers):
        for sid in layers[depth]:
            all_ids.append(sid)
            depth_map[sid] = depth
    return all_ids, depth_map


def fit_umap_local(X, event_ids, n_neighbors=15, min_dist=0.1):
    """
    X: (len(event_ids), D) float32 matrix, rows aligned to event_ids.
    Returns a dict keyed by str(event_id) to avoid JS/TS BigInt issues.
    """
    if not event_ids:
        return {}

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=42,
        metric="cosine",
    )

    logger.info(
        f"[tier3] UMAP input: shape={X.shape}, "
        f"unique_rows={np.unique(X, axis=0).shape[0]}, "
        f"rank≈{np.linalg.matrix_rank(X)}"
    )

    emb = reducer.fit_transform(X)
    return {str(eid): (float(emb[i, 0]), float(emb[i, 1]))
            for i, eid in enumerate(event_ids)}


def fit_umap_global(X, event_ids):
    """
    X: (len(event_ids), D) float32 matrix, rows aligned to event_ids.
    Returns a dict keyed by str(event_id) to avoid JS/TS BigInt issues.
    """
    logger.info(f"[tier3] fitting global PaCMAP on {len(event_ids):,} points")
    reducer = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=15,
        MN_ratio=0.5,
        FP_ratio=2.0,
        random_state=42,
    )

    logger.info(
        f"[tier3] UMAP input: shape={X.shape}, "
        f"unique_rows={np.unique(X, axis=0).shape[0]}, "
        f"rank≈{np.linalg.matrix_rank(X)}"
    )

    emb = reducer.fit_transform(X)

    return {str(eid): (float(emb[i, 0]), float(emb[i, 1]))
            for i, eid in enumerate(event_ids)}


def fit_leiden_on_fused_graph(
    event_ids,
    index,  # dict of scale -> EeboFaissIndex
    lookup,
    k: int = 25,
    resolution: float = 1.0,
    n_iterations: int = -1,  # -1 for default
):
    """
    Build a fused multi-scale graph using multiscale_search and run Leiden clustering.
    Returns cluster_labels list aligned to event_ids.
    """
    if len(event_ids) < 10:
        return [-1] * len(event_ids)

    logger.info(f"[tier3] Building fused graph for Leiden on {len(event_ids)} points")

    # Get positions for all points
    positions = np.array(
        [lookup.get_pos(int(eid)) for eid in event_ids], dtype=np.int64
    )

    # Use multiscale_search to get fused neighbors
    fused_neighbors = multiscale_search(index, lookup, positions, top_n=k)

    # Build adjacency list / edge list for igraph
    edge_list = []
    node_map = {str(eid): i for i, eid in enumerate(event_ids)}  # map str eid to index 0..N

    for i, fused_list in enumerate(fused_neighbors):
        src = i  # index in event_ids
        for entry in fused_list:
            tgt_eid = str(entry["event_id"])
            if tgt_eid in node_map:
                tgt = node_map[tgt_eid]
                if src != tgt:  # no self-loops
                    edge_list.append((src, tgt))

    if not edge_list:
        logger.warning("[tier3] No edges in fused graph")
        return [-1] * len(event_ids)

    # Create undirected graph with an explicit vertex count so vertex i
    # always corresponds to event_ids[i]. Graph.TupleList must NOT be used
    # here: it only creates vertices for ids that appear in edge_list (so
    # any event with zero fused neighbors in the sample silently vanishes,
    # shortening the label list below the length of event_ids), and it
    # assigns internal vertex ids in first-appearance order rather than by
    # the integer value itself (so even when nothing is dropped, vertex i
    # is not guaranteed to be event_ids[i] - labels end up scrambled).
    g = ig.Graph(n=len(event_ids), edges=edge_list, directed=False)

    # Run Leiden
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        n_iterations=n_iterations,
    )

    labels = partition.membership
    logger.info(f"[tier3] Leiden found {len(set(labels))} clusters (resolution={resolution})")

    return labels


def fit_cluster_local(
    X, event_ids,
    local_concept_coords=None,
    clustering_method="hdbscan", # TODO
    index=None,
    lookup=None,
    leiden_resolution=1.0,
):
    """
    Clustering pipeline:

    Supports 'hdbscan' (default) or 'leiden' on fused multi-scale graph.

    For Leiden:
        - Builds graph from multiscale_search fused neighbors
        - Runs leidenalg RBConfigurationVertexPartition

    X: (len(event_ids), D) float32 matrix, rows aligned to event_ids.
    local_concept_coords: optional {str(event_id) -> (x, y)} ...
    clustering_method: 'hdbscan' or 'leiden'
    index, lookup: required for 'leiden'

    Returns local_coords keyed by str(event_id).
    """
    if len(event_ids) < 10:
        return {str(eid): (0.0, 0.0) for eid in event_ids}, [-1] * len(event_ids)

    if clustering_method == "leiden":
        if index is None or lookup is None:
            logger.warning("[tier3] Leiden requires index and lookup; falling back to HDBSCAN")
            clustering_method = "hdbscan"
        else:
            labels = fit_leiden_on_fused_graph(
                event_ids, index, lookup, k=25, resolution=1.0
            )
            X_norm = None  # not needed for Leiden path
    else:
        # Original HDBSCAN path
        # normalize embeddings for cosine stability
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

        # 1. 5D UMAP reduction for clustering
        reducer_5d = umap.UMAP(
            n_components=5,
            n_neighbors=15,
            min_dist=0.0,
            metric='cosine',
            random_state=42,
        )

        logger.info(
            f"[tier3] UMAP input: shape={X.shape}, "
            f"unique_rows={np.unique(X, axis=0).shape[0]}, "
            f"rank≈{np.linalg.matrix_rank(X)}"
        )

        X_5d = reducer_5d.fit_transform(X)

        # 2. HDBSCAN on the 5D reduction
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(5, len(event_ids) // 200),
            min_samples=3,
            metric='euclidean',
            cluster_selection_method='eom',
            cluster_selection_epsilon=0.0,
        )
        labels = clusterer.fit_predict(X_5d).tolist()

    # 3. Build cluster structures + post-processing
    if clustering_method == "leiden":
        final_labels = labels[:]  # Leiden labels are 0-based integers
        # Optional: treat very small clusters as noise (-1)
        count = Counter(final_labels)
        min_size = max(5, len(event_ids) // 200)
        for i, lbl in enumerate(final_labels):
            if count[lbl] < min_size:
                final_labels[i] = -1
    else:
        # HDBSCAN path with centroid merge
        clusters = {}
        for i, cid in enumerate(labels):
            if cid == -1:
                continue
            clusters.setdefault(cid, []).append(i)

        # 4. Compute centroids (normalized)
        centroids = {}
        for cid, idxs in clusters.items():
            vecs = X_norm[idxs]
            centroid = vecs.mean(axis=0)
            centroid /= (np.linalg.norm(centroid) + 1e-12)
            centroids[cid] = centroid

        # 5. Merge near-duplicate clusters only (conservative threshold)
        def cosine(a, b):
            return float(np.dot(a, b))

        parent = {cid: cid for cid in clusters.keys()}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        cluster_ids = list(clusters.keys())

        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                a, b = cluster_ids[i], cluster_ids[j]
                sim = cosine(centroids[a], centroids[b])
                if sim > 0.97:  # near-duplicate merge only
                    union(a, b)

        merged = {}
        for cid in cluster_ids:
            root = find(cid)
            merged.setdefault(root, []).extend(clusters[cid])

        final_labels = [-1] * len(event_ids)
        for new_cid, idxs in enumerate(merged.values()):
            for idx in idxs:
                final_labels[idx] = new_cid

    # 6. 2D coords for visualization — reuse if already computed.
    if local_concept_coords is not None:
        local_coords = {str(eid): local_concept_coords[str(eid)] for eid in event_ids}
    else:
        reducer_2d = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42,
        )

        logger.info(
            f"[tier3] UMAP input: shape={X.shape}, "
            f"unique_rows={np.unique(X, axis=0).shape[0]}, "
            f"rank≈{np.linalg.matrix_rank(X)}"
        )

        emb_2d = reducer_2d.fit_transform(X)
        local_coords = {
            str(eid): (float(emb_2d[i, 0]), float(emb_2d[i, 1]))
            for i, eid in enumerate(event_ids)
        }

    return local_coords, final_labels


def compute_cluster_aggregates(
    event_ids,
    cluster_labels,
    lookup,
    local_coords,
    global_coords,
    local_bounds,
    global_bounds,
    top_n=25
):
    """
    Returns two things:
      1. aggregates: list of rows for concept_aggregate table (with cluster_id)
      2. cluster_info: list of dicts for concept_cluster_info table
    """
    from collections import defaultdict, Counter

    token_counters = defaultdict(Counter)
    doc_counters   = defaultdict(Counter)

    # Build counters per cluster
    for eid, cid in zip(event_ids, cluster_labels):
        if cid == -1:
            continue
        event = lookup.get_event(eid)
        token_counters[cid][event["token"]] += 1
        doc_counters[cid][event["doc_id"]] += 1

    aggregates = []
    cluster_info = []

    for cid in sorted(token_counters.keys()):
        # Top tokens & docs
        top_tokens = token_counters[cid].most_common(top_n)
        top_docs   = doc_counters[cid].most_common(top_n)

        # Aggregate rows for concept_aggregate
        for rank, (value, count) in enumerate(top_tokens, 1):
            aggregates.append((cid, 'token', rank, value, count))
        for rank, (value, count) in enumerate(top_docs, 1):
            aggregates.append((cid, 'doc', rank, value, count))

        # Centroid calculation
        cluster_idx = [i for i, c in enumerate(cluster_labels) if c == cid]
        if cluster_idx:
            nx_vals = [local_coords[event_ids[i]][0] for i in cluster_idx]
            ny_vals = [local_coords[event_ids[i]][1] for i in cluster_idx]
            gnx_vals = [global_coords[event_ids[i]][0] for i in cluster_idx]
            gny_vals = [global_coords[event_ids[i]][1] for i in cluster_idx]

            # raw centroids in projection space
            mean_x = float(np.mean(nx_vals))
            mean_y = float(np.mean(ny_vals))
            mean_gx = float(np.mean(gnx_vals))
            mean_gy = float(np.mean(gny_vals))

            # normalize into 0-1
            centroid_nx = (mean_x - local_bounds["minX"]) / (local_bounds["maxX"] - local_bounds["minX"])
            centroid_ny = (mean_y - local_bounds["minY"]) / (local_bounds["maxY"] - local_bounds["minY"])

            centroid_gnx = (mean_gx - global_bounds["minX"]) / (global_bounds["maxX"] - global_bounds["minX"])
            centroid_gny = (mean_gy - global_bounds["minY"]) / (global_bounds["maxY"] - global_bounds["minY"])

            cluster_info.append({
                'cluster_id': cid,
                'cluster_label': chr(65 + cid),
                'centroid_nx': float(centroid_nx),
                'centroid_ny': float(centroid_ny),
                'centroid_gnx': float(centroid_gnx),
                'centroid_gny': float(centroid_gny),
                'point_count': len(cluster_idx)
            })

    return aggregates, cluster_info


def compute_bounds_from_coords(coords):
    xs = np.array([c[0] for c in coords], dtype=np.float32)
    ys = np.array([c[1] for c in coords], dtype=np.float32)
    return {
        "minX": float(xs.min()), "maxX": float(xs.max()),
        "minY": float(ys.min()), "maxY": float(ys.max()),
    }


def add_padding(bounds, pad_ratio=0.02):
    width  = bounds["maxX"] - bounds["minX"]
    height = bounds["maxY"] - bounds["minY"]
    return {
        "minX": bounds["minX"] - width  * pad_ratio,
        "maxX": bounds["maxX"] + width  * pad_ratio,
        "minY": bounds["minY"] - height * pad_ratio,
        "maxY": bounds["maxY"] + height * pad_ratio,
    }


def assemble_points(event_ids, local_coords, global_coords,
                    local_bounds, global_bounds,
                    extra_fields: dict[str, dict] | None = None):
    """
    Assemble output point dicts for JSON serialisation.

    All coord dicts (local_coords, global_coords) must be keyed by
    str(event_id).  extra_fields must also be str(event_id)-keyed.
    event_id is always written as a JSON string so JS/TS never
    encounters a value that would be parsed as BigInt.
    """
    lx0, lx1 = local_bounds["minX"],  local_bounds["maxX"]
    ly0, ly1 = local_bounds["minY"],  local_bounds["maxY"]
    gx0, gx1 = global_bounds["minX"], global_bounds["maxX"]
    gy0, gy1 = global_bounds["minY"], global_bounds["maxY"]

    ldx = (lx1 - lx0) + 1e-12
    ldy = (ly1 - ly0) + 1e-12
    gdx = (gx1 - gx0) + 1e-12
    gdy = (gy1 - gy0) + 1e-12

    points = []
    for eid in event_ids:
        sid = str(eid)          # canonical string key used for all lookups
        lx, ly = local_coords[sid]
        gx, gy = global_coords[sid]
        pt = {
            "event_id": sid,    # always a string in JSON output
            "x":   lx,
            "y":   ly,
            "nx":  (lx - lx0) / ldx,
            "ny":  (ly - ly0) / ldy,
            "gx":  gx,
            "gy":  gy,
            "gnx": (gx - gx0) / gdx,
            "gny": (gy - gy0) / gdy,
        }
        if extra_fields and sid in extra_fields:
            pt.update(extra_fields[sid])
        points.append(pt)
    return points


def write_json(path, payload):
    logger.warning("[tier3] Not writing JSON, use SQLite")
    return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"[tier3] Wrote {path}")


def write_projections_to_sqlite(
    db_path,
    concept_name,
    event_ids,
    local_coords,
    global_coords,
    local_bounds,
    global_bounds,
    cluster_labels=None,
    target="events",      # "events" | "neighbours"
    depth_map=None,
    lookup=None,
    write_coords=True,
):
    logger.info(f"[tier3] write_projections_to_sqlite target={target} concept={concept_name} write_coords={write_coords}")
    con = sqlite3_connection(db_path)

    con.execute("""
        CREATE TEMP TABLE IF NOT EXISTS _proj_update (
            event_id      INTEGER PRIMARY KEY,
            nx            REAL, ny REAL,
            gnx           REAL, gny REAL,
            cluster_id    INTEGER,
            depth         INTEGER
        )
    """)
    con.execute("DELETE FROM _proj_update")

    # Normalize into local_bounds / global_bounds...
    lx0, lx1 = local_bounds["minX"],  local_bounds["maxX"]
    ly0, ly1 = local_bounds["minY"],  local_bounds["maxY"]
    gx0, gx1 = global_bounds["minX"], global_bounds["maxX"]
    gy0, gy1 = global_bounds["minY"], global_bounds["maxY"]

    ldx = (lx1 - lx0) + 1e-12
    ldy = (ly1 - ly0) + 1e-12
    gdx = (gx1 - gx0) + 1e-12
    gdy = (gy1 - gy0) + 1e-12

    data = []
    for i, eid in enumerate(event_ids):
        sid = str(eid)
        if sid not in local_coords or sid not in global_coords:
            continue
        lx, ly = local_coords[sid]
        gx, gy = global_coords[sid]
        data.append((
            int(eid),
            float((lx - lx0) / ldx), float((ly - ly0) / ldy),
            float((gx - gx0) / gdx), float((gy - gy0) / gdy),
            cluster_labels[i] if cluster_labels is not None else None,
            depth_map.get(sid) if depth_map is not None else None,
        ))

    if data:
        con.executemany(
            "INSERT OR IGNORE INTO _proj_update VALUES (?,?,?,?,?,?,?)",
            data
        )

    # Write point data
    if target == "events":
        if write_coords:
            con.execute("""
                UPDATE events SET
                    nx            = _proj_update.nx,
                    ny            = _proj_update.ny,
                    gnx           = _proj_update.gnx,
                    gny           = _proj_update.gny,
                    cluster_id    = _proj_update.cluster_id
                FROM _proj_update
                WHERE events.event_id = _proj_update.event_id
            """)
        else:
            con.execute("""
                UPDATE events SET cluster_id    = _proj_update.cluster_id
                FROM _proj_update
                WHERE events.event_id = _proj_update.event_id
            """)
    else:
        con.execute("""
            UPDATE neighbours SET
                nx  = _proj_update.nx,
                ny  = _proj_update.ny,
                gnx = _proj_update.gnx,
                gny = _proj_update.gny
            FROM _proj_update
            WHERE neighbours.neighbour_event_id = _proj_update.event_id
        """)

    # Cluster aggregates & centroids
    if target == "events" and cluster_labels is not None and lookup is not None:
        aggregates = []
        cluster_info = []

        if any(c != -1 for c in cluster_labels):
            aggregates, cluster_info = compute_cluster_aggregates(
                event_ids,
                cluster_labels,
                lookup,
                local_coords,
                global_coords,
                local_bounds,
                global_bounds,
                top_n=25
            )
        else:
            logger.info(f"[tier3] No valid clusters for {concept_name}")

        # concept_aggregate
        con.executemany("""
            INSERT INTO concept_aggregate
            (concept, cluster_id, kind, rank, value, count)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [
            (concept_name, cid, kind, rank, value, count)
            for cid, kind, rank, value, count in aggregates
        ])

        # concept_cluster_info
        con.executemany("""
            INSERT OR REPLACE INTO concept_cluster_info
            (concept, cluster_id, cluster_label, centroid_nx, centroid_ny,
             centroid_gnx, centroid_gny, point_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            (concept_name,
             info['cluster_id'],
             info['cluster_label'],
             info['centroid_nx'],
             info['centroid_ny'],
             info['centroid_gnx'],
             info['centroid_gny'],
             info['point_count'])
            for info in cluster_info
        ])

    con.commit()
    con.close()
    logger.info(f"[tier3] wrote projections + cluster data for {concept_name}")


def run_tier3_core(
    *,
    db_path,
    index,
    lookup,
    concept=None,
    false_positives=None,
    mode="full",
    use_concatenated_clustering=True,
    skip_global_bfs=False,
    emit=None,
    leiden_resolution=leiden_resolution
):
    logger.info("[tier3 run_tier3_core] Enter")

    lookup.attach_index(index)

    false_positives = false_positives or []
    emb_cache = EmbeddingCache(lookup)

    all_concept_events = []
    buffered_concept       = []   # (path, meta, event_ids, local_coords)
    buffered_concept_neigh = []   # (path, meta, event_ids, local_coords, depth_map)
    buffered_clusters      = []   # (path, meta, event_ids, local_coords, cluster_labels)
    all_event_ids = set()

    #
    # Pass 1 — concept + clustering inputs
    #
    if mode == "full":
        for concept_name, concept_def in resolve_concepts(concept=concept, false_positives=false_positives):
            logger.info(f"[tier3] processing concept={concept_name}")
            if emit:
                emit("concept_start", {"concept": concept_name})

            seed_ids       = list(lookup.iter_matching_event_ids(set(concept_def["forms"])))
            seed_ids       = [str(eid) for eid in seed_ids]
            concept_sample = seed_ids # [:1000]

            if not concept_sample:
                logger.warning(f"[tier3] no events found for concept={concept_name}, skipping")
                if emit:
                    emit("concept_done", {"concept": concept_name})
                continue

            # one embedding fetch for the concept sample, reused below
            X_concept = emb_cache.matrix(concept_sample)
            local_concept = fit_umap_local(X_concept, concept_sample)

            buffered_concept.append((
                CONCEPT_DIR / f"{concept_name}.json",
                {"type": "concept", "concept": concept_name},
                concept_sample,
                local_concept,
            ))
            all_event_ids.update(concept_sample)

            # depth=2 neighbour expansion; seed ring is depth 0
            neigh_layers = build_depth_layers(
                index, lookup, concept_sample, k=K, max_depth=2, max_nodes=3000
            )
            neigh_ids, neigh_depth_map = depth_layers_to_flat(neigh_layers)
            X_neigh     = emb_cache.matrix(neigh_ids)
            local_neigh = fit_umap_local(X_neigh, neigh_ids)

            buffered_concept_neigh.append((
                CONCEPT_NEIGH_DIR / f"{concept_name}.json",
                {"type": "concept_neighbours", "concept": concept_name},
                neigh_ids,
                local_neigh,
                neigh_depth_map,
            ))
            all_event_ids.update(neigh_ids)

            logger.info(f"[tier3] clustering {concept_name} ({len(concept_sample)} points)")

            X_cluster = lookup.get_concatenated_embeddings(concept_sample) if use_concatenated_clustering else X_concept

            cluster_local_coords, cluster_labels = fit_cluster_local(
                X_cluster, concept_sample, local_concept_coords=None,
                clustering_method="leiden", index=index, lookup=lookup,
                leiden_resolution=leiden_resolution,
            )

            buffered_clusters.append((
                CLUSTER_DIR / f"{concept_name}.json",
                {"type": "concept_clusters", "concept": concept_name},
                concept_sample,
                cluster_local_coords,
                cluster_labels,
            ))

            all_concept_events.extend(concept_sample)
            if emit:
                emit("concept_done", {"concept": concept_name})

    else:
        logger.info("[tier3] clustering-only mode (loading from sqlite)")
        con = sqlite3_connection(db_path)
        concept_names = [
            row[0] for row in
            con.execute("SELECT DISTINCT concept FROM events WHERE concept != '__derived__'")
        ]
        con.close()

        for concept_name in concept_names:
            # load event_ids from sqlite rather than JSON
            con = sqlite3_connection(db_path)
            rows = con.execute(
                "SELECT event_id FROM events WHERE concept = ?", (concept_name,)
            ).fetchall()
            con.close()

            concept_sample = [str(row[0]) for row in rows]
            if not concept_sample:
                continue

            logger.info(f"[tier3] clustering {concept_name} ({len(concept_sample)})")
            if emit:
                emit("concept_start", {"concept": concept_name})

            X_concept = emb_cache.matrix(concept_sample)
            X_cluster = lookup.get_concatenated_embeddings(concept_sample) if use_concatenated_clustering else X_concept

            cluster_local_coords, cluster_labels = fit_cluster_local(
                X_cluster, concept_sample, local_concept_coords=None,
                clustering_method="leiden", index=index, lookup=lookup
            )

            buffered_clusters.append((
                None,
                {"type": "concept_clusters", "concept": concept_name},
                concept_sample,
                cluster_local_coords,
                cluster_labels,
            ))
            all_concept_events.extend(concept_sample)

            if emit:
                emit("concept_done", {"concept": concept_name})

    #
    # BFS global expansion (full mode only)
    #
    if mode == "full" and not skip_global_bfs:
        if emit:
            emit("bfs_start", {})
        logger.info(f"[tier3] all_concept_events count: {len(all_concept_events)}")
        bfs_layers = build_depth_layers(
            index, lookup, all_concept_events, k=K, max_depth=2, max_nodes=5000
        )
        bfs_ids, bfs_depth_map = depth_layers_to_flat(bfs_layers)
        X_bfs     = emb_cache.matrix(bfs_ids) if bfs_ids else None
        local_bfs = fit_umap_local(X_bfs, bfs_ids) if bfs_ids else {}
        all_event_ids.update(bfs_ids)
    else:
        bfs_ids       = []
        bfs_depth_map = {}
        local_bfs     = {}

    # Backfill safety
    if all_event_ids:
        backfill_missing_events_from_zarr(db_path, lookup, all_event_ids)

    # Global projection
    if emit:
        emit("global_projection_start", {"n_events": len(all_event_ids)})

    if all_event_ids:
        all_event_ids_list   = list(all_event_ids)
        X_global             = emb_cache.matrix(all_event_ids_list)
        global_coords        = fit_umap_global(X_global, all_event_ids_list)
        global_bounds        = compute_bounds_from_coords(list(global_coords.values()))
        global_bounds_padded = add_padding(global_bounds)
    else:
        logger.warning("[tier3] no event ids — skipping global projection")
        global_coords        = {}
        global_bounds        = {"minX": 0, "maxX": 0, "minY": 0, "maxY": 0}
        global_bounds_padded = global_bounds

    logger.info(f"[tier3] global bounds (padded): {global_bounds_padded}")

    #
    # Pass 3a — concept seed projections -> events table
    #
    for path, meta, event_ids, local_coords in buffered_concept:
        concept_name = meta["concept"]
        local_bounds = add_padding(
            compute_bounds_from_coords([local_coords[str(e)] for e in event_ids])
        )
        write_projections_to_sqlite(
            db_path         = db_path,
            concept_name    = concept_name,
            event_ids       = event_ids,
            local_coords    = local_coords,
            global_coords   = global_coords,
            local_bounds    = local_bounds,
            global_bounds   = global_bounds_padded,
            target          = "events",
            lookup          = lookup,
            write_coords    = False,
        )

    #
    # Pass 3b — neighbour projections -> neighbours table
    #
    for path, meta, event_ids, local_coords, depth_map in buffered_concept_neigh:
        concept_name = meta["concept"]
        local_bounds = add_padding(
            compute_bounds_from_coords([local_coords[str(e)] for e in event_ids])
        )
        write_projections_to_sqlite(
            db_path         = db_path,
            concept_name    = concept_name,
            event_ids       = event_ids,
            local_coords    = local_coords,
            global_coords   = global_coords,
            local_bounds    = local_bounds,
            global_bounds   = global_bounds_padded,
            depth_map       = depth_map,
            target          = "neighbours",
        )

    #
    # Pass 3c — cluster labels + coords -> events table
    #
    for path, meta, event_ids, local_coords, cluster_labels in buffered_clusters:
        concept_name = meta["concept"]
        local_bounds = add_padding(
            compute_bounds_from_coords([local_coords[str(e)] for e in event_ids])
        )
        write_projections_to_sqlite(
            db_path         = db_path,
            concept_name    = concept_name,
            event_ids       = event_ids,
            local_coords    = local_coords,
            global_coords   = global_coords,
            local_bounds    = local_bounds,
            global_bounds   = global_bounds_padded,
            cluster_labels  = cluster_labels,
            target          = "events",
            lookup          = lookup,
        )

    if emit:
        emit("tier3_done", {})
    logger.info("[tier3 run_tier3_core] Complete")


def run_tier3_service(
    *,
    db_path,
    index,
    lookup,
    concept=None,
    false_positives=None,
    mode="full",
    use_concatenated_clustering=True,
    skip_global_bfs=False,
    emit=None,
    leiden_resolution=leiden_resolution,
):
    logger = setEmit(
        emit,
        "[tier3]",
        concept,
    )
    logger.info("[tier3 run_tier3_service] Enter")

    return run_tier3_core(
        db_path                     = db_path,
        index                       = index,
        lookup                      = lookup,
        concept                     = concept,
        false_positives             = false_positives,
        mode                        = mode,
        use_concatenated_clustering = use_concatenated_clustering,
        skip_global_bfs             = skip_global_bfs,
        emit                        = emit,
        leiden_resolution           = leiden_resolution,
    )


def main():
    logger.info("[tier3 main] Enter")

    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", type=str, default=None)
    parser.add_argument("--mode", type=str, default="full", choices=["full", "clustering"])
    parser.add_argument("--false_positives", type=str, nargs="*", default=[])
    parser.add_argument("--mask", action="store_true", help="Use masked data")
    parser.add_argument("--no-ensemble", action="store_true", help="Cluster by concatinating ensemble vectors (legacy)")
    parser.add_argument("--skip-bfs", action="store_true", help="Skip the global BFS expansion pass (full mode only); useful for fast single-concept runs")
    parser.add_argument( "--leiden-resolution", type=float, default=1.0,
        help=( "Leiden clustering resolution. Higher values produce more, smaller clusters; lower values produce fewer, larger clusters." ),
    )

    args = parser.parse_args()

    if args.mask:
        zarr_path = MASKED_ZARR_PATH
        masked = True
        db_path = CORPUS_TIER2_MASKED_DB_PATH
        use_concatenated_clustering = True
    else:
        zarr_path = ZARR_PATH
        masked = False
        db_path = CORPUS_TIER2_DB_PATH
        use_concatenated_clustering = False

    lookup = ZarrEventLookup(zarr_path)
    index  = load_all_year_indices(masked)
    lookup.attach_index(index)

    if args.no_ensemble:
        use_concatenated_clustering = True

    logger.info(
        f"[Tier3.main] loading index+lookup, mode={'masked' if args.mask else 'unmasked'}, "
        f"clustering={'concatenated' if use_concatenated_clustering else 'ensemble'}"
    )

    run_tier3_core(
        db_path                      = db_path,
        index                        = index,
        lookup                       = lookup,
        concept                      = args.concept,
        false_positives              = args.false_positives,
        mode                         = args.mode,
        use_concatenated_clustering  = use_concatenated_clustering,
        skip_global_bfs              = args.skip_bfs,
        emit                         = None,
        leiden_resolution            = leiden_resolution,
    )

    logger.info("[tier3] complete")


if __name__ == "__main__":
    main()
