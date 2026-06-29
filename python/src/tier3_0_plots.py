#!/usr/bin/env python

"""
Outputs:
    indexes/
        umap/
            concept/
                PREROGATIVE.json
            concept_neighbours/
                PREROGATIVE.json
            bfs_global/
                global.json
            concept_clusters/
                PREROGATIVE.json
            manifest.json

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
    cluster_id      — integer HDBSCAN cluster id (-1 = noise)
    cluster_label   — letter label A, B, C... (null for noise)

Note: all event_id values are serialised as JSON strings to avoid JS/TS
BigInt issues with integers exceeding Number.MAX_SAFE_INTEGER (2^53 - 1).
"""

import argparse
import json
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import umap
import hdbscan
import pacmap
import sqlite3

from lib.eebo_config import ZARR_ROOT, FAISS_TIER1_INDEX, PLOT_DIR, CORPUS_TIER2_DB_PATH
from lib.eebo_faiss import EeboFaissIndex
from lib.concept_resolve import resolve_concepts
from lib.eebo_logging import logger, setEmit
from lib.eebo_db import get_connection

from tier2_0_concept_events import ZarrEventLookup

CONCEPT_DIR       = PLOT_DIR / "concept"
CONCEPT_NEIGH_DIR = PLOT_DIR / "concept_neighbours"
BFS_DIR           = PLOT_DIR / "bfs_global"
CLUSTER_DIR       = PLOT_DIR / "concept_clusters"

K = 25


class EmbeddingCache:
    """
    Single point of access for embeddings.

    Fetches each event's embedding from the lookup at most once and keeps
    it in memory as a row in a contiguous float32 matrix, so repeated
    np.stack([lookup.get_event(eid)["embedding"] for eid in ids]) calls
    across fit_umap_local / fit_cluster_local / build_depth_layers
    collapse into cheap array slicing.

    Does not change any numeric results — same embeddings, same dtype,
    same ordering semantics (callers still build their own X via
    `matrix(event_ids)`, which preserves the order of `event_ids`).
    """

    def __init__(self, lookup):
        self._lookup = lookup
        self._row_of = {}      # event_id -> row index in _mat
        self._mat = None       # (N, D) float32, grows as needed
        self._cap = 0

    def _ensure_capacity(self, extra):
        needed = len(self._row_of) + extra
        if self._mat is None:
            cap = max(needed, 1024)
            self._mat = np.empty((cap, self._dim), dtype=np.float32)
            self._cap = cap
            return
        if needed > self._cap:
            new_cap = max(needed, self._cap * 2)
            new_mat = np.empty((new_cap, self._mat.shape[1]), dtype=np.float32)
            new_mat[: self._mat.shape[0]] = self._mat
            self._mat = new_mat
            self._cap = new_cap

    def _fetch(self, eid):
        emb = self._lookup.get_event(eid)["embedding"]
        return np.asarray(emb, dtype=np.float32)

    def warm(self, event_ids):
        """Fetch and cache any embeddings not already cached."""
        missing = [eid for eid in event_ids if eid not in self._row_of]
        if not missing:
            return

        if self._mat is None:
            first = self._fetch(missing[0])
            self._dim = first.shape[0]
            self._ensure_capacity(len(missing))
            row = len(self._row_of)
            self._mat[row] = first
            self._row_of[missing[0]] = row
            missing = missing[1:]

        if missing:
            self._ensure_capacity(len(missing))
            for eid in missing:
                row = len(self._row_of)
                self._mat[row] = self._fetch(eid)
                self._row_of[eid] = row

    def matrix(self, event_ids):
        """
        Return an (len(event_ids), D) float32 array with rows in the same
        order as event_ids, fetching/caching as needed.
        """
        if not event_ids:
            raise ValueError("[EmbeddingCache] matrix() called with empty event_ids")
        self.warm(event_ids)
        idx = np.fromiter((self._row_of[eid] for eid in event_ids), dtype=np.int64, count=len(event_ids))
        return self._mat[idx]

    def vector(self, event_id):
        self.warm([event_id])
        return self._mat[self._row_of[event_id]]


def backfill_missing_events_from_zarr(db_path, lookup, event_ids):
    sqlite_conn = sqlite3.connect(db_path)
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
    index,
    emb_cache,
    seed_ids,
    k: int = 25,
    max_depth: int = 2,
    max_nodes: int | None = None,
) -> dict[int, list[str]]:
    """
    BFS expansion via ANN search from seed_ids to max_depth hops.

    Each ring contains only nodes first reached at that depth — rings are
    guaranteed disjoint by the seen set.  The frontier for depth d is
    exactly the ring discovered at depth d-1, so each ANN search covers
    only new nodes rather than the entire accumulated set.

    Args:
        index      — FAISS index supporting .search(vecs, k)
        emb_cache  — EmbeddingCache
        seed_ids   — iterable of seed event ids (str or int)
        k          — number of ANN neighbours per query vector
        max_depth  — number of BFS hops (default 2)
        max_nodes  — optional total node cap across all depths

    Returns:
        {depth: [str_event_id, ...]}  for depth in 0..max_depth
        Depth 0 is always exactly the (deduplicated) seed set.
    """
    seen = set(map(str, seed_ids))
    layers: dict[int, list[str]] = {0: list(seen)}
    frontier = list(seen)

    for depth in range(1, max_depth + 1):
        if not frontier:
            break
        if max_nodes is not None and sum(len(v) for v in layers.values()) >= max_nodes:
            break

        X = emb_cache.matrix(frontier)
        _, nn = index.search(X, k)

        ring: set[str] = set()
        for row in nn:
            for nid in row:
                if nid != -1:
                    sid = str(int(nid))
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
    emb = reducer.fit_transform(X)
    return {str(eid): (float(emb[i, 0]), float(emb[i, 1]))
            for i, eid in enumerate(event_ids)}


def fit_cluster_local(X, event_ids, local_concept_coords=None):
    """
    Clustering pipeline:

    1. 5-D UMAP reduction of the embeddings for HDBSCAN (restores the
       structure-finding behaviour of the original 5D pipeline; raw
       high-D cosine embeddings are too uniform under euclidean HDBSCAN
       and tend to collapse to one cluster + noise).
    2. HDBSCAN on the 5D reduction.
    3. Conservative semantic merge of near-duplicate clusters only
       (sim > 0.97), so genuinely distinct senses stay separate.
    4. Local 2D coords for display: reuse `local_concept_coords` if the
       caller already computed a 2D UMAP for this same point set (e.g.
       the concept's own local projection), instead of fitting a second,
       near-identical 2D UMAP.

    X: (len(event_ids), D) float32 matrix, rows aligned to event_ids.
    local_concept_coords: optional {str(event_id) -> (x, y)} from an existing
        2D UMAP fit on the same event_ids (same params: n_neighbors=15,
        min_dist=0.1, metric='cosine'). If provided, skips the redundant
        2D UMAP fit inside this function.

    Returns local_coords keyed by str(event_id).
    """
    if len(event_ids) < 10:
        return {str(eid): (0.0, 0.0) for eid in event_ids}, [-1] * len(event_ids)

    # normalize embeddings for cosine stability (used for clustering input
    # and for centroid similarity in the merge step)
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    # 1. 5D UMAP reduction for clustering
    reducer_5d = umap.UMAP(
        n_components=5,
        n_neighbors=15,
        min_dist=0.0,
        metric='cosine',
        random_state=42,
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

    # 3. Build cluster structures
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
    # local_concept_coords is expected to be str(event_id)-keyed.
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
        emb_2d = reducer_2d.fit_transform(X)
        local_coords = {
            str(eid): (float(emb_2d[i, 0]), float(emb_2d[i, 1]))
            for i, eid in enumerate(event_ids)
        }

    return local_coords, final_labels


def compute_cluster_aggregates(event_ids, cluster_labels, lookup, top_n=25):
    token_counters = defaultdict(Counter)
    doc_counters   = defaultdict(Counter)

    for eid, cid in zip(event_ids, cluster_labels):
        if cid == -1:
            continue
        event = lookup.get_event(eid)
        token_counters[cid][event["token"]] += 1
        doc_counters[cid][event["doc_id"]]  += 1

    return {
        str(cid): {
            "top_tokens": token_counters[cid].most_common(top_n),
            "top_docs":   doc_counters[cid].most_common(top_n),
        }
        for cid in token_counters
    }


def compute_bounds_from_coords(coords):
    xs = np.array([c[0] for c in coords], dtype=np.float32)
    ys = np.array([c[1] for c in coords], dtype=np.float32)
    return {
        "minX": float(xs.min()), "maxX": float(xs.max()),
        "minY": float(ys.min()), "maxY": float(ys.max()),
    }


def add_padding(bounds, pad_ratio=0.1):
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"[tier3] Wrote {path}")


def run_tier3_core(
    *,
    db_path,
    index,
    lookup,
    concept=None,
    false_positives=None,
    mode="full",
    emit=None,
):
    logger.info("[tier3 run_tier3_core] Enter")

    false_positives = false_positives or []
    emb_cache = EmbeddingCache(lookup)

    manifest = {"concepts": [], "global": None, "globalBounds": None}
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
                index, emb_cache, concept_sample, k=K, max_depth=2, max_nodes=3000
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
            # reuse X_concept and the 2D local_concept coords — avoids a
            # second embedding fetch and a second near-identical 2D UMAP fit
            cluster_local_coords, cluster_labels = fit_cluster_local(
                X_concept, concept_sample, local_concept_coords=local_concept
            )

            buffered_clusters.append((
                CLUSTER_DIR / f"{concept_name}.json",
                {"type": "concept_clusters", "concept": concept_name},
                concept_sample,
                cluster_local_coords,
                cluster_labels,
            ))

            all_concept_events.extend(concept_sample)

            manifest["concepts"].append({
                "name":               concept_name,
                "concept":            f"/umap/concept/{concept_name}.json",
                "concept_neighbours": f"/umap/concept_neighbours/{concept_name}.json",
                "concept_clusters":   f"/umap/concept_clusters/{concept_name}.json",
            })

            if emit:
                emit("concept_done", {"concept": concept_name})

    else:
        logger.info("[tier3] clustering-only mode (loading concept outputs)")

        for concept_file in CONCEPT_DIR.glob("*.json"):
            concept_name = concept_file.stem

            with open(concept_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # event_id is a string in the JSON; keep it that way
            concept_sample = [p["event_id"] for p in data["points"]]

            logger.info(f"[tier3] clustering {concept_name} ({len(concept_sample)})")
            if emit:
                emit("concept_start", {"concept": concept_name})

            X_concept = emb_cache.matrix(concept_sample)

            # In clustering-only mode we don't have a freshly computed
            # local_concept 2D projection in hand (it was written to disk
            # by a previous "full" run), so fit_cluster_local computes its
            # own 2D UMAP for display coords here.
            cluster_local_coords, cluster_labels = fit_cluster_local(X_concept, concept_sample)

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

    #
    # BFS global expansion (full mode only)
    #
    if mode == "full":
        if emit:
            emit("bfs_start", {})
        logger.info(f"[tier3] all_concept_events count: {len(all_concept_events)}")
        bfs_layers = build_depth_layers(
            index, emb_cache, all_concept_events, k=K, max_depth=2, max_nodes=5000
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
    # Pass 3a — concept outputs (seeds; no depth field needed, implicitly 0)
    #
    for path, meta, event_ids, local_coords in buffered_concept:
        local_bounds = add_padding(
            compute_bounds_from_coords([local_coords[str(e)] for e in event_ids])
        )
        points = assemble_points(
            event_ids, local_coords, global_coords,
            local_bounds, global_bounds_padded,
        )
        write_json(path, {
            **meta,
            "bounds":       local_bounds,
            "globalBounds": global_bounds_padded,
            "points":       points,
        })

    #
    # Pass 3b — concept_neighbours outputs (with depth field per point)
    #
    for path, meta, event_ids, local_coords, depth_map in buffered_concept_neigh:
        local_bounds = add_padding(
            compute_bounds_from_coords([local_coords[str(e)] for e in event_ids])
        )
        extra = {str(eid): {"depth": depth_map[str(eid)]} for eid in event_ids}
        points = assemble_points(
            event_ids, local_coords, global_coords,
            local_bounds, global_bounds_padded,
            extra_fields=extra,
        )
        write_json(path, {
            **meta,
            "max_depth":    2,
            "bounds":       local_bounds,
            "globalBounds": global_bounds_padded,
            "points":       points,
        })

    #
    # Pass 3c — cluster outputs
    #
    for path, meta, event_ids, local_coords, cluster_labels in buffered_clusters:
        concept_name = meta["concept"]
        unique_clusters = sorted(c for c in set(cluster_labels) if c != -1)
        label_map = {cid: chr(65 + i) for i, cid in enumerate(unique_clusters)}

        extra = {
            str(eid): {
                "cluster_id":    int(cluster_labels[i]),
                "cluster_label": label_map.get(cluster_labels[i]),
            }
            for i, eid in enumerate(event_ids)
        }

        local_bounds = add_padding(
            compute_bounds_from_coords([local_coords[str(e)] for e in event_ids])
        )

        write_projections_to_sqlite(
            db_path         = db_path,
            concept_name    = concept_name,
            points          = None,
            local_coords    = local_coords,
            global_coords   = global_coords,
            local_bounds    = local_bounds,
            global_bounds   = global_bounds_padded,
            cluster_labels  = cluster_labels,
            event_ids       = event_ids,
        )

        points = assemble_points(
            event_ids, local_coords, global_coords,
            local_bounds, global_bounds_padded,
            extra_fields=extra,
        )
        aggregates = compute_cluster_aggregates(event_ids, cluster_labels, lookup)

        write_json(path, {
            **meta,
            "generated_at": datetime.now().isoformat(),
            "n_events":     len(event_ids),
            "bounds":       local_bounds,
            "globalBounds": global_bounds_padded,
            "clusters": {
                "label_map":  {str(k): v for k, v in label_map.items()},
                "aggregates": aggregates,
            },
            "points": points,
        })

    #
    # BFS global output (with depth field per point)
    #
    if bfs_ids:
        local_bounds_bfs = add_padding(
            compute_bounds_from_coords([local_bfs[str(e)] for e in bfs_ids])
        )
        extra_bfs = {str(eid): {"depth": bfs_depth_map[str(eid)]} for eid in bfs_ids}
        points_bfs = assemble_points(
            bfs_ids, local_bfs, global_coords,
            local_bounds_bfs, global_bounds_padded,
            extra_fields=extra_bfs,
        )
        write_json(
            BFS_DIR / "global.json",
            {
                "type":         "bfs_global",
                "bounds":       local_bounds_bfs,
                "globalBounds": global_bounds_padded,
                "max_depth":    2,
                "k":            K,
                "points":       points_bfs,
            }
        )
        manifest["global"] = "/umap/bfs_global/global.json"
    else:
        manifest["global"] = None

    manifest["globalBounds"] = global_bounds_padded
    write_json(PLOT_DIR / "manifest.json", manifest)

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
    emit=None,
):
    logger = setEmit(
        emit,
        "[tier3]",
        concept,
    )
    logger.info("[tier3 run_tier3_service] Enter")

    return run_tier3_core(
        db_path         = db_path,
        index           = index,
        lookup          = lookup,
        concept         = concept,
        false_positives = false_positives,
        mode            = mode,
        emit            = emit,
    )


def write_projections_to_sqlite(db_path, concept_name, points, local_coords, global_coords, local_bounds, global_bounds, cluster_labels=None, event_ids=None):
    """
    Write projection coordinates and cluster assignments to SQLite.
    points: list of assembled point dicts from assemble_points
    """
    con = sqlite3.connect(db_path)

    # Update events table (seeds with cluster assignments)
    if cluster_labels and event_ids:
        label_map = {
            str(eid): cluster_labels[i]
            for i, eid in enumerate(event_ids)
        }
        con.executemany(
            """UPDATE events
               SET local_x = ?, local_y = ?, global_x = ?, global_y = ?,
                   cluster_id = ?, cluster_label = ?
               WHERE event_id = ?""",
            [
                (
                    local_coords[str(eid)][0],
                    local_coords[str(eid)][1],
                    global_coords[str(eid)][0],
                    global_coords[str(eid)][1],
                    int(label_map[str(eid)]),
                    chr(65 + int(label_map[str(eid)])) if label_map[str(eid)] != -1 else None,
                    int(eid),
                )
                for eid in event_ids
                if str(eid) in local_coords and str(eid) in global_coords
            ]
        )

    # Update neighbours table (no cluster assignments)
    con.executemany(
        """UPDATE neighbours
           SET local_x = ?, local_y = ?, global_x = ?, global_y = ?
           WHERE neighbour_event_id = ?""",
        [
            (
                local_coords[str(eid)][0],
                local_coords[str(eid)][1],
                global_coords[str(eid)][0],
                global_coords[str(eid)][1],
                int(eid),
            )
            for eid in (event_ids or [])
            if str(eid) in local_coords and str(eid) in global_coords
        ]
    )

    # Update concept_projection_bounds
    con.execute(
        """INSERT OR REPLACE INTO concept_projection_bounds (
               concept,
               local_min_x, local_max_x, local_min_y, local_max_y,
               global_min_x, global_max_x, global_min_y, global_max_y
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            concept_name,
            local_bounds["minX"],  local_bounds["maxX"],
            local_bounds["minY"],  local_bounds["maxY"],
            global_bounds["minX"], global_bounds["maxX"],
            global_bounds["minY"], global_bounds["maxY"],
        )
    )

    con.commit()
    con.close()


def main():
    logger.info("[tier3 main] Enter")

    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", type=str, default=None)
    parser.add_argument("--mode", type=str, default="full", choices=["full", "clustering"])
    parser.add_argument("--false_positives", type=str, nargs="*", default=[])

    args = parser.parse_args()

    logger.info("[tier3] loading index + lookup")

    lookup = ZarrEventLookup(ZARR_ROOT / "tier1")
    index  = EeboFaissIndex.load(FAISS_TIER1_INDEX)

    run_tier3_core(
        db_path         = CORPUS_TIER2_DB_PATH,
        index           = index,
        lookup          = lookup,
        concept         = args.concept,
        false_positives = args.false_positives,
        mode            = args.mode,
        emit            = None,
    )

    logger.info("[tier3] complete")


if __name__ == "__main__":
    main()
