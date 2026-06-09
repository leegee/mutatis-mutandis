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

concept_clusters files additionally carry:
    cluster_id      — integer HDBSCAN cluster id (-1 = noise)
    cluster_label   — letter label A, B, C... (null for noise)
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

from lib.eebo_config import ZARR_ROOT, FAISS_TIER1_INDEX, UMAP_DIR, SQLITE_DB_PATH
from lib.eebo_faiss import EeboFaissIndex
from lib.concept_resolve import resolve_concepts
from lib.eebo_logging import logger
from lib.eebo_db import get_connection

from tier2_0_concept_events import ZarrEventLookup

CONCEPT_DIR       = UMAP_DIR / "concept"
CONCEPT_NEIGH_DIR = UMAP_DIR / "concept_neighbours"
BFS_DIR           = UMAP_DIR / "bfs_global"
CLUSTER_DIR       = UMAP_DIR / "concept_clusters"

K = 25


class EventUniverse:
    def __init__(self):
        self._ids = set()

    def add(self, ids):
        self._ids.update(ids)

    def add_one(self, eid):
        self._ids.add(eid)

    def ids(self):
        return self._ids

    def snapshot(self):
        return set(self._ids)


def backfill_missing_events_from_zarr(lookup, event_ids):
    sqlite_conn = sqlite3.connect(SQLITE_DB_PATH)
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



def bfs_event_expansion(lookup, index, seed_ids, k=25, max_nodes=5000, depth=2):
    visited  = set(seed_ids)
    frontier = set(seed_ids)
    all_nodes = set(seed_ids)

    for _ in range(depth):
        if len(all_nodes) >= max_nodes:
            break

        vecs = np.stack([
            lookup.get_event(eid)["embedding"]
            for eid in frontier
        ])
        _, nn_ids = index.search(vecs, k)

        next_frontier = set()
        for row in nn_ids:
            for nid in row:
                nid = int(nid)
                if nid == -1:
                    continue
                if nid not in visited:
                    visited.add(nid)
                    next_frontier.add(nid)

        all_nodes.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break

    return list(all_nodes)[:max_nodes]


def expand_neighbors(index, lookup, event_ids, k=25, max_points=3000):
    ids = set(event_ids)
    vecs = np.stack([
        lookup.get_event(eid)["embedding"]
        for eid in event_ids
    ])
    _, nn_ids = index.search(vecs, k)
    for row in nn_ids:
        for nid in row:
            nid = int(nid)
            if nid != -1:
                ids.add(nid)
            if len(ids) >= max_points:
                break
    return list(ids)[:max_points]



def fit_umap_local(lookup, event_ids, n_neighbors=15, min_dist=0.1):
    if not event_ids:
        return {}
    X = np.stack([
        lookup.get_event(eid)["embedding"]
        for eid in event_ids
    ]).astype(np.float32)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=42,
        metric="cosine",
    )
    emb = reducer.fit_transform(X)
    return {int(eid): (float(emb[i, 0]), float(emb[i, 1]))
            for i, eid in enumerate(event_ids)}


def fit_umap_global(lookup, all_event_ids, n_neighbors=15, min_dist=0.1):
    event_ids = list(all_event_ids)
    logger.info(f"[tier3] fitting global PaCMAP on {len(event_ids):,} points")
    X = np.stack([
        lookup.get_event(eid)["embedding"]
        for eid in event_ids
    ]).astype(np.float32)
    reducer = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=15,
        MN_ratio=0.5,
        FP_ratio=2.0,
        random_state=42,
    )
    emb = reducer.fit_transform(X)
    return {int(eid): (float(emb[i, 0]), float(emb[i, 1]))
            for i, eid in enumerate(event_ids)}



def fit_cluster_local(lookup, event_ids):
    """
    5-D UMAP for HDBSCAN, then a separate 2-D UMAP for the local display
    coordinates (nx/ny).  Returns {event_id -> (x, y)} local coords and
    the raw cluster labels list.
    """
    if len(event_ids) < 10:
        return {int(eid): (0.0, 0.0) for eid in event_ids}, [-1] * len(event_ids)

    X = np.stack([
        lookup.get_event(eid)["embedding"]
        for eid in event_ids
    ]).astype(np.float32)

    # 5-D reduction for clustering
    reducer_5d = umap.UMAP(
        n_components=5,
        n_neighbors=15,
        min_dist=0.0,
        metric='cosine',
        random_state=42,
    )
    reduced_5d = reducer_5d.fit_transform(X)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=3,
        metric='euclidean',
        cluster_selection_method='eom',
        cluster_selection_epsilon=0.5,  # merge clusters closer than this distance
    )
    labels = clusterer.fit_predict(reduced_5d).tolist()

    # 2-D reduction for local display coords
    reducer_2d = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42,
    )
    emb_2d = reducer_2d.fit_transform(X)

    local_coords = {
        int(eid): (float(emb_2d[i, 0]), float(emb_2d[i, 1]))
        for i, eid in enumerate(event_ids)
    }
    return local_coords, labels


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
                    extra_fields: dict[int, dict] | None = None):
    """
    extra_fields: optional {event_id -> {field: value}} merged into each point.
    Used by the cluster output to attach cluster_id / cluster_label.
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
        lx, ly = local_coords[eid]
        gx, gy = global_coords[eid]
        pt = {
            "event_id": str(eid),
            "x":   lx,
            "y":   ly,
            "nx":  (lx - lx0) / ldx,
            "ny":  (ly - ly0) / ldy,
            "gx":  gx,
            "gy":  gy,
            "gnx": (gx - gx0) / gdx,
            "gny": (gy - gy0) / gdy,
        }
        if extra_fields and eid in extra_fields:
            pt.update(extra_fields[eid])
        points.append(pt)
    return points


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"[tier3] Wrote {path}")




def main_OLD():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", type=str, default=None)
    parser.add_argument( "--mode", type=str, default="full", choices=["full", "clustering"] )
    parser.add_argument( "--false_positives", type=str, nargs="*", default=[] )
    args = parser.parse_args()
    concept = args.concept
    mode = args.mode
    false_positives = getattr(args, "false_positives", [])

    logger.info("[tier3] loading index + lookup")

    lookup = ZarrEventLookup(ZARR_ROOT / "tier1")
    index  = EeboFaissIndex.load(FAISS_TIER1_INDEX)

    manifest = {"concepts": [], "global": None, "globalBounds": None}

    all_concept_events = []

    buffered_concept       = []
    buffered_concept_neigh = []
    buffered_clusters      = []

    all_event_ids = set()

    #
    # Pass 1 — concept + clustering inputs
    #
    if mode == "full":
        for concept_name, concept in resolve_concepts(concept=concept, false_positives=false_positives):
            logger.info(f"[tier3] processing concept={concept_name}")

            seed_ids = list(lookup.iter_matching_event_ids(set(concept["forms"])))
            concept_sample = seed_ids[:1000]
            local_concept  = fit_umap_local(lookup, concept_sample)

            buffered_concept.append((
                CONCEPT_DIR / f"{concept_name}.json",
                {"type": "concept", "concept": concept_name},
                concept_sample,
                local_concept,
            ))
            all_event_ids.update(concept_sample)

            neigh_ids   = expand_neighbors(index, lookup, concept_sample, max_points=3000)
            local_neigh = fit_umap_local(lookup, neigh_ids)

            buffered_concept_neigh.append((
                CONCEPT_NEIGH_DIR / f"{concept_name}.json",
                {"type": "concept_neighbours", "concept": concept_name},
                neigh_ids,
                local_neigh,
            ))
            all_event_ids.update(neigh_ids)

            logger.info(f"[tier3] clustering {concept_name} ({len(concept_sample)} points)")
            cluster_local_coords, cluster_labels = fit_cluster_local(lookup, concept_sample)

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

    else:
        logger.info("[tier3] clustering-only mode (loading concept outputs)")

        for concept_file in CONCEPT_DIR.glob("*.json"):
            concept_name = concept_file.stem

            with open(concept_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            concept_sample = [int(p["event_id"]) for p in data["points"]]

            logger.info(f"[tier3] clustering {concept_name} ({len(concept_sample)})")

            cluster_local_coords, cluster_labels = fit_cluster_local(lookup, concept_sample)

            buffered_clusters.append((
                CLUSTER_DIR / f"{concept_name}.json",
                {"type": "concept_clusters", "concept": concept_name},
                concept_sample,
                cluster_local_coords,
                cluster_labels,
            ))

            all_concept_events.extend(concept_sample)

    # BFS (full only)
    if mode == "full":
        bfs_ids   = bfs_event_expansion(
            lookup, index, all_concept_events,
            max_nodes=5000, depth=2
        )
        local_bfs = fit_umap_local(lookup, bfs_ids)
        all_event_ids.update(bfs_ids)
    else:
        bfs_ids = []
        local_bfs = {}

    # Backfill safety
    if all_event_ids:
        backfill_missing_events_from_zarr(lookup, all_event_ids)

    # Global projection (SAFE GUARD ADDED)
    if all_event_ids:
        global_coords = fit_umap_global(lookup, all_event_ids)
        global_bounds = compute_bounds_from_coords(list(global_coords.values()))
        global_bounds_padded = add_padding(global_bounds)
    else:
        logger.warning("[tier3] no event ids — skipping global projection")
        global_coords = {}
        global_bounds = {"minX": 0, "maxX": 0, "minY": 0, "maxY": 0}
        global_bounds_padded = global_bounds

    logger.info(f"[tier3] global bounds (padded): {global_bounds_padded}")

    #
    # Pass 3 — concept + neighbour outputs
    #
    for path, meta, event_ids, local_coords in buffered_concept + buffered_concept_neigh:
        local_bounds = add_padding(
            compute_bounds_from_coords([local_coords[e] for e in event_ids])
        )

        points = assemble_points(
            event_ids, local_coords, global_coords,
            local_bounds, global_bounds_padded,
        )

        write_json(path, {
            **meta,
            "bounds": local_bounds,
            "globalBounds": global_bounds_padded,
            "points": points,
        })

    #
    # Cluster outputs
    #
    for path, meta, event_ids, local_coords, cluster_labels in buffered_clusters:
        unique_clusters = sorted(c for c in set(cluster_labels) if c != -1)
        label_map = {cid: chr(65 + i) for i, cid in enumerate(unique_clusters)}

        extra = {
            int(eid): {
                "cluster_id":    int(cluster_labels[i]),
                "cluster_label": label_map.get(cluster_labels[i]),
            }
            for i, eid in enumerate(event_ids)
        }

        local_bounds = add_padding(
            compute_bounds_from_coords([local_coords[e] for e in event_ids])
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
            "n_events": len(event_ids),
            "bounds": local_bounds,
            "globalBounds": global_bounds_padded,
            "clusters": {
                "label_map": {str(k): v for k, v in label_map.items()},
                "aggregates": aggregates,
            },
            "points": points,
        })

    #
    # BFS output (SAFE GUARD ADDED)
    #
    if bfs_ids:
        local_bounds_bfs = add_padding(
            compute_bounds_from_coords([local_bfs[e] for e in bfs_ids])
        )

        points_bfs = assemble_points(
            bfs_ids, local_bfs, global_coords,
            local_bounds_bfs, global_bounds_padded,
        )

        write_json(
            BFS_DIR / "global.json",
            {
                "type": "bfs_global",
                "bounds": local_bounds_bfs,
                "globalBounds": global_bounds_padded,
                "depth": 2,
                "k": K,
                "points": points_bfs,
            }
        )

        manifest["global"] = "/umap/bfs_global/global.json"
    else:
        manifest["global"] = None

    manifest["globalBounds"] = global_bounds_padded
    write_json(UMAP_DIR / "manifest.json", manifest)

    logger.info("[tier3] complete")


def run_tier3_core(
    *,
    index,
    lookup,
    concept=None,
    false_positives=None,
    mode="full",
    emit=None,
):
    logger.info("[tier3 run_tier3_core] Enter")

    false_positives = false_positives or []

    manifest = {"concepts": [], "global": None, "globalBounds": None}
    all_concept_events = []
    buffered_concept       = []
    buffered_concept_neigh = []
    buffered_clusters      = []
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
            concept_sample = seed_ids[:1000]
            local_concept  = fit_umap_local(lookup, concept_sample)

            buffered_concept.append((
                CONCEPT_DIR / f"{concept_name}.json",
                {"type": "concept", "concept": concept_name},
                concept_sample,
                local_concept,
            ))
            all_event_ids.update(concept_sample)

            neigh_ids   = expand_neighbors(index, lookup, concept_sample, max_points=3000)
            local_neigh = fit_umap_local(lookup, neigh_ids)

            buffered_concept_neigh.append((
                CONCEPT_NEIGH_DIR / f"{concept_name}.json",
                {"type": "concept_neighbours", "concept": concept_name},
                neigh_ids,
                local_neigh,
            ))
            all_event_ids.update(neigh_ids)

            logger.info(f"[tier3] clustering {concept_name} ({len(concept_sample)} points)")
            cluster_local_coords, cluster_labels = fit_cluster_local(lookup, concept_sample)

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

            concept_sample = [int(p["event_id"]) for p in data["points"]]

            logger.info(f"[tier3] clustering {concept_name} ({len(concept_sample)})")
            if emit:
                emit("concept_start", {"concept": concept_name})

            cluster_local_coords, cluster_labels = fit_cluster_local(lookup, concept_sample)

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

    # BFS (full only)
    if mode == "full":
        if emit:
            emit("bfs_start", {})
        bfs_ids   = bfs_event_expansion(lookup, index, all_concept_events, max_nodes=5000, depth=2)
        local_bfs = fit_umap_local(lookup, bfs_ids)
        all_event_ids.update(bfs_ids)
    else:
        bfs_ids   = []
        local_bfs = {}

    # Backfill safety
    if all_event_ids:
        backfill_missing_events_from_zarr(lookup, all_event_ids)

    # Global projection
    if emit:
        emit("global_projection_start", {"n_events": len(all_event_ids)})

    if all_event_ids:
        global_coords        = fit_umap_global(lookup, all_event_ids)
        global_bounds        = compute_bounds_from_coords(list(global_coords.values()))
        global_bounds_padded = add_padding(global_bounds)
    else:
        logger.warning("[tier3] no event ids — skipping global projection")
        global_coords        = {}
        global_bounds        = {"minX": 0, "maxX": 0, "minY": 0, "maxY": 0}
        global_bounds_padded = global_bounds

    logger.info(f"[tier3] global bounds (padded): {global_bounds_padded}")

    #
    # Pass 3 — concept + neighbour outputs
    #
    for path, meta, event_ids, local_coords in buffered_concept + buffered_concept_neigh:
        local_bounds = add_padding(
            compute_bounds_from_coords([local_coords[e] for e in event_ids])
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

    # Cluster outputs
    for path, meta, event_ids, local_coords, cluster_labels in buffered_clusters:
        unique_clusters = sorted(c for c in set(cluster_labels) if c != -1)
        label_map = {cid: chr(65 + i) for i, cid in enumerate(unique_clusters)}

        extra = {
            int(eid): {
                "cluster_id":    int(cluster_labels[i]),
                "cluster_label": label_map.get(cluster_labels[i]),
            }
            for i, eid in enumerate(event_ids)
        }

        local_bounds = add_padding(
            compute_bounds_from_coords([local_coords[e] for e in event_ids])
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

    # BFS output
    if bfs_ids:
        local_bounds_bfs = add_padding(
            compute_bounds_from_coords([local_bfs[e] for e in bfs_ids])
        )
        points_bfs = assemble_points(
            bfs_ids, local_bfs, global_coords,
            local_bounds_bfs, global_bounds_padded,
        )
        write_json(
            BFS_DIR / "global.json",
            {
                "type":         "bfs_global",
                "bounds":       local_bounds_bfs,
                "globalBounds": global_bounds_padded,
                "depth":        2,
                "k":            K,
                "points":       points_bfs,
            }
        )
        manifest["global"] = "/umap/bfs_global/global.json"
    else:
        manifest["global"] = None

    manifest["globalBounds"] = global_bounds_padded
    write_json(UMAP_DIR / "manifest.json", manifest)

    if emit:
        emit("tier3_done", {})
    logger.info("[tier3 run_tier3_core] Complete")


def run_tier3_service(
    *,
    index,
    lookup,
    concept=None,
    false_positives=None,
    mode="full",
    emit=None,
):
    logger.info("[tier3 run_tier3_service] Enter")
    return run_tier3_core(
        index=index,
        lookup=lookup,
        concept=concept,
        false_positives=false_positives,
        mode=mode,
        emit=emit,
    )


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
        index=index,
        lookup=lookup,
        concept=args.concept,
        false_positives=args.false_positives,
        mode=args.mode,
        emit=None,
    )

    logger.info("[tier3] complete")


if __name__ == "__main__":
    main()
