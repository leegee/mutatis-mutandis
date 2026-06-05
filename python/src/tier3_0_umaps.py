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

            manifest.json

"""
import argparse
import json
from pathlib import Path
import numpy as np
import umap

from lib.eebo_config import ZARR_ROOT, FAISS_TIER1_INDEX, UMAP_DIR
from lib.eebo_faiss import EeboFaissIndex
from lib.concept_resolve import resolve_concepts
from lib.eebo_logging import logger

from tier2_0_concept_events import ZarrEventLookup


CONCEPT_DIR = UMAP_DIR / "concept"
CONCEPT_NEIGH_DIR = UMAP_DIR / "concept_neighbours"
BFS_DIR = UMAP_DIR / "bfs_global"

K = 25


def bfs_event_expansion(lookup, index, seed_ids, k=25, max_nodes=5000, depth=2):
    visited = set(seed_ids)
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


def compute_bounds(points):
    xs = np.array([p["x"] for p in points], dtype=np.float32)
    ys = np.array([p["y"] for p in points], dtype=np.float32)

    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())

    return {
        "minX": min_x,
        "maxX": max_x,
        "minY": min_y,
        "maxY": max_y,
    }


def add_padding(bounds, pad_ratio=0.1):
    width = bounds["maxX"] - bounds["minX"]
    height = bounds["maxY"] - bounds["minY"]

    pad_x = width * pad_ratio
    pad_y = height * pad_ratio

    return {
        "minX": bounds["minX"] - pad_x,
        "maxX": bounds["maxX"] + pad_x,
        "minY": bounds["minY"] - pad_y,
        "maxY": bounds["maxY"] + pad_y,
    }


def normalize_points(points, bounds):
    min_x, max_x = bounds["minX"], bounds["maxX"]
    min_y, max_y = bounds["minY"], bounds["maxY"]

    dx = (max_x - min_x) + 1e-12
    dy = (max_y - min_y) + 1e-12

    out = []
    for p in points:
        out.append({
            **p,
            "nx": (p["x"] - min_x) / dx,
            "ny": (p["y"] - min_y) / dy,
        })

    return out


def run_umap(lookup, event_ids, n_neighbors=15, min_dist=0.1):
    if len(event_ids) == 0:
        return []

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

    return [
        {
            "event_id": int(eid),
            "x": float(emb[i][0]),
            "y": float(emb[i][1]),
        }
        for i, eid in enumerate(event_ids)
    ]


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


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"[tier3] Wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", type=str, default=None)
    args = parser.parse_args()

    logger.info("[tier3] loading index + lookup")

    lookup = ZarrEventLookup(ZARR_ROOT / "tier1")
    index = EeboFaissIndex.load(FAISS_TIER1_INDEX)

    manifest = {
        "concepts": [],
        "global": None
    }

    all_concept_events = []

    for concept_name, concept in resolve_concepts(args):

        logger.info(f"[tier3] processing concept={concept_name}")

        seed_ids = list(lookup.iter_matching_event_ids(set(concept["forms"])))

        # 1. concept UMAP (≤1k)
        concept_sample = seed_ids[:1000]
        points = run_umap(lookup, concept_sample)
        bounds_raw = compute_bounds(points)
        bounds_padded = add_padding(bounds_raw)
        points_norm = normalize_points(points, bounds_padded)

        write_json(
            CONCEPT_DIR / f"{concept_name}.json",
            {
                "type": "concept",
                "concept": concept_name,
                "bounds": bounds_padded,
                "points": points_norm,
            }
        )

        # 2. concept + neighbours (≤3k)
        neigh_ids = expand_neighbors(index, lookup, concept_sample, max_points=3000)
        concept_neigh_points = run_umap(lookup, neigh_ids)
        bounds_raw = compute_bounds(concept_neigh_points)
        bounds_padded = add_padding(bounds_raw)
        points_norm = normalize_points(concept_neigh_points, bounds_padded)

        write_json(
            CONCEPT_NEIGH_DIR / f"{concept_name}.json",
            {
                "type": "concept_neighbours",
                "concept": concept_name,
                "bounds": bounds_padded,
                "points": points_norm
            }
        )

        # collect for global BFS
        all_concept_events.extend(concept_sample)

        manifest["concepts"].append({
            "name": concept_name,
            "concept": f"/umap/concept/{concept_name}.json",
            "concept_neighbours": f"/umap/concept_neighbours/{concept_name}.json",
        })

    # 3. BFS global (≤5k, depth=2)
    bfs_ids = bfs_event_expansion(
        lookup,
        index,
        all_concept_events,
        max_nodes=5000,
        depth=2
    )

    global_ponts = run_umap(lookup, bfs_ids)

    bounds_raw = compute_bounds(global_ponts)
    bounds_padded = add_padding(bounds_raw)
    points_norm = normalize_points(global_ponts, bounds_padded)

    write_json(
        BFS_DIR / "global.json",
        {
            "type": "bfs_global",
            "points": points_norm,
            "bounds": bounds_padded,
            "depth": 2,
            "k": K,
        }
    )

    manifest["global"] = "/umap/bfs_global/global.json"

    write_json(
        UMAP_DIR / "manifest.json",
        manifest
    )

    logger.info("[tier3] complete")


if __name__ == "__main__":
    main()
