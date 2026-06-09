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

Each point carries six coordinates:

    Local projection  (independent UMAP fit on this file's points only)
        x,  y   — raw local UMAP output
        nx, ny  — normalised within this file's own padded bounds  [0, 1]

    Global projection  (single joint UMAP fit across ALL points in the run)
        gx,  gy  — raw global UMAP output
        gnx, gny — normalised within the shared global padded bounds  [0, 1]

The local projection is optimised for within-file structure: cluster shapes,
local density, and fine-grained neighbourhood relationships.

The global projection places every point from every file into a single shared
semantic space.  Points that are close in gnx/gny are semantically similar
regardless of which file they appear in, making cross-file comparison and
a unified canvas view meaningful.

The joint UMAP is fit on the union of all event IDs across all three output
types (concept samples + neighbour expansions + BFS expansion).  n_neighbors
and min_dist are tuned globally, which gives a slightly coarser local
structure than the independent fits — use nx/ny when local detail matters,
gnx/gny when cross-file position matters.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import umap
import pacmap
import sqlite3

from lib.eebo_config import ZARR_ROOT, FAISS_TIER1_INDEX, PLOT_DIR, SQLITE_DB_PATH
from lib.eebo_faiss import EeboFaissIndex
from lib.concept_resolve import resolve_concepts
from lib.eebo_logging import logger
from lib.eebo_db import get_connection

from tier2_0_concept_events import ZarrEventLookup

CONCEPT_DIR       = PLOT_DIR / "concept"
CONCEPT_NEIGH_DIR = PLOT_DIR / "concept_neighbours"
BFS_DIR           = PLOT_DIR / "bfs_global"

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

        # Gather doc_ids we need pub_year for
        missing_doc_ids = set()
        for eid in missing_ids:
            event = lookup.get_event(eid)
            if event is not None:
                missing_doc_ids.add(event["doc_id"])

        # Fetch pub_year from Postgres
        pg_conn = get_connection()
        try:
            with pg_conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT doc_id, pub_year
                    FROM pamphlet_tokens
                    WHERE doc_id = ANY(%s)
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


# FAISS / BFS helpers
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
    """Fit an independent UMAP on this set of points only.

    Returns a dict {event_id -> (x, y)} using the local projection.
    """
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
    """Fit a single joint UMAP across all event IDs in the run.

    Returns a dict {event_id -> (gx, gy)} using the shared projection.
    Points that are semantically similar will be close in this space
    regardless of which output file they belong to.
    """
    event_ids = list(all_event_ids)

    logger.info(f"[tier3] fitting global UMAP on {len(event_ids):,} points")

    X = np.stack([
        lookup.get_event(eid)["embedding"]
        for eid in event_ids
    ]).astype(np.float32)

    # reducer = umap.UMAP(
    #     n_neighbors=n_neighbors,
    #     min_dist=min_dist,
    #     n_components=2,
    #     random_state=42,
    #     metric="cosine",
    # )
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


# Bounds helpers
def compute_bounds_from_coords(coords):
    """coords: list of (x, y) tuples."""
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
                    local_bounds, global_bounds):
    """Build the final point list for one output file.

    Each point:
        event_id            — event identifier
        x,  y               — raw local UMAP coords
        nx, ny              — local coords normalised to [0, 1]
        gx, gy              — raw global UMAP coords
        gnx, gny            — global coords normalised to [0, 1]
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
        points.append({
            "event_id": str(eid),
            "x":   lx,
            "y":   ly,
            "nx":  (lx - lx0) / ldx,
            "ny":  (ly - ly0) / ldy,
            "gx":  gx,
            "gy":  gy,
            "gnx": (gx - gx0) / gdx,
            "gny": (gy - gy0) / gdy,
        })
    return points


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
    index  = EeboFaissIndex.load(FAISS_TIER1_INDEX)

    manifest = {"concepts": [], "global": None, "globalBounds": None}

    all_concept_events = []

    # Pass 1 — collect all event ID sets; run local UMAP fits per file.
    # We buffer everything because the joint global UMAP needs the full
    # union of IDs, which isn't known until all concepts are processed.

    # Each entry: (output_path, metadata_dict, ordered event_id list,
    #              local_coords {eid -> (x,y)})
    buffered_concept       = []
    buffered_concept_neigh = []

    all_event_ids = set()   # union for the joint global UMAP

    for concept_name, concept in resolve_concepts(args):
        logger.info(f"[tier3] processing concept={concept_name}")

        seed_ids = list(lookup.iter_matching_event_ids(set(concept["forms"])))

        # 1. concept (≤1 k) — local fit
        concept_sample = seed_ids[:1000]
        local_concept  = fit_umap_local(lookup, concept_sample)

        buffered_concept.append((
            CONCEPT_DIR / f"{concept_name}.json",
            {"type": "concept", "concept": concept_name},
            concept_sample,
            local_concept,
        ))

        all_event_ids.update(concept_sample)

        # 2. concept + neighbours (≤3 k) — local fit
        neigh_ids    = expand_neighbors(index, lookup, concept_sample, max_points=3000)
        local_neigh  = fit_umap_local(lookup, neigh_ids)

        buffered_concept_neigh.append((
            CONCEPT_NEIGH_DIR / f"{concept_name}.json",
            {"type": "concept_neighbours", "concept": concept_name},
            neigh_ids,
            local_neigh,
        ))

        all_event_ids.update(neigh_ids)

        # collect seed events for BFS
        all_concept_events.extend(concept_sample)

        manifest["concepts"].append({
            "name":               concept_name,
            "concept":            f"/umap/concept/{concept_name}.json",
            "concept_neighbours": f"/umap/concept_neighbours/{concept_name}.json",
        })

    # 3. BFS global (≤5 k, depth=2) — local fit
    bfs_ids     = bfs_event_expansion(lookup, index, all_concept_events, max_nodes=5000, depth=2)
    local_bfs   = fit_umap_local(lookup, bfs_ids)

    all_event_ids.update(bfs_ids)

    backfill_missing_events_from_zarr(lookup, all_event_ids)

    # Pass 2 — fit the single joint global UMAP on the full union.
    global_coords = fit_umap_global(lookup, all_event_ids)

    # Global bounds from the joint projection (padded once, shared everywhere)
    global_bounds        = compute_bounds_from_coords(list(global_coords.values()))
    global_bounds_padded = add_padding(global_bounds)

    logger.info(f"[tier3] global bounds (padded): {global_bounds_padded}")

    # Pass 3 — assemble and write all output files.
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

    # BFS global
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

    manifest["global"]       = "/umap/bfs_global/global.json"
    manifest["globalBounds"] = global_bounds_padded

    write_json(PLOT_DIR / "manifest.json", manifest)

    logger.info("[tier3] complete")


if __name__ == "__main__":
    main()

