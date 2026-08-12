#!/usr/bin/env python

# tier3_tune.py
#
# Fast iteration harness for tuning UMAP / Leiden parameters used in
# tier3_0_project_cluster.py, without repeatedly paying for:
#   - SQLite event lookups
#   - zarr embedding concatenation
#   - the global UMAP projection over the full event set
#
# Usage (script mode, one-shot grid + report):
#   python src/tools/tier3_tune.py --concept "CHURCH" \
#       --n-neighbors 10 15 30 --min-dist 0.0 0.05 0.1 --resolution 0.5 0.8 1.2
#
# Usage (interactive, recommended for real tuning):
#   ipython
#   >>> from src.tools.tier3_tune import *
#   >>> con, lookup = setup()
#   >>> event_ids, vectors = get_vectors(con, lookup, "CHURCH")
#   >>> labels = leiden_cluster_r(vectors, resolution=1.0)
#   >>> coords = project(vectors, {"n_neighbors": 15, "min_dist": 0.05, "metric": "cosine"})
#   >>> import matplotlib.pyplot as plt
#   >>> fig = plot_clusters(coords, labels)
#   >>> plt.show()
#
# Nothing in this script writes to SQLite. Use tier3_0_project_cluster.py's
# write_geometry / write_cluster_info once you've settled on final params.

from __future__ import annotations

import argparse
import itertools
import pickle
from pathlib import Path

import numpy as np
import igraph as ig
import leidenalg

from lib.corpus_config import (
    CORPUS_TIER2_DB_PATH,
    EVENTSTORE_T1_PATH,
    TMP_DIR,
    faiss_index_paths,
)
from lib.corpus_logging import logger

# Reuse the plumbing from the real pipeline instead of duplicating it.
from tier3_0_project_cluster import (
    sqlite_connection,
    load_event_rows,
    load_vectors,
    load_indices,
    project,
    build_knn_graph,
    build_global_projection,
)
from lib.zarr_event_lookup import ZarrEventLookup
from lib.concept_resolve import resolve_concepts


CACHE_DIR = Path(TMP_DIR) / "tier3_tune_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

VECTOR_CACHE = CACHE_DIR / "vectors"
VECTOR_CACHE.mkdir(exist_ok=True)

GLOBAL_CACHE = CACHE_DIR / "global_coords.pkl"


# ---------------------------------------------------------------------------
# One-time setup, kept cheap to call repeatedly (e.g. from ipython) since it
# skips index loading unless explicitly requested.
# ---------------------------------------------------------------------------

def setup(mask: bool = False, with_index: bool = True):
    """
    Open the DB connection and zarr lookup. Call once per session.

    with_index=True (default) attaches the FAISS index, which
    get_concatenated_embeddings requires -- so it's needed the first time
    you load vectors for a given concept, cached or not. Only pass
    with_index=False if you're just poking at cached .npz vector files
    directly and never calling get_vectors()/get_global_coords() fresh.
    """
    con = sqlite_connection(CORPUS_TIER2_DB_PATH)
    lookup = ZarrEventLookup(EVENTSTORE_T1_PATH)

    if with_index:
        years = sorted(
            int(y) for y in lookup.pub_year if y > 0
        )
        years = sorted(set(years))
        index = load_indices(years, masked=mask)
        lookup.attach_index(index)

    return con, lookup


# ---------------------------------------------------------------------------
# Cached vector loading, per concept
# ---------------------------------------------------------------------------

def _vector_cache_path(concept: str) -> Path:
    safe = concept.replace("/", "_")
    return VECTOR_CACHE / f"{safe}.npz"


def get_vectors(con, lookup, concept: str, refresh: bool = False):
    """
    Load (event_ids, vectors) for a concept, caching to disk so repeated
    calls across tuning runs skip the SQLite + zarr lookup entirely.
    """
    cache_path = _vector_cache_path(concept)

    if cache_path.exists() and not refresh:
        data = np.load(cache_path, allow_pickle=False)
        logger.info(f"[tune] {concept}: loaded {len(data['event_ids'])} vectors from cache")
        return data["event_ids"].tolist(), data["vectors"]

    rows = load_event_rows(con, concept)
    if not rows:
        logger.warning(f"[tune] {concept}: no events")
        return [], np.empty((0, 0), dtype=np.float32)

    event_ids, vectors = load_vectors(lookup, rows)
    np.savez(cache_path, event_ids=np.array(event_ids), vectors=vectors)
    logger.info(f"[tune] {concept}: cached {len(event_ids)} vectors -> {cache_path}")
    return event_ids, vectors


# ---------------------------------------------------------------------------
# Cached global projection (expensive: runs over every concept's events)
# ---------------------------------------------------------------------------

def get_global_coords(con, lookup, refresh: bool = False):
    if GLOBAL_CACHE.exists() and not refresh:
        with open(GLOBAL_CACHE, "rb") as f:
            logger.info(f"[tune] loaded global coords from cache")
            return pickle.load(f)

    global_concepts = [c for c, _ in resolve_concepts(concept=None)]

    all_field_event_ids = []
    for concept in global_concepts:
        rows = load_event_rows(con, concept)
        all_field_event_ids.extend(int(row[0]) for row in rows)

    all_field_event_ids = sorted(set(all_field_event_ids))
    global_coords = build_global_projection(lookup, all_field_event_ids)

    with open(GLOBAL_CACHE, "wb") as f:
        pickle.dump(global_coords, f)
    logger.info(f"[tune] cached global coords -> {GLOBAL_CACHE}")
    return global_coords


# ---------------------------------------------------------------------------
# Leiden clustering with resolution exposed (the real pipeline hardcodes 0.8)
# ---------------------------------------------------------------------------

def leiden_cluster_r(vectors, resolution: float = 0.8, min_in_cluster: int = 7):
    if len(vectors) < min_in_cluster:
        return np.full(len(vectors), -1, dtype=np.int32)

    edges = build_knn_graph(vectors)

    graph = ig.Graph(edges=edges, directed=False)

    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        seed=42,
        resolution_parameter=resolution,
    )

    labels = np.full(len(vectors), -1, dtype=np.int32)
    for cluster_id, members in enumerate(partition):
        for member in members:
            labels[member] = cluster_id

    return labels


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def cluster_summary(labels: np.ndarray) -> dict:
    n = len(labels)
    noise = int((labels == -1).sum())
    real = labels[labels != -1]
    n_clusters = len(set(real.tolist()))
    sizes = np.bincount(real) if len(real) else np.array([])
    return {
        "n_points": n,
        "n_clusters": n_clusters,
        "n_noise": noise,
        "noise_frac": round(noise / n, 3) if n else 0.0,
        "median_cluster_size": int(np.median(sizes)) if len(sizes) else 0,
        "max_cluster_size": int(sizes.max()) if len(sizes) else 0,
    }


def plot_clusters(coords: np.ndarray, labels: np.ndarray, title: str = "", save_path: Path | None = None):
    """Quick matplotlib scatter, colored by cluster. Import kept local so
    this module doesn't require matplotlib unless you actually plot."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    noise_mask = labels == -1
    ax.scatter(coords[noise_mask, 0], coords[noise_mask, 1], c="lightgray", s=8, label="noise")
    ax.scatter(coords[~noise_mask, 0], coords[~noise_mask, 1], c=labels[~noise_mask], cmap="tab20", s=10)
    ax.set_title(title)
    ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"[tune] saved plot -> {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Grid search over UMAP + Leiden params for a single concept
# ---------------------------------------------------------------------------

def grid_search(
    vectors,
    n_neighbors_options,
    min_dist_options,
    resolution_options,
):
    results = []

    for n_neighbors, min_dist, resolution in itertools.product(
        n_neighbors_options, min_dist_options, resolution_options
    ):
        params = {"n_neighbors": n_neighbors, "min_dist": min_dist, "metric": "cosine"}
        coords = project(vectors, params)
        labels = leiden_cluster_r(vectors, resolution=resolution)

        summary = cluster_summary(labels)
        summary.update({
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "resolution": resolution,
        })
        results.append((summary, coords, labels))

    return results


def print_grid_results(results):
    header = f"{'n_neighbors':>11} {'min_dist':>9} {'resolution':>10} {'clusters':>9} {'noise%':>7} {'med_size':>9} {'max_size':>9}"
    print(header)
    print("-" * len(header))
    for summary, _, _ in results:
        print(
            f"{summary['n_neighbors']:>11} "
            f"{summary['min_dist']:>9} "
            f"{summary['resolution']:>10} "
            f"{summary['n_clusters']:>9} "
            f"{summary['noise_frac']*100:>6.1f}% "
            f"{summary['median_cluster_size']:>9} "
            f"{summary['max_cluster_size']:>9}"
        )


# ---------------------------------------------------------------------------
# CLI entry point: one-shot grid over a concept, no DB writes
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", required=True)
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--refresh-vectors", action="store_true", help="ignore cached vectors")
    parser.add_argument("--n-neighbors", type=int, nargs="+", default=[15])
    parser.add_argument("--min-dist", type=float, nargs="+", default=[0.05])
    parser.add_argument("--resolution", type=float, nargs="+", default=[0.8])
    parser.add_argument("--plot-best", action="store_true", help="plot the config with fewest noise points")
    args = parser.parse_args()

    con, lookup = setup(mask=args.mask)

    event_ids, vectors = get_vectors(con, lookup, args.concept, refresh=args.refresh_vectors)
    if len(event_ids) == 0:
        logger.warning(f"[tune] {args.concept}: nothing to tune")
        return

    logger.info(f"[tune] {args.concept}: {len(event_ids)} events, running grid of "
                f"{len(args.n_neighbors) * len(args.min_dist) * len(args.resolution)} configs")

    results = grid_search(vectors, args.n_neighbors, args.min_dist, args.resolution)
    print_grid_results(results)

    if args.plot_best:
        best = min(results, key=lambda r: r[0]["noise_frac"])
        summary, coords, labels = best
        title = f"{args.concept} nn={summary['n_neighbors']} md={summary['min_dist']} res={summary['resolution']}"
        out_path = CACHE_DIR / f"{args.concept}_best.png"
        plot_clusters(coords, labels, title=title, save_path=out_path)

    con.close()


if __name__ == "__main__":
    main()
