#!/usr/bin/env python
"""
usage_cluster2.py

Cluster diachronic neighbours of concepts using HDBSCAN and compute
configurable notions of cluster "mass".

This script subsumes:
- usage_clusterer.py (weighted semantic mass)
- usage_cluster_tracker.py (token-count tracking)

Mass is treated as an explicit modelling choice.

Outputs:
* JSON with cluster assignments and per-cluster mass metrics per slice.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Callable

import hdbscan
import numpy as np

import lib.eebo_config as config
import lib.eebo_db as eebo_db
from lib.eebo_logging import logger
from lib.faiss_slices import load_slice_index, get_vector

TARGET: Optional[str] = None  # set to 'LAW' to restrict

TOP_K = 100
SIM_THRESHOLD = 0.7

# Tracker mode: ["count"]
# Clusterer mode: ["count", "freq", "sim", "weighted"]
MASS_MODES = ["count", "freq", "sim", "weighted"]

if TARGET:
    OUT_FILE = config.OUT_DIR / f"usage_clusters_{TARGET.lower()}.json"
else:
    OUT_FILE = config.OUT_DIR / "usage_clusters_all_concepts.json"

# Types
Neighbour = Dict[str, Any]
MassFn = Callable[[List[Neighbour]], float]

#
# Mass functions
#
def mass_token_count(ns: List[Neighbour]) -> int:
    return len(ns)

def mass_freq(ns: List[Neighbour]) -> int:
    return sum(n.get("freq", 0) or 0 for n in ns)

def mass_sim(ns: List[Neighbour]) -> float:
    return sum(n["sim"] for n in ns)

def mass_weighted(ns: List[Neighbour]) -> float:
    return sum((n.get("freq", 0) or 0) * n["sim"] for n in ns)

MASS_FUNCTIONS: Dict[str, MassFn] = {
    "count": mass_token_count,
    "freq": mass_freq,
    "sim": mass_sim,
    "weighted": mass_weighted,
}


def collect_vectors(
    concept: str,
    *,
    include_freq: bool = True
) -> Dict[str, List[Neighbour]]:
    """
    Collect neighbour vectors (and optional frequencies) per slice.
    """
    all_slices: Dict[str, List[Neighbour]] = {}

    with eebo_db.get_connection() as conn:
        for slice_start, slice_end in config.SLICES:
            slice_key = f"{slice_start}_{slice_end}"
            all_slices[slice_key] = []

            index, vocab = load_slice_index((slice_start, slice_end))

            seed = concept.lower()
            seed_vec = get_vector(conn, seed, slice_start, slice_end)
            if seed_vec is None:
                logger.warning(f"No vector for '{seed}' in slice {slice_key}")
                continue

            D, Idx = index.search(seed_vec.reshape(1, -1), TOP_K)
            top_neighbors = [
                (vocab[idx], float(sim))
                for sim, idx in zip(D[0], Idx[0], strict=True)
                if idx != -1
            ]

            known_forms = config.CONCEPT_SETS[concept].get("forms", set())
            false_positives = config.CONCEPT_SETS[concept].get("false_positives", set())

            top_neighbors = [
                (t, s)
                for t, s in top_neighbors
                if t not in known_forms and t not in false_positives
            ]

            if not top_neighbors:
                continue

            freq_map: Dict[str, int] = {}
            if include_freq:
                tokens = [t for t, _ in top_neighbors]
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT token, COUNT(*)
                        FROM pamphlet_tokens
                        WHERE token = ANY(%s)
                          AND slice_start = %s
                          AND slice_end = %s
                        GROUP BY token
                        """,
                        (tokens, slice_start, slice_end),
                    )
                    freq_map = {r[0]: r[1] for r in cur.fetchall()}

            for token, sim in top_neighbors:
                vec = get_vector(conn, token, slice_start, slice_end)
                if vec is None:
                    continue

                all_slices[slice_key].append({
                    "token": token,
                    "vector": vec,
                    "sim": sim,
                    "freq": freq_map.get(token) if include_freq else None,
                })

    return all_slices


def cluster_slice(neighbours: List[Neighbour]) -> np.ndarray:
    X = np.array([n["vector"] for n in neighbours])
    return hdbscan.HDBSCAN(
        min_cluster_size=3,
        metric="euclidean",
    ).fit_predict(X)

def assemble_clusters(
    neighbours: List[Neighbour],
    labels: np.ndarray,
    mass_modes: List[str],
) -> Dict[int, Dict[str, Any]]:
    clusters: Dict[int, List[Neighbour]] = defaultdict(list)

    for n, label in zip(neighbours, labels, strict=True):
        clusters[label].append(n)

    out: Dict[int, Dict[str, Any]] = {}

    for cid, members in clusters.items():
        out[cid] = {
            "tokens": [n["token"] for n in members],
            "masses": {
                mode: MASS_FUNCTIONS[mode](members)
                for mode in mass_modes
            },
        }

    return out

def cluster_slices(
    all_slices: Dict[str, List[Neighbour]],
    *,
    mass_modes: List[str],
) -> Dict[str, Any]:

    results: Dict[str, Any] = {}

    for slice_key, neighbours in all_slices.items():
        if not neighbours:
            results[slice_key] = {}
            continue

        labels = cluster_slice(neighbours)
        clusters = assemble_clusters(neighbours, labels, mass_modes)
        results[slice_key] = clusters

        logger.info(f"Slice {slice_key} clusters:")
        for cid, data in clusters.items():
            mass_str = ", ".join(
                f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in data["masses"].items()
            )
            if cid == -1:
                logger.info(f"  Outliers ({mass_str})")
            else:
                logger.info(f"  Cluster {cid} ({mass_str})")

    return results


def convert_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): convert_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_keys(v) for v in obj]
    return obj


def main():
    concepts = [TARGET] if TARGET else list(config.CONCEPT_SETS.keys())
    logger.info(f"Starting usage clustering for concepts: {concepts}")

    all_results: Dict[str, Any] = {}

    for concept in concepts:
        logger.info(f"Processing concept '{concept}'")
        all_slices = collect_vectors(
            concept,
            include_freq=("freq" in MASS_MODES or "weighted" in MASS_MODES),
        )
        cluster_results = cluster_slices(
            all_slices,
            mass_modes=MASS_MODES,
        )
        all_results[concept] = cluster_results

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(convert_keys(all_results), f, indent=2)

    logger.info(f"Wrote merged cluster results to {OUT_FILE}")

if __name__ == "__main__":
    main()
