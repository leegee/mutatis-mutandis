#!/usr/bin/env python
"""
usage_cluster_tracker.py

Cluster diachronic neighbours of concepts (e.g., LAW) using HDBSCAN
and track cluster prevalence ("mass").

Outputs:

* JSON file with cluster assignments and cluster mass for each slice.
* Optional logging of cluster content for inspection.

Requires: hdbscan, numpy
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional

import hdbscan
import numpy as np

import lib.eebo_config as config
import lib.eebo_db as eebo_db
from lib.eebo_logging import logger
from lib.faiss_slices import load_slice_index, get_vector

# Set to None to cluster all concepts in CONCEPT_SETS
TARGET: Optional[str] = None

if TARGET:
    OUT_FILE = config.OUT_DIR / f"usage_clusters_{TARGET.lower()}.json"
else:
    OUT_FILE = config.OUT_DIR / "usage_clusters_all_concepts.json"

TOP_K = 100
SIM_THRESHOLD = 0.7


def collect_vectors(target: str) -> Dict[str, List[Dict[str, Any]]]:
    """Collect vectors for each slice for the target concept neighbours."""
    all_slices: Dict[str, List[Dict[str, Any]]] = {}

    with eebo_db.get_connection() as conn:
        for slice_start, slice_end in config.SLICES:
            slice_key = f"{slice_start}_{slice_end}"
            index, vocab = load_slice_index((slice_start, slice_end))
            all_slices[slice_key] = []

            seed = target.lower()
            vec = get_vector(conn, seed, slice_start, slice_end)
            if vec is None:
                logger.warning(f"No vector for probe '{seed}' in slice {slice_key}")
                continue

            # FAISS search
            D, Idx = index.search(vec.reshape(1, -1), TOP_K)
            top_neighbors = [
                (vocab[idx], float(sim))
                for sim, idx in zip(D[0], Idx[0], strict=True)
                if idx != -1
            ]

            # Remove known forms / false positives
            known_forms = config.CONCEPT_SETS[target].get("forms", set())
            false_positives = config.CONCEPT_SETS[target].get("false_positives", set())
            top_neighbors = [
                (token, sim)
                for token, sim in top_neighbors
                if token not in known_forms and token not in false_positives
            ]

            if not top_neighbors:
                continue

            # Store vectors for clustering
            for token, sim in top_neighbors:
                token_vec = get_vector(conn, token, slice_start, slice_end)
                if token_vec is not None:
                    all_slices[slice_key].append({"token": token, "vector": token_vec, "sim": sim})

    return all_slices


def cluster_slices(all_slices: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Cluster neighbours per slice using HDBSCAN and compute cluster mass."""
    cluster_results: Dict[str, Any] = {}

    for slice_key, neighbours in all_slices.items():
        if not neighbours:
            cluster_results[slice_key] = []
            continue

        X = np.array([n["vector"] for n in neighbours])
        tokens = [n["token"] for n in neighbours]

        clusterer = hdbscan.HDBSCAN(min_cluster_size=3, metric='euclidean')
        labels = clusterer.fit_predict(X)

        slice_clusters: Dict[int, List[str]] = defaultdict(list)
        cluster_mass: Dict[int, int] = {}  # <--- mypy annotation

        for token, label in zip(tokens, labels, strict=True):
            slice_clusters[label].append(token)

        # Compute mass for each cluster
        for label, toks in slice_clusters.items():
            cluster_mass[label] = len(toks)

        cluster_results[slice_key] = {
            "clusters": slice_clusters,
            "cluster_mass": cluster_mass
        }

        # Logging
        logger.info(f"Slice {slice_key} clusters:")
        for cid, toks in slice_clusters.items():
            mass = cluster_mass[cid]
            if cid == -1:
                logger.info(f"  Outliers ({mass} tokens): {toks}")
            else:
                logger.info(f"  Cluster {cid} ({mass} tokens): {toks}")

    return cluster_results


def convert_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): convert_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys(elem) for elem in obj]
    else:
        return obj


def main():
    logger.info("Starting usage cluster tracking")

    concepts_to_run: List[str] = [TARGET] if TARGET else list(config.CONCEPT_SETS.keys())
    all_results: Dict[str, Any] = {}

    for concept in concepts_to_run:
        logger.info(f"Processing concept '{concept}'")
        all_slices = collect_vectors(concept)
        cluster_results = cluster_slices(all_slices)
        all_results[concept] = cluster_results

    # Save JSON
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(convert_keys(all_results), f, indent=2)

    logger.info(f"Wrote cluster tracker results to {OUT_FILE}")


if __name__ == "__main__":
    main()
