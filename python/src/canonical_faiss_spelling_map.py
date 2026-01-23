#!/usr/bin/env python
"""
canonical_faiss_spelling_map.py
Phase 3: Canonical expansion + diagnostics (reuses Phase 2 FAISS index)
"""

from __future__ import annotations
from pathlib import Path
import argparse
import logging
from typing import Any, Dict, List, Optional, Protocol, Tuple

import fasttext
import numpy as np
import faiss
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cosine

from lib import eebo_db, eebo_config as config
from lib.eebo_logging import logger

# ---------------- Tunables ----------------
MAX_CLUSTER_RADIUS = 0.35
MAX_INTERNAL_DISPERSION = 0.40
SUBCLUSTER_DISTANCE_THRESHOLD = 0.35
LEAKAGE_SCORE_THRESHOLD = 1.0
DISTANCE_CUTOFF = 0.25
# ------------------------------------------

class FastTextModel(Protocol):
    def get_word_vector(self, word: str) -> np.ndarray: ...

def is_reasonable_orthographic_variant(candidate: str, canonical: str) -> bool:
    if candidate == canonical:
        return False
    if abs(len(candidate) - len(canonical)) > 3:
        return False
    if canonical[:4] not in candidate and candidate[:4] not in canonical:
        return False
    for bad_prefix in ("un", "in", "dis", "non"):
        if candidate.startswith(bad_prefix) and not canonical.startswith(bad_prefix):
            return False
    return True


# ---------------- Canonical Centroids ----------------

def build_weighted_canonical_vectors(
    token_index: Dict[str, int],
    token_vectors: Optional[np.ndarray],
    token_counts: Dict[str, int],
    model: FastTextModel,
    distance_weight: bool = False,
) -> Tuple[List[str], np.ndarray]:

    canonicals: List[str] = []
    canonical_vectors: List[np.ndarray] = []

    for canonical, rule in config.KEYWORDS_TO_NORMALISE.items():
        allowed = set(rule.get("allowed_variants", []))
        allowed.add(canonical)
        present_tokens = [t for t in allowed if t in token_index]

        if not present_tokens:
            logger.warning("Canonical '%s': no allowed variants found; skipping", canonical)
            continue

        weighted_vectors: List[np.ndarray] = []
        weights: List[float] = []

        for t in present_tokens:
            freq = token_counts.get(t, 1)

            if token_vectors is None:
                vec = model.get_word_vector(t)
                w = float(freq)
            else:
                vec = token_vectors[token_index[t]]
                w = float(freq)

                if distance_weight:
                    if canonical in token_index:
                        canon_vec = token_vectors[token_index[canonical]]
                        d = np.linalg.norm(vec - canon_vec)
                        if d > DISTANCE_CUTOFF:
                            continue
                        w *= float(np.exp(-d))

            weighted_vectors.append(vec)
            weights.append(w)

        if not weighted_vectors:
            logger.warning("Canonical '%s': all variants filtered out; skipping", canonical)
            continue

        centroid = np.average(
            np.vstack(weighted_vectors),
            axis=0,
            weights=np.array(weights),
        )

        canonicals.append(canonical)
        canonical_vectors.append(centroid)

    if not canonical_vectors:
        raise RuntimeError("No canonical vectors constructed")

    return canonicals, np.vstack(canonical_vectors)


# ---------------- FAISS Expansion ----------------

def expand_canonicals_with_faiss(
    canonicals: List[str],
    canonical_vectors: np.ndarray,
    index: Any,
    tokens: List[str],
    top_k: int,
    distance_weight_neighbors: bool = False,
) -> Dict[str, List[Tuple[str, float]]]:

    results: Dict[str, List[Tuple[str, float]]] = {}
    distances, indices = index.search(canonical_vectors, top_k)

    for row, canonical in enumerate(canonicals):
        rule = config.KEYWORDS_TO_NORMALISE[canonical]
        false_positives = set(rule.get("false_positives", []))
        neighbors: List[Tuple[str, float]] = []

        for d, idx in zip(distances[row], indices[row], strict=True):
            token = tokens[idx]
            if token == canonical or token in false_positives:
                continue

            # 🔬 Your experimental neighbour weighting (UNCHANGED)
            score = float(1.0 / ((1 + d) ** 2))
            neighbors.append((token, score))

        neighbors.sort(key=lambda x: x[1], reverse=distance_weight_neighbors)
        results[canonical] = neighbors
        logger.info("Canonical '%s': %d neighbors after exclusions", canonical, len(neighbors))

    return results


# ---------------- Diagnostics ----------------

def mean_cosine_distance(vec, vecs):
    return np.mean([cosine(vec, v) for v in vecs])

def mean_pairwise_distance(vecs):
    dists = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            dists.append(cosine(vecs[i], vecs[j]))
    return np.mean(dists) if dists else 0

def count_subclusters(vecs):
    if len(vecs) < 3:
        return 1
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=SUBCLUSTER_DISTANCE_THRESHOLD,
    )
    labels = model.fit_predict(vecs)
    return len(set(labels))

def generate_diagnostic_plots(plot_data):
    out_dir = Path(config.OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    radii = [r for _, r, _ in plot_data]
    cohesion = [c for _, _, c in plot_data]
    plt.figure(figsize=(8, 6))
    plt.scatter(radii, cohesion)
    plt.axvline(MAX_CLUSTER_RADIUS, linestyle="--", color="red")
    plt.axhline(MAX_INTERNAL_DISPERSION, linestyle="--", color="red")
    out_path = out_dir / "canonical_integrity_diagnostics.svg"
    plt.savefig(out_path, format="svg")
    plt.close()
    logger.info(f"Canonical integrity plot saved to {out_path}")

def run_canonical_integrity_diagnostics(expansion, model):
    plot_data = []
    flagged = []

    for canonical, neighbors in expansion.items():
        variants = [v for v, _ in neighbors if v in model]
        if not variants:
            continue

        canon_vec = model.get_word_vector(canonical)
        vecs = [model.get_word_vector(v) for v in variants]

        radius = mean_cosine_distance(canon_vec, vecs)
        cohesion = mean_pairwise_distance(vecs)
        subclusters = count_subclusters(vecs)
        leakage_score = radius / MAX_CLUSTER_RADIUS + cohesion / MAX_INTERNAL_DISPERSION + (subclusters - 1)

        plot_data.append((canonical, radius, cohesion))
        logger.info("Diagnostic `%s` | radius=%.3f cohesion=%.3f clusters=%d score=%.2f",
                    canonical, radius, cohesion, subclusters, leakage_score)

        if leakage_score > LEAKAGE_SCORE_THRESHOLD:
            flagged.append((canonical, leakage_score))

    generate_diagnostic_plots(plot_data)


# ---------------- Main ----------------

def main(dry: bool, top_k: int, distance_weight_neighbors: bool):
    logger.info("Loading fastText model")
    model: FastTextModel = fasttext.load_model(str(config.FASTTEXT_GLOBAL_MODEL_PATH))

    logger.info("Fetching tokens from DB")
    token_counts: Dict[str, int] = {}
    tokens: List[str] = []

    with eebo_db.get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT token, COUNT(*) FROM tokens GROUP BY token")
        for token, count in cur:
            token_counts[token] = count
            tokens.append(token)

    token_index = {t: i for i, t in enumerate(tokens)}

    logger.info("Loading FAISS index from Phase 2")
    index = faiss.read_index(str(config.FAISS_CANONICAL_INDEX_PATH))

    token_vectors = None
    if distance_weight_neighbors:
        logger.info("Computing token vectors for centroid weighting")
        token_vectors = np.array([model.get_word_vector(t) for t in tokens], dtype=np.float32)

    canonicals, canonical_vectors = build_weighted_canonical_vectors(
        token_index=token_index,
        token_vectors=token_vectors,
        token_counts=token_counts,
        model=model,
        distance_weight=distance_weight_neighbors,
    )

    expansion = expand_canonicals_with_faiss(
        canonicals, canonical_vectors, index, tokens, top_k, distance_weight_neighbors
    )

    run_canonical_integrity_diagnostics(expansion, model)

    logger.info("Pipeline complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--top_k", type=int, default=config.TOP_K)
    parser.add_argument("--distance_weight_neighbors", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(args.dry, args.top_k, args.distance_weight_neighbors)
