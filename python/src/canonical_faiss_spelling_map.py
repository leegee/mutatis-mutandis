#!/usr/bin/env python
"""
canonical_faiss_spelling_map.py (optimized + diagnostics)

Pipeline:
- Load fastText model
- Compute all token vectors once
- Build weighted canonical vectors
- Build FAISS index
- Expand canonicals (orthographic + optional distance weighting)
- Run semantic integrity diagnostics and produce SVG plots
- Apply orthographic filter and insert into DB
- Save FAISS index

Canonical centroid construction:
    c⃗ = ( Σᵢ fᵢ · v⃗ᵢ ) / Σᵢ fᵢ

Optional neighbour distance weighting:
    wₜ = 1 / (1 + dₜ)

Semantic diagnostics tunables at top.
"""

from __future__ import annotations
from pathlib import Path
import argparse
import logging
from typing import Any, Dict, List, Tuple

import fasttext
import numpy as np
import faiss
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cosine

from lib import eebo_db, eebo_config as config
from lib.eebo_logging import logger

# ================= SEMANTIC INTEGRITY CONTROLS =================
MAX_CLUSTER_RADIUS = 0.35
MAX_INTERNAL_DISPERSION = 0.40
SUBCLUSTER_DISTANCE_THRESHOLD = 0.35
LEAKAGE_SCORE_THRESHOLD = 1.0

# ================================================================

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

def build_weighted_canonical_vectors(
    token_index: Dict[str, int],
    token_vectors: np.ndarray,
    token_counts: Dict[str, int],
) -> Tuple[List[str], np.ndarray]:
    canonicals: List[str] = []
    canonical_vectors: List[np.ndarray] = []

    for canonical, rule in config.KEYWORDS_TO_NORMALISE.items():
        allowed = set(rule.get("allowed_variants", []))
        allowed.add(canonical)
        present_tokens = [t for t in allowed if t in token_index]

        if not present_tokens:
            logger.warning(
                "Canonical '%s': no allowed variants found in corpus; skipping",
                canonical,
            )
            continue

        weighted_vectors = []
        weights = []

        for t in present_tokens:
            freq = token_counts.get(t, 1)
            weighted_vectors.append(token_vectors[token_index[t]] * freq)
            weights.append(freq)

        centroid = np.sum(weighted_vectors, axis=0) / np.sum(weights)
        canonicals.append(canonical)
        canonical_vectors.append(centroid)

    if not canonical_vectors:
        raise RuntimeError("No canonical vectors could be constructed")

    return canonicals, np.vstack(canonical_vectors)

def build_faiss_index(vectors: np.ndarray) -> Any:
    index: Any = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index

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
            # Gentle roll off:
            # score = float(1.0 / (1.0 + d)) if distance_weight_neighbors else float(d)
            # Threshold inverse distance:
            score = float(1.0 / (1 + d)) if d < 0.25 else 0.0  # ignore neighbors beyond 0.25
            # Inverse power law
            # score = float(1.0 / ((1 + d)**2))  # steeper than 1/(1+d)
            # Exponential decay:
            # score = float(np.exp(-d))  # d=0 → 1.0, d=1 → 0.367, d=2 → 0.135
            neighbors.append((token, score))

        neighbors.sort(key=lambda x: x[1], reverse=distance_weight_neighbors)
        results[canonical] = neighbors

        logger.info(
            "Canonical '%s': %d neighbors after exclusions",
            canonical,
            len(neighbors),
        )

    return results

# ----------------- Semantic Diagnostics -------------------

def mean_cosine_distance(vec, vecs):
    return np.mean([cosine(vec, v) for v in vecs])

def mean_pairwise_distance(vecs):
    dists = []
    for i in range(len(vecs)):
        for j in range(i+1, len(vecs)):
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

    plt.figure(figsize=(8,6))
    plt.scatter(radii, cohesion)
    plt.axvline(MAX_CLUSTER_RADIUS, linestyle="--", color='red')
    plt.axhline(MAX_INTERNAL_DISPERSION, linestyle="--", color='red')
    plt.xlabel("Cluster Radius (canonical → variants)")
    plt.ylabel("Internal Cohesion (variant ↔ variant)")
    plt.title("Canonical Integrity Diagnostics")
    out_path = out_dir / "canonical_integrity_diagnostics.svg"
    plt.savefig(out_path, format="svg")
    plt.close()
    logger.info(f"Canonical integrity plot saved to {out_path}")

def run_canonical_integrity_diagnostics(expansion, model):
    logger.info("Running canonical integrity diagnostics")
    flagged = []
    plot_data = []

    for canonical, neighbors in expansion.items():
        variants = [v for v, _ in neighbors]
        if canonical not in model or not variants:
            continue
        canon_vec = model.get_word_vector(canonical)
        vecs = [model.get_word_vector(v) for v in variants if v in model]
        if not vecs:
            continue
        radius = mean_cosine_distance(canon_vec, vecs)
        cohesion = mean_pairwise_distance(vecs)
        subclusters = count_subclusters(vecs)
        leakage_score = (
            radius / MAX_CLUSTER_RADIUS
            + cohesion / MAX_INTERNAL_DISPERSION
            + (subclusters - 1)
        )
        plot_data.append((canonical, radius, cohesion))
        logger.info(
            "Diagnostic of `%s` | radius=%.3f cohesion=%.3f subclusters=%d score=%.2f",
            canonical, radius, cohesion, subclusters, leakage_score
        )
        if leakage_score > LEAKAGE_SCORE_THRESHOLD:
            flagged.append((canonical, leakage_score))

    generate_diagnostic_plots(plot_data)

    if flagged:
        logger.warning("Semantic leakage detected in canonicals:")
        for c, s in flagged:
            logger.warning("  %s (score=%.2f)", c, s)
    else:
        logger.info("All canonicals passed integrity diagnostics")

# ----------------- DB Helpers -------------------

def insert_spelling_map(conn: Any, variant: str, canonical: str, dry: bool) -> None:
    if dry:
        logger.debug(f"[DRY] spelling_map: {variant} -> {canonical}")
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO spelling_map (variant, canonical, concept_type)
            VALUES (%s, %s, 'orthographic')
            ON CONFLICT (variant)
            DO UPDATE SET canonical = EXCLUDED.canonical;
            """,
            (variant, canonical),
        )

def update_tokens_canonical(conn: Any, variant: str, canonical: str, dry: bool) -> None:
    if dry:
        logger.debug(f"[DRY] tokens.canonical: {variant} -> {canonical}")
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE tokens
            SET canonical = %s
            WHERE token = %s;
            """,
            (canonical, variant),
        )

# ----------------- Main -------------------

def main(dry: bool, top_k: int, distance_weight_neighbors: bool) -> None:
    logger.info("Loading global fastText model")
    model = fasttext.load_model(str(config.FASTTEXT_GLOBAL_MODEL_PATH))

    logger.info("Fetching tokens and counts from database")
    token_counts: Dict[str,int] = {}
    tokens: List[str] = []

    with eebo_db.get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT token, COUNT(*) FROM tokens GROUP BY token")
        for token, count in cur:
            token_counts[token] = count
            tokens.append(token)

    token_index = {t: i for i, t in enumerate(tokens)}

    logger.info(f"Computing vectors for {len(tokens)} tokens")
    token_vectors = np.array(
        [model.get_word_vector(t) for t in tokens], dtype=np.float32
    )

    logger.info("Building weighted canonical vectors")
    canonicals, canonical_vectors = build_weighted_canonical_vectors(
        token_index, token_vectors, token_counts
    )

    logger.info("Building FAISS index")
    index = build_faiss_index(token_vectors)

    faiss_path = Path(config.FAISS_CANONICAL_INDEX_PATH)
    logger.info(f"Saving FAISS index to {faiss_path}")
    faiss.write_index(index, str(faiss_path))

    logger.info("Expanding canonicals via FAISS")
    expansion = expand_canonicals_with_faiss(
        canonicals=canonicals,
        canonical_vectors=canonical_vectors,
        index=index,
        tokens=tokens,
        top_k=top_k,
        distance_weight_neighbors=distance_weight_neighbors,
    )

    # ----------------- RUN DIAGNOSTICS -------------------
    run_canonical_integrity_diagnostics(expansion, model)

    with eebo_db.get_connection(application_name="canonical_spelling_map") as conn:
        for canonical, neighbors in expansion.items():
            for variant, _score in neighbors:
                if not is_reasonable_orthographic_variant(variant, canonical):
                    continue
                insert_spelling_map(conn, variant, canonical, dry)
                update_tokens_canonical(conn, variant, canonical, dry)

        if not dry:
            conn.commit()
            logger.info("Database updated")

    logger.info("Canonical expansion + spelling map complete")

# ----------------- CLI -------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true", help="Dry run, no DB writes")
    parser.add_argument("--top_k", type=int, default=config.TOP_K)
    parser.add_argument(
        "--distance_weight_neighbors",
        action="store_true",
        help="Weight FAISS neighbours by inverse distance",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(
        dry=args.dry,
        top_k=args.top_k,
        distance_weight_neighbors=args.distance_weight_neighbors,
    )
