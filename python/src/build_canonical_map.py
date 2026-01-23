#!/usr/bin/env python
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import fasttext
import faiss
import numpy as np

import lib.eebo_db as eebo_db
from lib.eebo_config import KEYWORDS_TO_NORMALISE, FASTTEXT_GLOBAL_MODEL_PATH, TOP_K
from lib.eebo_logging import logger


def get_corpus_tokens(conn: Any) -> List[str]:
    """Fetch all unique tokens from the database."""
    tokens: List[str] = []
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT token FROM tokens;")
        for (token,) in cur:
            tokens.append(token)
    return tokens


def compute_token_vectors(model: Any, tokens: List[str]) -> np.ndarray:
    """Compute fastText vectors for all tokens."""
    dim = model.get_dimension()
    vectors = np.zeros((len(tokens), dim), dtype=np.float32)
    for i, token in enumerate(tokens):
        vectors[i] = model.get_word_vector(token)
    return vectors


def build_faiss_index(vectors: np.ndarray) -> Any:
    """Build a flat L2 FAISS index over all token vectors."""
    index: Any = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index


def build_weighted_canonical_vectors(
    *,
    vectors: np.ndarray,
    token_index: Dict[str, int],
) -> Tuple[List[str], np.ndarray]:
    """
    Construct one vector per canonical head by averaging all
    attested allowed variants (including the canonical itself if present).
    """
    canonicals: List[str] = []
    canonical_vectors: List[np.ndarray] = []

    for canonical, rule in KEYWORDS_TO_NORMALISE.items():
        allowed = set(rule["allowed_variants"])
        allowed.add(canonical)

        present_tokens = [t for t in allowed if t in token_index]

        if not present_tokens:
            logger.warning(
                "Canonical '%s': no allowed variants found in corpus; skipping",
                canonical,
            )
            continue

        logger.info(
            "Canonical '%s' represented by %d variants: %s",
            canonical,
            len(present_tokens),
            ", ".join(sorted(present_tokens)),
        )

        variant_vectors = np.vstack([vectors[token_index[t]] for t in present_tokens])
        centroid = variant_vectors.mean(axis=0)
        canonicals.append(canonical)
        canonical_vectors.append(centroid)

    if not canonical_vectors:
        raise RuntimeError("No canonical vectors could be constructed")

    return canonicals, np.vstack(canonical_vectors)


def expand_canonicals_with_faiss(
    *,
    canonicals: List[str],
    canonical_vectors: np.ndarray,
    index: Any,
    tokens: List[str],
    k: int,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    For each canonical vector, query FAISS and return nearest neighbors,
    explicitly excluding the canonical head itself and known false positives.
    """
    results: Dict[str, List[Tuple[str, float]]] = {}
    distances, indices = index.search(canonical_vectors, k)

    for row, canonical in enumerate(canonicals):
        rule = KEYWORDS_TO_NORMALISE[canonical]
        false_positives = rule["false_positives"]

        neighbors: List[Tuple[str, float]] = []

        for dist, idx in zip(distances[row], indices[row], strict=True):
            token = tokens[idx]

            if token == canonical or token in false_positives:
                continue

            neighbors.append((token, float(dist)))

        results[canonical] = neighbors
        logger.info(
            "Canonical '%s': %d neighbors after exclusions",
            canonical,
            len(neighbors),
        )

    return results


def main(
    *,
    tokens: List[str],
    vectors: np.ndarray,
    token_index: Dict[str, int],
    k: int = 30,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    End-to-end canonical expansion:
      1. Build weighted canonical vectors
      2. Build FAISS index over full vocabulary
      3. Expand canonicals with explicit false-positive exclusion
    """
    logger.info("Building weighted canonical vectors")
    canonicals, canonical_vectors = build_weighted_canonical_vectors(
        vectors=vectors,
        token_index=token_index,
    )

    logger.info("Persisting canonical vectors to canonical_centroids")

    with eebo_db.get_connection(application_name="persist_canonicals") as conn:
        with conn.cursor() as cur:
            for canonical, vec in zip(canonicals, canonical_vectors, strict=True):
                cur.execute(
                    """
                    INSERT INTO canonical_centroids
                        (canonical, vector, weighting_scheme, source_model)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (canonical) DO UPDATE
                        SET vector = EXCLUDED.vector,
                            weighting_scheme = EXCLUDED.weighting_scheme,
                            source_model = EXCLUDED.source_model;
                    """,
                    (canonical, vec.tolist(), "average_variants", "fasttext_global")
                )
        conn.commit()
    logger.info("Persisted %d canonical vectors to canonical_centroids", len(canonicals))

    logger.info("Building FAISS index")
    index = build_faiss_index(vectors)

    logger.info("Expanding canonicals via FAISS (k=%d)", k)
    return expand_canonicals_with_faiss(
        canonicals=canonicals,
        canonical_vectors=canonical_vectors,
        index=index,
        tokens=tokens,
        k=k,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logger.info("Loading global fastText model")
    model = fasttext.load_model(str(FASTTEXT_GLOBAL_MODEL_PATH))

    logger.info("Fetching corpus tokens")
    with eebo_db.get_connection() as conn:
        tokens = get_corpus_tokens(conn)

    token_index = {t: i for i, t in enumerate(tokens)}

    logger.info("Computing token vectors")
    vectors = compute_token_vectors(model, tokens)

    logger.info("Running canonical expansion")
    results = main(
        tokens=tokens,
        vectors=vectors,
        token_index=token_index,
        k=TOP_K,
    )

    logger.info("Canonical expansion complete")
