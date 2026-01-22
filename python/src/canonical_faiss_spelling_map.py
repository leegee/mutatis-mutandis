#!/usr/bin/env python
"""
canonical_faiss_spelling_map.py (optimized)

Merged Phase 3 + Phase 4 with single computation of token vectors:

- Load fastText model
- Compute all token vectors once
- Build weighted canonical vectors
- Build FAISS index
- Expand canonicals
- Apply orthographic filter and insert into DB
- Save FAISS index
"""

from __future__ import annotations
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Tuple

import fasttext
import numpy as np
import faiss

from lib import eebo_db, eebo_config as config
from lib.eebo_logging import logger


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
    token_vectors: np.ndarray
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
                canonical
            )
            continue

        variant_vectors = np.vstack([token_vectors[token_index[t]] for t in present_tokens])
        centroid = variant_vectors.mean(axis=0)
        canonicals.append(canonical)
        canonical_vectors.append(centroid)

    if not canonical_vectors:
        raise RuntimeError("No canonical vectors could be constructed")

    return canonicals, np.vstack(canonical_vectors)

def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatL2:
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index

def expand_canonicals_with_faiss(
    canonicals: List[str],
    canonical_vectors: np.ndarray,
    index: faiss.IndexFlatL2,
    tokens: List[str],
    top_k: int
) -> Dict[str, List[Tuple[str, float]]]:
    results: Dict[str, List[Tuple[str, float]]] = {}
    distances, indices = index.search(canonical_vectors, top_k)

    for row, canonical in enumerate(canonicals):
        rule = config.KEYWORDS_TO_NORMALISE[canonical]
        false_positives = set(rule.get("false_positives", []))

        neighbors: List[Tuple[str, float]] = []
        for dist, idx in zip(distances[row], indices[row]):
            token = tokens[idx]
            if token == canonical or token in false_positives:
                continue
            neighbors.append((token, float(dist)))
        results[canonical] = neighbors
        logger.info(
            "Canonical '%s': %d neighbors after exclusions",
            canonical,
            len(neighbors)
        )

    return results


def insert_spelling_map(conn, variant: str, canonical: str, dry: bool):
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
            (variant, canonical)
        )

def update_tokens_canonical(conn, variant: str, canonical: str, dry: bool):
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
            (canonical, variant)
        )


def main(dry: bool, top_k: int):
    # Load fastText model
    logger.info("Loading global fastText model")
    model = fasttext.load_model(str(config.FASTTEXT_GLOBAL_MODEL_PATH))

    # Fetch corpus tokens
    with eebo_db.get_connection() as conn:
        tokens: List[str] = []
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT token FROM tokens")
        for (tok,) in cur:
            tokens.append(tok)
    token_index = {t: i for i, t in enumerate(tokens)}

    # Compute all token vectors **once**
    logger.info(f"Computing vectors for {len(tokens)} tokens")
    token_vectors = np.array([model.get_word_vector(t) for t in tokens], dtype=np.float32)

    # Build canonical vectors
    logger.info("Building weighted canonical vectors")
    canonicals, canonical_vectors = build_weighted_canonical_vectors(token_index, token_vectors)

    # Build FAISS index over full vocabulary
    logger.info("Building FAISS index")
    index = build_faiss_index(token_vectors)

    # Save FAISS index
    faiss_path = Path(config.FAISS_CANONICAL_INDEX_PATH)
    logger.info(f"Saving FAISS index to {faiss_path}")
    faiss.write_index(index, str(faiss_path))

    # Expand canonicals
    logger.info("Expanding canonicals via FAISS")
    expansion = expand_canonicals_with_faiss(
        canonicals=canonicals,
        canonical_vectors=canonical_vectors,
        index=index,
        tokens=tokens,
        top_k=top_k
    )

    # Insert into spelling_map and update tokens
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true", help="Dry run, no DB writes")
    parser.add_argument("--top_k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(dry=args.dry, top_k=args.top_k)
