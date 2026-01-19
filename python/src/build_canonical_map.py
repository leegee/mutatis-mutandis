#!/usr/bin/env python
# build_canonical_map.py
"""
"Normalisation": equivalence-mapping of orthological variants.

Build a canonical spelling_map from a global fastText model with Levenshtein distance.
Stores mapping in the eebo database for downstream slicing and drift analysis.
"""

import fasttext
from tqdm import tqdm
import Levenshtein  # pip install python-Levenshtein

import lib.eebo_config as config
import lib.eebo_db as eebo_db
from lib.eebo_logging import logger


def get_corpus_tokens(conn):
    """
    Fetch all unique tokens from the corpus.
    """
    tokens = []
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT token FROM tokens;")
        for row in cur:
            tokens.append(row[0])
    logger.info(f"Found {len(tokens)} unique tokens in the corpus")
    return tokens


def build_variant_clusters(model, tokens, top_k=30, max_dist=2):
    """
    Returns a dict mapping variant -> canonical.
    """
    spelling_map = {}
    logger.info(f"Building variant clusters (top_k={top_k}, max_dist={max_dist})")

    for token in tqdm(tokens):
        if token in spelling_map:
            continue  # already assigned

        try:
            neighbors = model.get_nearest_neighbors(token, k=top_k)
        except KeyError:
            continue

        # Include the token itself
        cluster = [token]

        for _sim, neighbor in neighbors:
            # Only orthographic-like neighbours
            if Levenshtein.distance(token, neighbor) <= max_dist:
                cluster.append(neighbor)

        # Decide canonical form (here: lexically smallest)
        canonical = min(cluster)

        # Assign all variants to canonical
        for variant in cluster:
            spelling_map[variant] = canonical

    return spelling_map


def persist_spelling_map(conn, spelling_map):
    """
    Write spelling_map to DB.
    """
    logger.info(f"Persisting {len(spelling_map)} variants to DB")

    with conn.transaction():
        with conn.cursor() as cur:
            for variant, canonical in spelling_map.items():
                cur.execute(
                    """
                    INSERT INTO spelling_map(variant, canonical, concept_type)
                    VALUES (%s, %s, 'orthographic')
                    ON CONFLICT (variant) DO UPDATE SET canonical = EXCLUDED.canonical;
                    """,
                    (variant, canonical),
                )
    logger.info("Canonical map persisted")


def main():
    # Load global model
    logger.info("Loading global fastText model")
    model_path = config.FASTTEXT_GLOBAL_MODEL_PATH
    model = fasttext.load_model(str(model_path))

    # Get all corpus tokens
    with eebo_db.get_connection() as conn:
        tokens = get_corpus_tokens(conn)
        spelling_map = build_variant_clusters(model, tokens, top_k=config.TOP_K)
        persist_spelling_map(conn, spelling_map)

    logger.info("Canonical spelling map construction complete")


if __name__ == "__main__":
    main()
