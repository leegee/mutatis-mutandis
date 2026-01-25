#!/usr/bin/env python
"""
build_canonical_map.py

For each keyword in config.KEYWORDS_TO_NORMALISE:
- Find candidate variants in token_vectors
- Compute cosine similarity
- Filter by semantic similarity threshold
- Insert mappings into token_canonical_map table
"""

from pathlib import Path
import numpy as np
from tqdm import tqdm

import lib.eebo_config as config
import lib.eebo_db as eebo_db
from lib.eebo_logging import logger

from psycopg import sql

SIMILARITY_THRESHOLD = 0.7  # cosine similarity threshold
BATCH_SIZE = 5_000


def fetch_token_vectors(conn):
    """
    Load all token vectors from DB into memory as a dict {token: vector}.
    """
    logger.info("Fetching token vectors from DB")
    vectors = {}
    with conn.cursor() as cur:
        cur.execute("SELECT token, vector FROM token_vectors")
        for row in cur.fetchall():
            vectors[row[0]] = np.array(row[1], dtype=np.float32)
    logger.info(f"Fetched {len(vectors)} token vectors")
    return vectors


def compute_cosine(u: np.ndarray, v: np.ndarray) -> float:
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return float(np.dot(u, v) / (norm_u * norm_v))


def generate_mappings(vectors, slice_start=None, slice_end=None):
    """
    Generate mappings from token_vectors to KEYWORDS_TO_NORMALISE based on cosine similarity.
    """
    mappings = []
    for canonical, rules in tqdm(config.KEYWORDS_TO_NORMALISE.items(), desc="Keywords"):
        canonical_vec = vectors.get(canonical)
        if canonical_vec is None:
            logger.warning(f"Canonical keyword '{canonical}' not in token_vectors; skipping")
            continue

        for token, vec in vectors.items():
            if token == canonical:
                continue
            cosine = compute_cosine(canonical_vec, vec)
            if cosine >= SIMILARITY_THRESHOLD:
                # Respect allowed/false positives
                if rules.get("false_positives") and token in rules["false_positives"]:
                    continue
                if rules.get("allowed_variants") and token not in rules["allowed_variants"]:
                    continue
                mappings.append((token, canonical, slice_start, slice_end, cosine, "semantic"))

    logger.info(f"Generated {len(mappings)} candidate canonical mappings")
    return mappings


def store_mappings(conn, mappings):
    """
    Bulk insert mappings into canonical map table.
    """
    logger.info(f"Inserting {len(mappings)} mappings into DB")
    with conn.transaction():
        with conn.cursor() as cur:
            for i in range(0, len(mappings), BATCH_SIZE):
                batch = mappings[i:i+BATCH_SIZE]
                cur.executemany(
                    f"""INSERT INTO token_canonical_map
                    (variant_token, canonical_token, slice_start, slice_end, cosine_similarity, method)
                    VALUES
                    (%s, %s, %s, %s, %s, %s)
                    """,
                    batch
                )


def main():
    with eebo_db.get_connection() as conn:
        drop_indexes_token_canonical_map(conn)
        vectors = fetch_token_vectors(conn)
        # For now, global mapping (slice=None)
        mappings = generate_mappings(vectors)
        store_mappings(conn, mappings)

    logger.info("Creating indexes and primary key")
    with eebo_db.get_connection() as conn:
        eebo_db.create_indexes_token_canonical_map(conn)

    logger.info("Canonical mapping table ready")


if __name__ == "__main__":
    main()
