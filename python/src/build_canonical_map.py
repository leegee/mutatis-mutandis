#!/usr/bin/env python
# build_canonical_map_faiss_clean.py

"""
Canonical spelling map builder using fastText + FAISS for EEBO-TCP.

- Orders-of-magnitude faster than repeated get_nearest_neighbors calls.
- Filters neighbors by Levenshtein distance ≤ MAX_DIST.
- Uses batch DB inserts (~5000 rows per transaction).
- Multiprocessing for Levenshtein filtering.
- Per-token progress bar for long-running jobs.
- Fully typed and readable; no type-ignore comments needed.
"""

from typing import Any, List, Dict, Tuple
import fasttext
import numpy as np
import faiss
from tqdm import tqdm
import Levenshtein
from multiprocessing import Pool, cpu_count

import lib.eebo_config as config
import lib.eebo_db as eebo_db
from lib.eebo_logging import logger

MIN_FREQUENCY = 1
TOP_K = config.TOP_K
MAX_DIST = 2


def get_corpus_tokens(conn) -> List[str]:
    """Fetch all unique tokens from the corpus meeting MIN_FREQUENCY."""
    tokens: List[str] = []
    with conn.cursor() as cur:
        cur.execute("SELECT token, COUNT(*) as freq FROM tokens GROUP BY token;")
        for token, freq in cur:
            if freq >= MIN_FREQUENCY:
                tokens.append(token)
    logger.info(f"Found {len(tokens)} unique tokens meeting MIN_FREQUENCY={MIN_FREQUENCY}")
    return tokens


def compute_token_vectors(model: Any, tokens: List[str]) -> np.ndarray:
    """Compute fastText embedding vectors for all tokens."""
    dim: int = model.get_dimension()
    vectors: np.ndarray = np.zeros((len(tokens), dim), dtype=np.float32)
    for i, token in enumerate(tokens):
        vectors[i, :] = model.get_word_vector(token)
    return vectors


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatL2:
    """Build a FAISS L2 index on the token vectors."""
    dim: int = vectors.shape[1]
    index: Any = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index


def process_leven_chunk(args: Tuple[List[str], np.ndarray, List[str]]) -> Dict[str, str]:
    """
    Worker function for Levenshtein filtering and canonical assignment.

    Args:
        tokens: List of tokens in this chunk.
        neighbor_indices: Array of FAISS neighbor indices for this chunk.
        all_tokens: Full token list for lookup.
    Returns:
        Dict mapping variant -> canonical.
    """
    tokens, neighbor_indices, all_tokens = args
    batch_map: Dict[str, str] = {}
    for i, token in enumerate(tokens):
        cluster: List[str] = [token]
        for idx in neighbor_indices[i]:
            neighbor_token = all_tokens[idx]
            if Levenshtein.distance(token, neighbor_token) <= MAX_DIST:
                cluster.append(neighbor_token)
        canonical: str = min(cluster)
        for variant in cluster:
            batch_map[variant] = canonical
    return batch_map


def merge_dicts(dict_list: List[Dict[str, str]]) -> Dict[str, str]:
    """Merge multiple dictionaries. Later entries overwrite earlier ones."""
    merged: Dict[str, str] = {}
    for d in dict_list:
        merged.update(d)
    return merged


def persist_spelling_map(conn, spelling_map: Dict[str, str], batch_size: int = 5000) -> None:
    """Batch-insert the spelling map into the database."""
    logger.info(f"Persisting {len(spelling_map)} variants to DB in batches of {batch_size}")
    items: List[Tuple[str, str]] = list(spelling_map.items())
    with conn.transaction():
        with conn.cursor() as cur:
            for i in range(0, len(items), batch_size):
                batch: List[Tuple[str, str]] = items[i:i + batch_size]
                args: List[Tuple[str, str]] = [(variant, canonical) for variant, canonical in batch]
                cur.executemany(
                    """
                    INSERT INTO spelling_map(variant, canonical, concept_type)
                    VALUES (%s, %s, 'orthographic')
                    ON CONFLICT (variant) DO UPDATE SET canonical = EXCLUDED.canonical;
                    """,
                    args,
                )
    logger.info("Canonical map persisted")


def main() -> None:
    logger.info("Loading global fastText model")
    model: Any = fasttext.load_model(str(config.FASTTEXT_GLOBAL_MODEL_PATH))

    with eebo_db.get_connection() as conn:
        tokens: List[str] = get_corpus_tokens(conn)

    logger.info("Computing token vectors")
    token_vectors: np.ndarray = compute_token_vectors(model, tokens)

    logger.info("Building FAISS index")
    faiss_index: Any = build_faiss_index(token_vectors)

    logger.info(f"Querying top-{TOP_K} neighbors for all tokens")
    distances: np.ndarray
    neighbor_indices: np.ndarray
    distances, neighbor_indices = faiss_index.search(token_vectors, TOP_K)

    logger.info("Filtering neighbors by Levenshtein distance using multiprocessing")
    num_cores: int = min(8, cpu_count())
    chunk_size: int = (len(tokens) + num_cores - 1) // num_cores
    chunks: List[Tuple[List[str], np.ndarray, List[str]]] = [
        (tokens[i:i + chunk_size], neighbor_indices[i:i + chunk_size], tokens)
        for i in range(0, len(tokens), chunk_size)
    ]

    with Pool(num_cores) as pool:
        results = list(tqdm(pool.imap(process_leven_chunk, chunks), total=len(chunks)))

    spelling_map: Dict[str, str] = merge_dicts(results)

    with eebo_db.get_connection() as conn:
        persist_spelling_map(conn, spelling_map)

    logger.info("Canonical spelling map construction complete")


if __name__ == "__main__":
    main()
