#!/usr/bin/env python
# build_canonical_map_faiss_keywords.py

"""
Canonical spelling map builder (targeted to keywords) using fastText + FAISS.

Design notes
------------
- FAISS index is still full corpus, but only target keywords are canonicalized.
- Orthographic variation only: Levenshtein distance <= MAX_DIST.
- Multiprocessing + drain_progress for keyword chunks.
"""

from typing import Any, List, Dict, Tuple

import fasttext
import numpy as np
import faiss
import Levenshtein
from multiprocessing import Manager, Pool, cpu_count
from tqdm import tqdm
import time

from lib.drain_progress import drain_progress
import lib.eebo_config as config
import lib.eebo_db as eebo_db
from lib.eebo_logging import logger

MIN_FREQUENCY = 1
TOP_K = config.TOP_K
MAX_DIST = 2


def get_corpus_tokens(conn: Any) -> List[str]:
    """Fetch all unique tokens meeting MIN_FREQUENCY for FAISS index."""
    tokens: List[str] = []
    with conn.cursor() as cur:
        cur.execute(
            "SELECT token, COUNT(*) AS freq FROM tokens GROUP BY token;"
        )
        for token, freq in cur:
            if freq >= MIN_FREQUENCY:
                tokens.append(token)
    logger.info("Corpus tokens for FAISS: %d", len(tokens))
    return tokens


def get_target_keywords(conn: Any) -> List[str]:
    """Fetch only the target keywords to canonicalize."""
    keywords: List[str] = []
    with conn.cursor() as cur:
        cur.execute("SELECT token FROM keywords;")  # adjust table as needed
        keywords = [row[0] for row in cur]
    logger.info("Target keywords: %d", len(keywords))
    return keywords


def compute_token_vectors(model: Any, tokens: List[str]) -> np.ndarray:
    """Compute fastText vectors."""
    dim = model.get_dimension()
    vectors = np.zeros((len(tokens), dim), dtype=np.float32)
    for i, token in enumerate(tokens):
        vectors[i] = model.get_word_vector(token)
    return vectors


def build_faiss_index(vectors: np.ndarray) -> Any:
    dim = vectors.shape[1]
    index: Any = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index


def process_leven_chunk(
    args: Tuple[List[str], np.ndarray, List[str], Any]
) -> Dict[str, str]:
    """Worker for Levenshtein filtering + canonical assignment for keyword chunk."""
    keywords_chunk, neighbor_indices, all_tokens, counter = args
    batch_map: Dict[str, str] = {}

    for i, kw in enumerate(keywords_chunk):
        cluster = [kw]
        for idx in neighbor_indices[i]:
            neighbor = all_tokens[idx]
            if Levenshtein.distance(kw, neighbor) <= MAX_DIST:
                cluster.append(neighbor)
        canonical = min(cluster)
        for variant in cluster:
            batch_map[variant] = canonical
        counter.value += 1

    return batch_map


def merge_dicts(dicts: List[Dict[str, str]]) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for d in dicts:
        merged.update(d)
    return merged


def persist_spelling_map(
    conn: Any,
    spelling_map: Dict[str, str],
    batch_size: int = 5000,
) -> None:
    """Persist spelling map to DB."""
    logger.info("Persisting %d variants (batch=%d)", len(spelling_map), batch_size)
    items = list(spelling_map.items())

    with conn.transaction():
        with conn.cursor() as cur:
            for i in range(0, len(items), batch_size):
                batch = items[i : i + batch_size]
                cur.executemany(
                    """
                    INSERT INTO spelling_map (variant, canonical, concept_type)
                    VALUES (%s, %s, 'orthographic')
                    ON CONFLICT (variant) DO UPDATE
                    SET canonical = EXCLUDED.canonical;
                    """,
                    batch,
                )
    logger.info("Canonical map persisted")


def main() -> None:
    logger.info("Loading global fastText model")
    model = fasttext.load_model(str(config.FASTTEXT_GLOBAL_MODEL_PATH))

    with eebo_db.get_connection() as conn:
        # Full vocabulary for FAISS index
        all_tokens = get_corpus_tokens(conn)

    # Compute embeddings for all tokens
    logger.info("Computing token vectors for full vocabulary")
    all_vectors = compute_token_vectors(model, all_tokens)

    # Build FAISS index over full vocab
    logger.info("Building FAISS index over full vocabulary")
    faiss_index = build_faiss_index(all_vectors)

    # Keywords to canonicalise (flatten sets into a single list)
    keywords = [kw for kws in config.KEYWORDS_TO_NORMALISE.values() for kw in kws]
    logger.info("Computing token vectors for %d target keywords", len(keywords))
    keyword_vectors = compute_token_vectors(model, keywords)

    # Query nearest neighbors for keywords only
    logger.info("Querying top-%d neighbors for %d keywords", TOP_K, len(keywords))
    _, neighbor_indices = faiss_index.search(keyword_vectors, TOP_K)

    # Multiprocessing Levenshtein filtering
    logger.info("Levenshtein filtering (multiprocessing)")
    num_cores = min(8, cpu_count())
    chunk_size = (len(keywords) + num_cores - 1) // num_cores

    manager = Manager()
    counter = manager.Value("i", 0)

    chunks: List[Tuple[List[str], np.ndarray, List[str], Any]] = [
        (
            keywords[i : i + chunk_size],      # chunk of keywords
            neighbor_indices[i : i + chunk_size],
            all_tokens,                        # full vocab for lookup
            counter,
        )
        for i in range(0, len(keywords), chunk_size)
    ]

    with Pool(num_cores) as pool:
        results_async = [pool.apply_async(process_leven_chunk, (chunk,)) for chunk in chunks]

        drain_progress(
            total=len(keywords),
            counter=counter,
            workers=results_async,
            pbar_factory=lambda total: tqdm(total=total, desc="Clustering keywords"),
            poll=lambda: time.sleep(0.1),
        )

        spelling_maps = [r.get() for r in results_async]

    # Merge and persist
    spelling_map = merge_dicts(spelling_maps)
    with eebo_db.get_connection() as conn:
        persist_spelling_map(conn, spelling_map)

    logger.info("Targeted canonicalisation complete")


if __name__ == "__main__":
    main()
