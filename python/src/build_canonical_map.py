#!/usr/bin/env python
# build_canonical_map_faiss_clean.py

"""
Canonical spelling map builder using fastText + FAISS for EEBO-TCP.

Canonicalisation policy
-----------------------
KEYWORDS_TO_NORMALISE is authoritative.

For each canonical form:
- allowed_variants are ALWAYS mapped to the canonical
- false_positives are NEVER mapped, regardless of distance or embedding similarity
- FAISS + Levenshtein is used only to discover *additional orthographic variants*
  for the targeted canonical tokens, excluding false positives

Semantic equivalence is explicitly out of scope.
"""

from typing import Any, List, Dict, Tuple, Set

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


TOP_K = config.TOP_K
MAX_DIST = 2


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def get_corpus_tokens(conn: Any) -> List[str]:
    tokens: List[str] = []
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT token FROM tokens;")
        for (token,) in cur:
            tokens.append(token)
    return tokens


def compute_token_vectors(model: Any, tokens: List[str]) -> np.ndarray:
    dim = model.get_dimension()
    vectors = np.zeros((len(tokens), dim), dtype=np.float32)
    for i, token in enumerate(tokens):
        vectors[i] = model.get_word_vector(token)
    return vectors


def build_faiss_index(vectors: np.ndarray) -> Any:
    index: Any = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index


def process_target_chunk(
    args: Tuple[
        List[str],            # target canonicals
        np.ndarray,           # neighbor indices
        List[str],            # all tokens
        Dict[str, Set[str]],  # allowed_variants
        Dict[str, Set[str]],  # false_positives
        Any,                  # counter
    ]
) -> Dict[str, str]:

    (
        canonicals,
        neighbor_indices,
        all_tokens,
        allowed_variants,
        false_positives,
        counter,
    ) = args

    batch_map: Dict[str, str] = {}

    for i, canonical in enumerate(canonicals):
        cluster: Set[str] = {canonical}

        for idx in neighbor_indices[i]:
            candidate = all_tokens[idx]

            if candidate in false_positives.get(canonical, set()):
                continue

            if Levenshtein.distance(canonical, candidate) <= MAX_DIST:
                cluster.add(candidate)

        # force include allowed variants
        cluster |= allowed_variants.get(canonical, set())

        for variant in cluster:
            batch_map[variant] = canonical

        counter.value += 1

    return batch_map


def merge_dicts(dicts: List[Dict[str, str]]) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for d in dicts:
        merged.update(d)
    return merged


def persist_spelling_map(conn: Any, spelling_map: Dict[str, str]) -> None:
    with conn.transaction():
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO spelling_map (variant, canonical, concept_type)
                VALUES (%s, %s, 'orthographic')
                ON CONFLICT (variant)
                DO UPDATE SET canonical = EXCLUDED.canonical;
                """,
                spelling_map.items(),
            )



def main() -> None:
    logger.info("Loading global fastText model")
    model = fasttext.load_model(str(config.FASTTEXT_GLOBAL_MODEL_PATH))

    with eebo_db.get_connection() as conn:
        all_tokens = get_corpus_tokens(conn)

    token_index = {t: i for i, t in enumerate(all_tokens)}

    logger.info("Computing token vectors")
    vectors = compute_token_vectors(model, all_tokens)

    logger.info("Building FAISS index")
    index = build_faiss_index(vectors)

    # Target canonicals only
    canonicals = list(config.KEYWORDS_TO_NORMALISE.keys())
    canonical_vectors = np.vstack(
        [vectors[token_index[c]] for c in canonicals]
    )

    logger.info("Querying FAISS neighbours for targeted canonicals only")
    _, neighbor_indices = index.search(canonical_vectors, TOP_K)

    manager = Manager()
    counter = manager.Value("i", 0)

    num_cores = min(8, cpu_count())
    chunk_size = (len(canonicals) + num_cores - 1) // num_cores

    chunks = [
        (
            canonicals[i : i + chunk_size],
            neighbor_indices[i : i + chunk_size],
            all_tokens,
            {k: v["allowed_variants"] for k, v in config.KEYWORDS_TO_NORMALISE.items()},
            {k: v["false_positives"] for k, v in config.KEYWORDS_TO_NORMALISE.items()},
            counter,
        )
        for i in range(0, len(canonicals), chunk_size)
    ]

    with Pool(num_cores) as pool:
        results = [
            pool.apply_async(process_target_chunk, (chunk,))
            for chunk in chunks
        ]

        drain_progress(
            total=len(canonicals),
            counter=counter,
            workers=results,
            pbar_factory=lambda total: tqdm(
                total=total,
                desc="Canonicalising targeted keywords",
            ),
            poll=lambda: time.sleep(0.1),
        )

        spelling_maps = [r.get() for r in results]

    spelling_map = merge_dicts(spelling_maps)

    with eebo_db.get_connection() as conn:
        persist_spelling_map(conn, spelling_map)

    logger.info(
        "Canonical map complete (%d variants, %d canonicals)",
        len(spelling_map),
        len(canonicals),
    )


if __name__ == "__main__":
    main()
