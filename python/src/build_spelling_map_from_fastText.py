#!/usr/bin/env python
"""
build_spelling_map_from_fastText.py

Stage 2: Generate the canonical spelling_map table from the global fastText model
and update tokens.canonical.

- Loads global fastText skipgram model (Stage 1)
- Queries nearest neighbours for all tokens
- Determines canonical form
- Inserts results into `spelling_map` table
- Updates `tokens.canonical` column
- Supports dry-run mode (--dry)
"""

from pathlib import Path
import argparse
from tqdm import tqdm
import fasttext
import numpy as np

from lib import eebo_db, eebo_config as config
from lib.eebo_logging import logger


def load_fasttext_model(model_path: Path):
    logger.info(f"Loading global fastText model from {model_path}")
    return fasttext.load_model(str(model_path))


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def get_corpus_tokens(conn):
    """Return the distinct token vocabulary from tokens table"""
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT token FROM tokens;")
        return [row[0] for row in cur.fetchall()]


def determine_canonical(neighbours):
    """
    Pick a canonical token from a set of variants.
    Strategy: first neighbour returned by fastText (highest similarity).
    """
    return neighbours[0][1]


def insert_spelling_map(conn, variant, canonical, dry: bool = False):
    """Insert one variant → canonical mapping into spelling_map table"""
    if dry:
        logger.debug(f"[DRY] Insert spelling_map: {variant} -> {canonical}")
        return
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO spelling_map(variant, canonical, concept_type)
            VALUES (%s, %s, 'orthographic')
            ON CONFLICT (variant) DO UPDATE SET canonical = EXCLUDED.canonical;
        """, (variant, canonical))


def update_tokens_canonical(conn, token, canonical, dry: bool = False):
    """Update tokens.canonical column"""
    if dry:
        logger.debug(f"[DRY] Update tokens.canonical: {token} -> {canonical}")
        return
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE tokens
            SET canonical = %s
            WHERE token = %s;
        """, (canonical, token))


def main():
    logger.info(
        f"Building the canonical `spelling_map` table from the global fastText model in {config.FASTTEXT_GLOBAL_MODEL_PATH}"
    )
    parser = argparse.ArgumentParser(
        description="Build canonical spelling_map from global fastText model and update tokens.canonical"
    )
    parser.add_argument("--model_path", type=str, default=str(config.FASTTEXT_GLOBAL_MODEL_PATH),
        help=f"Path to global fastText model (default: {config.FASTTEXT_GLOBAL_MODEL_PATH})"
    )
    parser.add_argument("--top_k", type=int, default=config.TOP_K,
        help=f"Number of nearest neighbours to consider for each token. Defaults to {config.TOP_K}"
    )
    parser.add_argument("--batch_size", type=int, default=config.CANONICALISATION_BATCH_SIZE,
        help=f"Batch size for processing tokens. Defaults to {config.CANONICALISATION_BATCH_SIZE}"
    )
    parser.add_argument("--dry", action="store_true", help="Dry run: do not commit to DB")
    args = parser.parse_args()

    if args.dry:
        logger.info("DRY RUN without any DB writse.")

    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return

    model = load_fasttext_model(model_path)

    # Open a connection (autocommit false; will commit at end)
    with eebo_db.get_connection(application_name="spelling_map_builder") as conn:

        tokens = get_corpus_tokens(conn)
        logger.info(f"Fetched {len(tokens)} distinct tokens from DB")

        # Dry run: just print a summary
        if args.dry:
            logger.info("Dry run mode: no DB writes")
            example = tokens[:10]
            for t in example:
                neighbours = model.get_nearest_neighbors(t, k=args.top_k)
                canonical = determine_canonical(neighbours)
                print(f"{t} -> {canonical} (top {args.top_k} neighbours)")
            return

        # Process in batches
        for i in tqdm(range(0, len(tokens), args.batch_size), desc="Building spelling_map"):
            batch = tokens[i:i + args.batch_size]
            for token in batch:
                neighbours = model.get_nearest_neighbors(token, k=args.top_k)
                canonical = determine_canonical(neighbours)
                insert_spelling_map(conn, token, canonical, dry=args.dry)
                update_tokens_canonical(conn, token, canonical, dry=args.dry)

        conn.commit()
        logger.info("All spelling_map entries inserted and tokens.canonical updated")

if __name__ == "__main__":
    main()
