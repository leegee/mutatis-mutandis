#!/usr/bin/env python
"""
generate_token_embeddings.py

Load slice-specific fastText models, generate embeddings for all tokens in each slice,
and store them in the `token_vectors` table in the database.

Optimized for speed: table is created without primary key/index during insertion,
then indexes/PK are added afterward via `lib.eebo_db`.

CLI arg `--dedup-only`: skip slice embedding generation and just deduplicate
existing token_vectors table to remove duplicates and safely create PK/index.
"""

from pathlib import Path
import argparse
import fasttext
import numpy as np

import lib.eebo_config as config
import lib.eebo_db as eebo_db
from lib.eebo_logging import logger
from train_slice_fasttext import slice_model_path
from align import load_aligned_vectors


USE_ALIGNED_FASTTEXT_VECTORS = True


def generate_embeddings_per_model(model_path: Path) -> dict[str, np.ndarray]:
    """Load a fastText slice model and generate embeddings for all words in its vocabulary."""
    logger.info(f"Loading model {model_path}")
    model = fasttext.load_model(str(model_path))

    tokens = model.get_words()
    embeddings: dict[str, np.ndarray] = {}
    for tok in tokens:
        embeddings[str(tok)] = model.get_word_vector(tok).astype(np.float32)
    logger.info(f"Generated embeddings for {len(tokens)} tokens in {model_path.name}")
    return embeddings


def store_embeddings(conn, embeddings: dict[str, np.ndarray], slice_start: int, slice_end: int):
    """Insert token embeddings into token_vectors table using batch insert, per slice."""
    BATCH_SIZE = 5000
    items = list(embeddings.items())

    logger.info(f"Inserting {len(items)} token embeddings for slice {slice_start}-{slice_end} into DB")
    with conn.transaction():
        with conn.cursor() as cur:
            for i in range(0, len(items), BATCH_SIZE):
                batch = items[i:i+BATCH_SIZE]
                args = [(tok, slice_start, slice_end, vec.tolist()) for tok, vec in batch]
                cur.executemany(
                    """
                    INSERT INTO token_vectors (token, slice_start, slice_end, vector)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (token, slice_start, slice_end) DO NOTHING
                    """,
                    args
                )


def main():
    parser = argparse.ArgumentParser(description="Generate token embeddings or deduplicate existing ones")
    parser.add_argument(
        "--dedup-only",
        action="store_true",
        help="Skip slice embedding generation and just deduplicate token_vectors table"
    )
    args = parser.parse_args()

    with eebo_db.get_connection() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("TRUNCATE token_vectors;")

        if not args.dedup_only:
            for start, end in config.SLICES:
                if USE_ALIGNED_FASTTEXT_VECTORS:
                    slice_id = f"{start}-{end}"
                    try:
                        embeddings = load_aligned_vectors(slice_id)
                    except FileNotFoundError:
                        logger.warning(f"Aligned vectors for slice {slice_id} missing, skipping")
                        continue
                else:
                    model_file = slice_model_path((start, end))
                    if not model_file.exists():
                        logger.warning(f"Model {model_file} missing, skipping")
                        continue
                    embeddings = generate_embeddings_per_model(model_file)

                # Insert embeddings regardless of source
                store_embeddings(conn, embeddings, start, end)
        else:
            logger.info("Skipping embedding generation due to --dedup-only flag")

    logger.info("Token embeddings table ready with indexes/PK")


if __name__ == "__main__":
    main()
