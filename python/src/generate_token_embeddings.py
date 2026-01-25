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


def generate_embeddings_for_model(model_path: Path):
    """Load a fastText slice model and generate embeddings for all words in its vocabulary."""
    logger.info(f"Loading model {model_path}")
    model = fasttext.load_model(str(model_path))

    tokens = model.get_words()
    embeddings = {}
    for tok in tokens:
        vec = model.get_word_vector(tok)
        embeddings[tok] = vec.astype(np.float32)
    logger.info(f"Generated embeddings for {len(tokens)} tokens in {model_path.name}")
    return embeddings


def store_embeddings(conn, embeddings: dict[str, np.ndarray]):
    """Insert token embeddings into token_vectors table using batch insert."""
    BATCH_SIZE = 5000
    items = list(embeddings.items())

    logger.info(f"Inserting {len(items)} token embeddings into DB")
    with conn.transaction():
        with conn.cursor() as cur:
            for i in range(0, len(items), BATCH_SIZE):
                batch = items[i:i+BATCH_SIZE]
                args = [(tok, vec.tolist()) for tok, vec in batch]
                cur.executemany(
                    "INSERT INTO token_vectors (token, vector) VALUES (%s, %s);",
                    args
                )


def deduplicate_token_vectors(conn):
    """
    Deduplicate token_vectors table by averaging embeddings for duplicate tokens.
    Replaces old table with deduplicated version ready for PK/index creation.
    """
    logger.info("Starting deduplication of token_vectors")
    with conn.transaction():
        with conn.cursor() as cur:
            # Create a temporary table with aggregated vectors
            cur.execute("""
                CREATE TEMP TABLE tmp_vectors AS
                    SELECT token, array_agg(vector) AS vectors
                    FROM token_vectors
                    GROUP BY token;
            """)

            # Replace original table with averaged vectors
            cur.execute("DROP TABLE token_vectors;")
            cur.execute("""
                CREATE TABLE token_vectors (
                    token TEXT PRIMARY KEY,
                    vector FLOAT4[] NOT NULL
                );
            """)

            # Insert averaged embeddings
            cur.execute("""
                INSERT INTO token_vectors (token, vector)
                SELECT token,
                       ARRAY(
                         SELECT avg(e)
                         FROM unnest(vectors) WITH ORDINALITY AS t(e, idx)
                         GROUP BY idx
                         ORDER BY idx
                       )::float4[]
                FROM tmp_vectors;
            """)

    logger.info("Deduplication complete: token_vectors now contains unique tokens")


def main():
    parser = argparse.ArgumentParser(description="Generate token embeddings or deduplicate existing ones")
    parser.add_argument(
        "--dedup-only",
        action="store_true",
        help="Skip slice embedding generation and just deduplicate token_vectors table"
    )
    args = parser.parse_args()

    with eebo_db.get_connection() as conn:
        if not args.dedup_only:
            # Drop the PK for speed.
            eebo_db.drop_indexes_token_vectors(conn)

            # Iterate over slices and generate embeddings
            for start, end in config.SLICES:
                model_file = config.FASTTEXT_SLICE_MODEL_DIR / f"slice_{start}_{end}.bin"
                if not model_file.exists():
                    logger.warning(f"Model {model_file} missing, skipping")
                    continue

                embeddings = generate_embeddings_for_model(model_file)
                store_embeddings(conn, embeddings)
        else:
            logger.info("Skipping embedding generation due to --dedup-only flag")
            eebo_db.drop_indexes_token_vectors(conn)

        # Deduplicate tokens to ensure PK/index can be created safely
        deduplicate_token_vectors(conn)

        # Create primary key/index
        eebo_db.create_indexes_token_vectors(conn)

    logger.info("Token embeddings table ready with indexes/PK")


if __name__ == "__main__":
    # main()
    with eebo_db.get_connection() as conn:
        eebo_db.create_indexes_token_vectors(conn)
