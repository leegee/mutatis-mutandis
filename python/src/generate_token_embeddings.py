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

EMBEDDING_DIM = config.FASTTEXT_PARAMS["dim"]
TOKEN_VECTORS_TABLE = "token_vectors"


def init_token_vectors_table(conn):
    """Create token_vectors table WITHOUT primary key/index for fast bulk insert."""
    with conn.transaction():
        with conn.cursor() as cur:
            logger.info(f"Dropping existing table {TOKEN_VECTORS_TABLE} if exists")
            cur.execute(f"DROP TABLE IF EXISTS {TOKEN_VECTORS_TABLE} CASCADE;")
            logger.info(f"Creating table {TOKEN_VECTORS_TABLE} (no PK/index yet)")
            cur.execute(f"""
                CREATE TABLE {TOKEN_VECTORS_TABLE} (
                    token TEXT,
                    vector FLOAT4[] NOT NULL
                );
            """)
    logger.info(f"Table {TOKEN_VECTORS_TABLE} ready for bulk insert")


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
                    f"INSERT INTO {TOKEN_VECTORS_TABLE} (token, vector) VALUES (%s, %s);",
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
            # 1. Create a temporary table with aggregated vectors
            cur.execute(f"""
                CREATE TEMP TABLE tmp_vectors AS
                SELECT token, array_agg(vector) AS vectors
                FROM {TOKEN_VECTORS_TABLE}
                GROUP BY token;
            """)

            # 2. Replace original table with averaged vectors
            cur.execute(f"DROP TABLE {TOKEN_VECTORS_TABLE};")
            cur.execute(f"""
                CREATE TABLE {TOKEN_VECTORS_TABLE} (
                    token TEXT PRIMARY KEY,
                    vector FLOAT4[] NOT NULL
                );
            """)

            # 3. Insert averaged embeddings
            cur.execute(f"""
                INSERT INTO {TOKEN_VECTORS_TABLE} (token, vector)
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
            # Create table WITHOUT PK/index for fast insertion
            init_token_vectors_table(conn)

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

        # Deduplicate tokens to ensure PK/index can be created safely
        deduplicate_token_vectors(conn)

        # Create primary key/index
        eebo_db.create_indexes_token_vectors()

    logger.info("Token embeddings table ready with indexes/PK")


if __name__ == "__main__":
    main()
