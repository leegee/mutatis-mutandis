#!/usr/bin/env python
# train_fastText.py
"""
Train one fastText skipgram model per slice using tokens from the EEBO Postgres database.

- Each slice defined in eebo_config.SLICES
- Tokens are read from the tokens table for documents in the slice
- Saves one .bin model per slice in eebo_config.MODELS_DIR
- Parallelized across slices using process_map from tqdm.contrib.concurrent
"""

import fasttext
from tqdm.contrib.concurrent import process_map  # handles parallelism + progress bar
import eebo_db
import eebo_config as config
from eebo_logging import logger

NUM_WORKERS = 4  # adjust for CPU cores


def fetch_tokens_for_slice(conn, start_year: int, end_year: int):
    """
    Generator yielding batches of tokens for a given slice (ordered by doc_id, token_idx)
    Uses a server-side cursor to avoid loading the entire slice into memory.
    """
    with conn.cursor(name=f"slice_{start_year}_{end_year}") as cur:
        cur.itersize = config.FASTTEXT_BATCH_SIZE
        cur.execute(
            """
            SELECT t.token
            FROM tokens t
            JOIN documents d ON t.doc_id = d.doc_id
            WHERE d.pub_year BETWEEN %s AND %s
            ORDER BY t.doc_id, t.token_idx
            """,
            (start_year, end_year),
        )
        while True:
            batch = cur.fetchmany(config.FASTTEXT_BATCH_SIZE)
            if not batch:
                break
            yield [row[0] for row in batch]


def train_slice_model(slice_tuple):
    """
    Train a fastText model for a single slice.
    Receives (start_year, end_year) tuple.
    This function runs in a separate process (each has its own DB connection).
    """
    start_year, end_year = slice_tuple
    slice_name = f"{start_year}-{end_year}"
    logger.info(f"Starting training for slice {slice_name}")

    # DB connection per process
    with eebo_db.get_connection() as conn:
        # Temp file for tokens
        tmp_path = config.OUT_DIR / f"{slice_name}_tokens.txt"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)

        total_tokens = 0
        with tmp_path.open("w", encoding="utf-8") as out_f:
            for batch_tokens in fetch_tokens_for_slice(conn, start_year, end_year):
                out_f.write(" ".join(batch_tokens) + "\n")
                total_tokens += len(batch_tokens)

        logger.info(f"Collected {total_tokens} tokens for slice {slice_name}")

        if total_tokens == 0:
            logger.warning(f"No tokens found for slice {slice_name}, skipping model")
            return

        # Train fastText
        model_path = config.MODELS_DIR / f"{slice_name}.bin"
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(f"Training fastText skipgram model for slice {slice_name}...")
        model = fasttext.train_unsupervised(
            str(tmp_path),
            model=config.FASTTEXT_PARAMS["model"],
            dim=config.FASTTEXT_PARAMS["dim"],
            ws=config.FASTTEXT_PARAMS["ws"],
            epoch=config.FASTTEXT_PARAMS["epoch"],
            minCount=config.FASTTEXT_PARAMS["minCount"],
            minn=config.FASTTEXT_PARAMS["minn"],
            maxn=config.FASTTEXT_PARAMS["maxn"],
            thread=config.FASTTEXT_PARAMS["thread"],
        )

        model.save_model(str(model_path))
        logger.info(f"Model saved: {model_path}")
        tmp_path.unlink()


def main():
    slices = config.SLICES
    max_workers = min(NUM_WORKERS, len(slices))
    logger.info(f"Training {len(slices)} slices with {max_workers} parallel workers")

    # process_map automatically runs slices in parallel with a tqdm progress bar
    process_map(
        train_slice_model,
        slices,
        max_workers=max_workers,
        chunksize=1,  # one slice per task
        desc="Slices trained"
    )

    logger.info("All slice models trained.")


if __name__ == "__main__":
    main()
