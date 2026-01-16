#!/usr/bin/env python
# train_fastText.py
"""
Train one fastText skipgram model per slice using tokens from the EEBO Postgres database.

- Each slice defined in eebo_config.SLICES
- Tokens are read from the tokens table for documents in the slice
- Saves one .bin model per slice in MODELS_DIR
"""

import fasttext
import eebo_config as config
from eebo_logging import logger
import eebo_db
from tqdm import tqdm


def fetch_tokens_for_slice(conn, start_year: int, end_year: int):
    """
    Generator yielding batches of tokens for a given slice (ordered by doc_id, token_idx)
    """
    offset = 0
    while True:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT t.token
                FROM tokens t
                JOIN documents d ON t.doc_id = d.doc_id
                WHERE d.pub_year BETWEEN %s AND %s
                ORDER BY t.doc_id, t.token_idx
                LIMIT %s OFFSET %s
                """,
                (start_year, end_year, config.FASTTEXT_BATCH_SIZE, offset),
            )
            batch = cur.fetchall()
            if not batch:
                break
            # Flatten list of tuples to list of strings
            yield [row[0] for row in batch]
            offset += len(batch)


def train_slice_model(conn, start_year: int, end_year: int):
    slice_name = f"{start_year}-{end_year}"
    logger.info(f"Training fastText model for slice {slice_name}")

    # Create temporary text file
    tmp_path = config.OUT_DIR / f"{slice_name}_tokens.txt"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    total_tokens = 0

    for batch_tokens in tqdm(
        fetch_tokens_for_slice(conn, start_year, end_year),
        desc=f"Slice {slice_name}",
        unit="batch"
    ):
        with tmp_path.open("a", encoding="utf-8") as out_f:  # append mode
            out_f.write(" ".join(batch_tokens) + "\n")
        total_tokens += len(batch_tokens)

    logger.info(f"Collected {total_tokens} tokens for slice {slice_name}")

    if total_tokens == 0:
        logger.warning(f"No tokens found for slice {slice_name}, skipping model")
        return

    # Train fastText skipgram model
    model_path = config.MODELS_DIR / f"{slice_name}.bin"
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Training fastText skipgram model...")
    model = fasttext.train_unsupervised(
        str(tmp_path),
        model=config.FASTTEXT_PARAMS["model"],
        dim=config.FASTTEXT_PARAMS["dim"],
        ws=config.FASTTEXT_PARAMS["ws"],
        epoch=config.FASTTEXT_PARAMS["epoch"],
        minCount=config.FASTTEXT_PARAMS["minCount"],
        minn=config.FASTTEXT_PARAMS["minn"],
        maxn=config.FASTTEXT_PARAMS["maxn"],
        thread=config.FASTTEXT_PARAMS["thread"]
    )

    model.save_model(str(model_path))
    logger.info(f"Model saved: {model_path}")
    tmp_path.unlink()

def main():
    with eebo_db.get_connection() as conn:
        for start, end in config.SLICES:
            train_slice_model(conn, start, end)

    logger.info("All slice models trained.")


if __name__ == "__main__":
    main()
