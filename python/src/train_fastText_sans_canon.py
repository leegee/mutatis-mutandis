#!/usr/bin/env python
# train_fastText.py
"""

Trains one single, large fastText model on the entire unsliced corpus.

Use the output model only to learn orthographic neighbourhoods from which
can be derived a canonical `spelling_map` to use as a normalisation layer
when creatingtime-sliced models on canonicalised text, enabling drift analysis
of semantic change rather than not orthological noise.

"""

import fasttext
from tqdm.contrib.concurrent import process_map

import lib.eebo_db as eebo_db
import lib.eebo_config as config
from lib.eebo_logging import logger


def fetch_tokens_for_slice(conn, start_year: int, end_year: int):
    """
    Generator yielding batches of tokens for a given slice.
    Uses a read-only server-side cursor.
    """
    cursor_name = f"slice_{start_year}_{end_year}"

    with conn.cursor(name=cursor_name) as cur:
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


def dump_tokens_to_disk(start_year: int, end_year: int, tmp_path):
    """
    Fetch tokens for a slice and write them to disk.
    DB connection is opened and closed entirely within this function.

    - Read-only mode
    - Per-statement timeout to prevent runaway queries
    """
    total_tokens = 0

    with eebo_db.get_connection(
        connect_timeout=10,
        application_name="fasttext_train",
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("SET default_transaction_read_only = ON;")
            cur.execute("SET statement_timeout = 10000;")  # 10,000 ms = 10 s

        with tmp_path.open("w", encoding="utf-8") as out_f:
            for batch_tokens in fetch_tokens_for_slice(conn, start_year, end_year):
                out_f.write(" ".join(batch_tokens) + "\n")
                total_tokens += len(batch_tokens)

    return total_tokens


def train_slice_model(slice_tuple):
    start_year, end_year = slice_tuple
    slice_name = f"{start_year}-{end_year}"
    logger.info(f"Starting training for slice {slice_name}")

    tmp_path = config.OUT_DIR / f"{slice_name}_tokens.txt"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Fetch tokens (DB only here)
    total_tokens = dump_tokens_to_disk(start_year, end_year, tmp_path)

    logger.info(f"Collected {total_tokens} tokens for slice {slice_name}")

    if total_tokens == 0:
        logger.warning(f"No tokens found for slice {slice_name}, skipping model")
        tmp_path.unlink(missing_ok=True)
        return

    # 2. Train fastText (NO DB CONNECTION OPEN)
    model_path = config.MODELS_DIR / f"{slice_name}.bin"
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training fastText skipgram model for slice {slice_name} @ {model_path}")
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
    max_workers = min(config.NUM_WORKERS, len(slices))

    logger.info(f"Training {len(slices)} slices with {max_workers} parallel workers")

    process_map(
        train_slice_model,
        slices,
        max_workers=max_workers,
        chunksize=1,
        desc="Slices trained",
    )

    logger.info("All slice models trained.")


if __name__ == "__main__":
    main()
