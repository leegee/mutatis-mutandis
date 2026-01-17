#!/usr/bin/env python
"""

create_fastText_ortho_canon.py

Train a global fastText skipgram model on EEBO DB tokens (CPU-friendly, fast).

- Dumps tokens from Postgres using COPY (one line per document, tokens space-separated)
- Trains fastText skipgram with subword ngrams
- Saves model to MODELS_DIR for canonicalisation / semantic drift analysis
"""

from pathlib import Path
import fasttext
from lib import eebo_db, eebo_config as config
from lib.eebo_logging import logger


def dump_tokens_to_file(tmp_file: Path):
    """
    Create a fastText-ready corpus file (one line per document, space-separated tokens)
    using PostgreSQL COPY TO STDOUT (psycopg3-compatible).
    """
    logger.info(f"Dumping tokens to corpus file: {tmp_file}")
    with eebo_db.get_connection() as conn:
        with open(tmp_file, "w", encoding="utf-8") as f:
            with conn.cursor() as cur:
                # cur.copy(...) returns a context manager; enter it
                with cur.copy(
                    "COPY (SELECT string_agg(token, ' ') FROM tokens GROUP BY doc_id ORDER BY doc_id) TO STDOUT WITH (FORMAT text)"
                ) as copy_obj:
                    # copy_obj is now iterable
                    for line in copy_obj:
                        f.write(line.tobytes().decode("utf-8"))
    logger.info("Token dump complete.")


def train_fasttext_model(corpus_file: Path, output_path: Path):
    """
    Train a skip-gram fastText model with subword ngrams on the given corpus file.
    """
    logger.info(f"Training fastText model on corpus: {corpus_file}")
    model = fasttext.train_unsupervised(
        input=str(corpus_file),
        model=config.FASTTEXT_PARAMS["model"],
        dim=config.FASTTEXT_PARAMS["dim"],
        ws=config.FASTTEXT_PARAMS["ws"],
        epoch=config.FASTTEXT_PARAMS["epoch"],
        minCount=config.FASTTEXT_PARAMS["minCount"],
        minn=config.FASTTEXT_PARAMS["minn"],
        maxn=config.FASTTEXT_PARAMS["maxn"],
        thread=config.FASTTEXT_PARAMS["thread"],
    )
    model.save_model(str(output_path))
    logger.info(f"fastText model saved to {output_path}")
    return model


# -----------------------------
# Main
# -----------------------------
def main():
    # Ensure MODELS_DIR exists
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    tmp_corpus = config.OUT_DIR / "fasttext_corpus.txt"
    model_path = config.MODELS_DIR / "global_eebo.bin"

    # 1. Dump corpus file
    dump_tokens_to_file(tmp_corpus)

    # 2. Train fastText model
    train_fasttext_model(tmp_corpus, model_path)

    # 3. Delete temporary corpus file
    tmp_corpus.unlink(missing_ok=True)
    logger.info("Temporary corpus file deleted. Training complete.")


if __name__ == "__main__":
    main()
