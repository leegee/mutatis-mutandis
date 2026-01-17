#!/usr/bin/env python
"""
EEBO Sentence Embedding Pipeline — CPU-friendly, progressive.

- Reads tokens per document from Postgres
- Builds sentences (simple token window fallback if needed)
- Computes MacBERTh embeddings
- Streams results into `sentences` table
"""

from __future__ import annotations
import argparse
import io
from typing import List, Optional
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from typing import cast

import lib.eebo_config as config
import lib.eebo_db as eebo_db
from lib.eebo_logging import logger

DEVICE = "cpu"   # Force CPU on this crappy old machine

# Load MacBERTh
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH, local_files_only=True)
model = AutoModel.from_pretrained(config.MODEL_PATH, local_files_only=True).to(DEVICE)
model.eval()


def get_document_tokens(doc_id: str) -> List[str]:
    """
    Fetch tokens for a single document from the DB.
    Returns list of token strings ordered by token_idx.
    """
    with eebo_db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT token FROM tokens WHERE doc_id = %s ORDER BY token_idx ASC", (doc_id,)
            )
            return [row[0] for row in cur.fetchall()]


def split_into_sentences(tokens: List[str], window: int = 10) -> List[str]:
    """
    Simple sentence splitter based on punctuation + fallback window.
    """
    sentences: List[str] = []
    current: List[str] = []
    for tok in tokens:
        current.append(tok)
        if any(p in tok for p in (".", "!", "?")):
            sentences.append(" ".join(current))
            current = []
    # Flush leftover tokens in sliding windows
    if current:
        for i in range(0, len(current), window):
            sentences.append(" ".join(current[i:i + window]))
    return sentences


def stream_sentences(doc_id: str, sentences: list[str], embeddings: list[list[float]]):
    """
    Stream sentences and embeddings into the `sentences` table using COPY (psycopg3),
    unless the work has already been done.
    """
    if not sentences:
        return

    buf = io.StringIO()
    with eebo_db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT sentence_id FROM sentences WHERE doc_id = %s", (doc_id,))
            existing_ids = {row[0] for row in cur.fetchall()}

        for idx, (sent_text, emb) in enumerate(zip(sentences, embeddings, strict=True)):
            if idx in existing_ids:
                continue
            emb_str = "{" + ",".join(f"{x:.6f}" for x in emb) + "}"
            sent_safe = sent_text.replace("\t", " ").replace("\n", " ")
            buf.write(f"{doc_id}\t{idx}\t{sent_safe}\t{emb_str}\n")

        if buf.tell() > 0:
            buf.seek(0)
            sql = "COPY sentences (doc_id, sentence_id, sentence_text_norm, embedding) " \
                  "FROM STDIN WITH (FORMAT text, DELIMITER E'\t', NULL '\\N')"
            with conn.cursor() as cur, cur.copy(sql) as copy:
                copy.write(buf.read())


def embed_batch(batch: list[str]) -> list[list[float]]:
    """Embed a single batch of sentences."""
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=False).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)
        return cast(list[list[float]], emb.cpu().tolist())


def embed_sentences_threaded(sentences: list[str]) -> list[list[float]]:
    """Compute embeddings in batches using threads for CPU speedup."""
    embeddings: list[list[float]] = []
    # Split sentences into batches
    batches = [sentences[i:i + config.EMBED_BATCH_SIZE] for i in range(0, len(sentences), config.EMBED_BATCH_SIZE)]

    with ThreadPoolExecutor(max_workers=config.NUM_WORKERS) as executor:
        future_to_batch = {executor.submit(embed_batch, batch): batch for batch in batches}
        for future in tqdm(as_completed(future_to_batch), total=len(future_to_batch), desc="Batches"):
            embeddings.extend(future.result())
    return embeddings


def main(limit: Optional[int] = None):
    logger.info("Fetching document list from DB")
    with eebo_db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT doc_id FROM documents ORDER BY doc_id")
            doc_ids: List[str] = [row[0] for row in cur.fetchall()]
    if limit:
        doc_ids = doc_ids[:limit]

    logger.info(f"Processing {len(doc_ids)} documents for sentence embeddings")
    for doc_id in tqdm(doc_ids, desc="Docs"):
        tokens = get_document_tokens(doc_id)
        sentences = split_into_sentences(tokens, window=config.INGEST_TOKEN_WINDOW_FALLBACK)
        embeddings = embed_sentences_threaded(sentences)
        stream_sentences(doc_id, sentences, embeddings)

    logger.info("Sentence embedding pipeline complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max documents to process")
    args = parser.parse_args()
    main(limit=args.limit)
