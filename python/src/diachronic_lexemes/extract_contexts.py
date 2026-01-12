#!/usr/bin/env python
"""
diachronic_lexemes/extract_contexts.py

Extracts sentence-bounded contexts for target words from EEBO.
Uses PostgreSQL token and document tables for fast, reproducible access.
"""

from pathlib import Path
from typing import List, Tuple, Dict, TypedDict

import psycopg

import eebo_config as config
import eebo_db


# ---------- typing ----------

Slice = Tuple[int, int]


class ContextRecord(TypedDict):
    doc_id: str
    token_idx: int
    context: str


# ---------- sentence helpers ----------

def get_sentences(tokens: List[str]) -> List[Tuple[int, int]]:
    """
    Identify sentence boundaries in a list of tokens.
    Returns a list of (start_idx, end_idx) for each sentence.
    """
    sentence_end_punct = {".", "?", "!"}
    sentences: List[Tuple[int, int]] = []
    start = 0

    for i, tok in enumerate(tokens):
        if tok in sentence_end_punct:
            end = i + 1  # include punctuation
            sentences.append((start, end))
            start = i + 1

    if start < len(tokens):
        sentences.append((start, len(tokens)))

    return sentences


def find_sentence_containing(
    idx: int,
    sentences: List[Tuple[int, int]],
) -> Tuple[int, int]:
    """
    Return the sentence span (start_idx, end_idx) containing token at idx.
    """
    for start, end in sentences:
        if start <= idx < end:
            return (start, end)

    # fallback ±5 tokens
    return (max(0, idx - 5), idx + 5)


# ---------- main extraction ----------

def extract_contexts(
    target_word: str,
    slices: List[Slice],
    max_tokens: int = 128,
) -> Dict[Slice, List[ContextRecord]]:
    """
    Extract sentence-bounded contexts for a target word, grouped by slice.
    """
    conn: psycopg.Connection = eebo_db.dbh

    contexts_by_slice: Dict[Slice, List[ContextRecord]] = {
        sl: [] for sl in slices
    }

    with conn.cursor() as cur:

        for slice_start, slice_end in slices:

            # Fetch document IDs for this slice
            cur.execute(
                """
                SELECT doc_id
                FROM documents
                WHERE pub_year BETWEEN %s AND %s
                """,
                (slice_start, slice_end),
            )
            doc_ids = [row[0] for row in cur.fetchall()]

            if not doc_ids:
                continue

            for doc_id in doc_ids:

                # Fetch ordered tokens
                cur.execute(
                    """
                    SELECT token_idx, token
                    FROM tokens
                    WHERE doc_id = %s
                    ORDER BY token_idx
                    """,
                    (doc_id,),
                )
                rows = cur.fetchall()
                if not rows:
                    continue

                tokens = [tok for _, tok in rows]

                sentences = get_sentences(tokens)

                for i, tok in enumerate(tokens):
                    if tok != target_word:
                        continue

                    start_idx, end_idx = find_sentence_containing(i, sentences)

                    if (end_idx - start_idx) > max_tokens:
                        start_idx = max(start_idx, i - max_tokens // 2)
                        end_idx = min(len(tokens), start_idx + max_tokens)

                    context_text = " ".join(tokens[start_idx:end_idx])

                    contexts_by_slice[(slice_start, slice_end)].append(
                        {
                            "doc_id": doc_id,
                            "token_idx": i,
                            "context": context_text,
                        }
                    )

    return contexts_by_slice


# ---------- CLI ----------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Extract sentence-bounded contexts for a target word."
    )
    parser.add_argument("word", type=str, help="Target word")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128,
        help="Fallback token window for long sentences",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.OUT_DIR / "contexts.json",
        help="Output JSON file",
    )

    args = parser.parse_args()

    contexts = extract_contexts(
        args.word,
        config.SLICES,
        max_tokens=args.max_tokens,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(contexts, f, indent=2)

    print(f"[DONE] Extracted contexts for '{args.word}' → {args.output}")
