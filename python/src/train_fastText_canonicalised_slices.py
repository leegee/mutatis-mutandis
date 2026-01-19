#!/usr/bin/env python
# train_fastText_canonicalised_slices.py
"""
Train fastText models on canonicalised slices of EEBO tokens.

IMPORTANT:
- We train ONLY on canonical keyword targets
- Canonical targets are defined exclusively by
  lib.eebo_config.KEYWORDS_TO_NORMALISE (keys)
- No fallback to surface forms
- No NULL canonicals allowed through
"""

from pathlib import Path
from tqdm.contrib.concurrent import process_map
from typing import Tuple, Iterator, List
import argparse
import fasttext
import shutil

import lib.eebo_config as config
import lib.eebo_db as db


# -----------------------------
# Ensure directories exist
# -----------------------------
TMP_DIR = Path(config.TMP_DIR)
TMP_DIR.mkdir(exist_ok=True, parents=True)
config.MODELS_DIR.mkdir(exist_ok=True)


# -----------------------------
# Argument parsing
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--clean",
    action="store_true",
    help="Delete temporary slices after training"
)
args = parser.parse_args()


# -----------------------------
# Canonical vocabulary
# -----------------------------
# Single source of truth: canonical keyword heads
CANONICAL_TARGETS = sorted(config.KEYWORDS_TO_NORMALISE.keys())


# -----------------------------
# Database access
# -----------------------------
def fetch_tokens_for_slice(
    conn: db.Connection,
    start_year: int,
    end_year: int,
    batch_size: int = 100_000,
) -> Iterator[str]:
    """
    Fetch ONLY canonical keyword tokens for a given year slice.

    Guarantees:
    - token is a string
    - token ∈ KEYWORDS_TO_NORMALISE.keys()
    - no NULL canonicals
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT t.canonical
            FROM tokens t
            JOIN documents d ON t.doc_id = d.doc_id
            WHERE d.pub_year >= %s
              AND d.pub_year <= %s
              AND t.canonical IS NOT NULL
              AND t.canonical = ANY(%s)
            """,
            (
                start_year,
                end_year,
                list(config.KEYWORDS_TO_NORMALISE.keys()),
            ),
        )

        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            for (token,) in rows:
                yield token


# -----------------------------
# Slice dumping
# -----------------------------
def dump_slice_to_disk(slice_range: Tuple[int, int]) -> Path:
    """
    Fetch canonical tokens for a slice and write them to disk.
    Returns the path of the slice file.
    """
    start_year, end_year = slice_range
    slice_file = TMP_DIR / f"{start_year}-{end_year}.txt"

    tokens: List[str] = []

    with db.get_connection() as conn:
        for token in fetch_tokens_for_slice(conn, start_year, end_year):
            tokens.append(token)

    # Write one token per line (fastText-compatible)
    with slice_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(tokens))

    return slice_file


# -----------------------------
# fastText training
# -----------------------------
def train_slice_model(slice_file: Path) -> Path:
    """
    Train a fastText skip-gram model on a canonical slice.
    Returns the path to the saved .bin model file.
    """
    training_dir = Path(config.MODELS_DIR) / "fastTextCanonSlice"
    training_dir.mkdir(parents=True, exist_ok=True)

    out_file = training_dir / slice_file.with_suffix(".bin").name

    model = fasttext.train_unsupervised(
        input=str(slice_file),
        model="skipgram",
        lr=0.05,
        epoch=5,
        dim=100,
        thread=4,
        minn=2,
        maxn=5,
        ws=5,
        verbose=2,
    )

    model.save_model(str(out_file))
    return out_file


# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    # Example slices (replace with config.SLICES when ready)
    slice_ranges = [(1640 + i, 1640 + i) for i in range(5)]

    print("Dumping canonical token slices to disk...")
    slice_files = process_map(
        dump_slice_to_disk,
        slice_ranges,
        max_workers=config.NUM_WORKERS,
    )

    print("Training fastText models on slices...")
    models = process_map(
        train_slice_model,
        slice_files,
        max_workers=config.NUM_WORKERS,
    )

    if args.clean:
        print("Cleaning up slice files...")
        shutil.rmtree(TMP_DIR)

    print("All canonical slices trained successfully.")
