#!/usr/bin/env python
"""
generate_training_files.py

Regenerate slice-specific plain text corpora from the EEBO tokens table.
Outputs one file per slice into SLICES_DIR, ready for fastText training.
"""

from pathlib import Path
from lib.eebo_config import SLICES, SLICES_DIR
from lib.eebo_db import get_connection

SLICES_DIR.mkdir(parents=True, exist_ok=True)

def write_slice(conn, start: int, end: int) -> Path:
    """
    Dump all tokens for the slice (start, end) into a plain text file.
    Tokens are ordered by doc_id, token_idx.
    Returns path to the slice file.
    """
    slice_file = SLICES_DIR / f"{start}-{end}.txt"
    print(f"Generating slice {start}-{end} → {slice_file}")

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT token
            FROM pamphlet_tokens
            WHERE slice_start = %s AND slice_end = %s
            ORDER BY doc_id, token_idx;
            """,
            (start, end)
        )

        # Write tokens to file, space-separated
        with open(slice_file, "w", encoding="utf-8") as f:
            batch_size = 10_000
            batch = cur.fetchmany(batch_size)
            while batch:
                tokens = [row[0] for row in batch]
                f.write(" ".join(tokens) + " ")
                batch = cur.fetchmany(batch_size)

    return slice_file


def main():
    with get_connection() as conn:
        for start, end in SLICES:
            write_slice(conn, start, end)

    print(f"All slices written to {SLICES_DIR}")


if __name__ == "__main__":
    main()
