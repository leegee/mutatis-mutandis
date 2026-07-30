#!/usr/bin/env python
"""
export_slices_for_colab.py
"""

import os
import json
from pathlib import Path
from collections import defaultdict

from lib.corpus_db import get_connection
from lib.eebo_config import SLICES_DIR, SLICES

OUT_DIR = Path("exported_slices")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Optional: store doc_id mapping per slice
DOCID_MAP_DIR = OUT_DIR / "docid_map"
DOCID_MAP_DIR.mkdir(exist_ok=True, parents=True)

conn = get_connection()

for start, end in SLICES:
    slice_id = f"{start}-{end}"
    print(f"Exporting slice {slice_id}...")
    cursor = conn.cursor()
    # Adjust this query to your schema; here we assume `pamphlet_tokens` table
    cursor.execute("""
        SELECT doc_id, token
        FROM pamphlet_tokens
        WHERE slice_id BETWEEN %s AND %s
        ORDER BY doc_id, token_order
    """, (start, end))

    lines = []
    token_to_docids = defaultdict(list)

    for doc_id, token in cursor.fetchall():
        lines.append(token)
        token_to_docids[token].append(doc_id)

    # Write tokens to a single .txt file per slice
    slice_file = OUT_DIR / f"{slice_id}.txt"
    with open(slice_file, "w", encoding="utf-8") as f:
        f.write(" ".join(lines))

    # Save doc_id mapping
    map_file = DOCID_MAP_DIR / f"{slice_id}_docids.json"
    with open(map_file, "w", encoding="utf-8") as f:
        json.dump(token_to_docids, f, indent=2)

    print(f"Saved slice to {slice_file}, doc_id mapping to {map_file}")

conn.close()
print("All slices exported.")
