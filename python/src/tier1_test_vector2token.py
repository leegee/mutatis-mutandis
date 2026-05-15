#!/usr/bin/env python
"""
tier1_test_vector2token.py
"""
import zarr
import numpy as np

from lib.eebo_db import get_connection
from lib.eebo_config import ZARR_ROOT

SLICE = (1641, 1641)
slice_id = f"{SLICE[0]}-{SLICE[1]}"

conn = get_connection()

with conn.cursor() as cur:
    cur.execute("""
        SELECT vector_id
        FROM pamphlet_tokens t
        JOIN pamphlet_corpus d ON d.doc_id = t.doc_id
        WHERE d.pub_year BETWEEN %s AND %s
        ORDER BY vector_id
    """, SLICE)

    db_ids = np.array([r[0] for r in cur], dtype=np.int64)

root = zarr.open(ZARR_ROOT / "tier1" / slice_id, mode="r")
zarr_ids = root["ids"][:]

db_set = set(db_ids.tolist())
zarr_set = set(zarr_ids.tolist())

missing = sorted(db_set - zarr_set)
extra = sorted(zarr_set - db_set)

print("DB IDs:", len(db_set))
print("Zarr IDs:", len(zarr_set))
print("Missing in Zarr:", len(missing))
print("Extra in Zarr:", len(extra))

if missing:
    print("Sample missing:", missing[:10])

if extra:
    print("Sample extra:", extra[:10])
