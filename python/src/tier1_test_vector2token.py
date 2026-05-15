#!/usr/bin/env python
"""
tier1_test_vector2token.py

Checks structural equivalence between DB token stream and Zarr output.

Focus:
- completeness
- ordering
- uniqueness
- gap detection
"""

import zarr
import numpy as np
from collections import Counter

from lib.eebo_db import get_connection
from lib.eebo_config import ZARR_ROOT

SLICE = (1641, 1641)
slice_id = f"{SLICE[0]}-{SLICE[1]}"


# Load DB vector stream

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

conn.close()



# Load Zarr output


root = zarr.open(ZARR_ROOT / "tier1" / slice_id, mode="r")
zarr_ids = np.array(root["ids"][:], dtype=np.int64)



# Basic set equivalence (your original test)


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



# 1. Strict uniqueness (detect accidental duplication)


db_counts = Counter(db_ids.tolist())
zarr_counts = Counter(zarr_ids.tolist())

db_dupes = [k for k, v in db_counts.items() if v != 1]
zarr_dupes = [k for k, v in zarr_counts.items() if v != 1]

print("DB duplicate ids:", len(db_dupes))
print("Zarr duplicate ids:", len(zarr_dupes))

assert len(db_dupes) == 0, f"DB has duplicates: {db_dupes[:10]}"
assert len(zarr_dupes) == 0, f"Zarr has duplicates: {zarr_dupes[:10]}"



# 2. Ordering checks (stream integrity)


assert np.all(np.diff(db_ids) > 0), "DB ids not strictly increasing"
assert np.all(np.diff(zarr_ids) > 0), "Zarr ids not strictly increasing"



# 3. Gap detection (detect missing ranges, not just items)


db_gaps = np.where(np.diff(db_ids) > 1)[0]
zarr_gaps = np.where(np.diff(zarr_ids) > 1)[0]

print("DB gaps:", len(db_gaps))
print("Zarr gaps:", len(zarr_gaps))



# 4. Boundary sanity


assert db_ids.min() == zarr_ids.min(), "Min id mismatch"
assert db_ids.max() == zarr_ids.max(), "Max id mismatch"



# 5. Final structural equivalence


assert np.array_equal(db_ids, zarr_ids), (
    "Exact sequence mismatch between DB and Zarr"
)

print("✔ Full structural equivalence confirmed")
