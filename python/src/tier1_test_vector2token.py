#!/usr/bin/env python
"""
tier1_test_vector2token.py

Checks structural equivalence between DB token stream and Zarr output.

Focus:
- completeness
- per-document ordering
- uniqueness
- boundary consistency
"""

import argparse
import zarr
import numpy as np
from collections import Counter, defaultdict

from lib.eebo_db import get_connection
from lib.eebo_config import ZARR_ROOT, SLICES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--first-slice-only",
        action="store_true",
        help="Only test the first slice (useful for debugging)",
    )
    return p.parse_args()


args = parse_args()
slices = SLICES[:1] if args.first_slice_only else SLICES


# Main test loop
for SLICE in slices:

    slice_id = f"{SLICE[0]}-{SLICE[1]}"
    print(f"\n=== Testing slice {slice_id} ===")

    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT t.doc_id, t.vector_id
            FROM pamphlet_tokens t
            JOIN pamphlet_corpus d ON d.doc_id = t.doc_id
            WHERE d.pub_year BETWEEN %s AND %s
            ORDER BY t.doc_id, t.token_idx
        """, SLICE)

        rows = list(cur)
    conn.close()

    db_by_doc = defaultdict(list)
    db_ids = []

    for doc_id, vid in rows:
        db_by_doc[doc_id].append(vid)
        db_ids.append(vid)

    db_ids = np.array(db_ids, dtype=np.int64)

    # Load Zarr stream
    root = zarr.open(ZARR_ROOT / "tier1" / slice_id, mode="r")
    zarr_ids = np.array(root["ids"][:], dtype=np.int64)

    # 1. Set equivalence (robust invariant)
    db_set = set(db_ids.tolist())
    zarr_set = set(zarr_ids.tolist())

    missing = sorted(db_set - zarr_set)
    extra = sorted(zarr_set - db_set)

    print("DB IDs:", len(db_set))
    print("Zarr IDs:", len(zarr_set))
    print("Missing in Zarr:", len(missing))
    print("Extra in Zarr:", len(extra))

    assert len(missing) == 0, f"Missing IDs detected: {missing[:10]}"
    assert len(extra) == 0, f"Extra IDs detected: {extra[:10]}"

    # 2. Uniqueness
    db_dupes = [k for k, v in Counter(db_ids.tolist()).items() if v != 1]
    zarr_dupes = [k for k, v in Counter(zarr_ids.tolist()).items() if v != 1]

    print("DB duplicate ids:", len(db_dupes))
    print("Zarr duplicate ids:", len(zarr_dupes))

    assert not db_dupes, f"DB duplicates: {db_dupes[:10]}"
    assert not zarr_dupes, f"Zarr duplicates: {zarr_dupes[:10]}"

    # 3. Global monotonic sanity (weaker check only)
    assert np.all(np.diff(zarr_ids) != 0), "Zarr has repeated ordering artefacts"

    # NOTE: we DO NOT enforce global strict ordering anymore
    # because pipeline is document-structured, not globally linear-preserving.

    # 4. Per-document correctness (core invariant)
    start = 0
    for doc_id, db_seq in db_by_doc.items():
        n = len(db_seq)

        zarr_seq = zarr_ids[start:start + n]
        start += n

        # length match
        assert len(db_seq) == len(zarr_seq), (
            f"Length mismatch in doc {doc_id}"
        )

        # identity per doc
        assert np.array_equal(
            np.array(db_seq, dtype=np.int64),
            zarr_seq
        ), f"Doc mismatch: {doc_id}"

        # ordering sanity (within doc only)
        assert np.all(np.diff(zarr_seq) > 0), (
            f"Non-monotonic vector_ids within doc {doc_id}"
        )

    print("✔ Per-document structural equivalence confirmed")

    # 5. Boundary sanity (optional weak global check)
    assert db_ids.min() == zarr_ids.min(), "Min id mismatch"
    assert db_ids.max() == zarr_ids.max(), "Max id mismatch"
