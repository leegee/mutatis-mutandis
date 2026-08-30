"""
test_lance_index.py

<string>:1: DeprecationWarning: table_names() is deprecated, use list_tables() instead
['broad', 'local', 'medium']
[{'event_id': 657581043937310007, 'year': 1476}]
(eebo)

python -c "import lancedb; import numpy as np; db=lancedb.connect(r'../out/indexes/lance'); t=db.open_table('local'); eid=657581043937310007; rows=t.search().where(f'event_id = {eid}', prefilter=True).select(['event_id','year','vector']).limit(1).to_list(); r=rows[0]; v=np.asarray(r['vector'], dtype=np.float32); print(r['event_id'], r['year'], v.shape, v.dtype, np.linalg.norm(v))"

"""

from __future__ import annotations

import argparse
from pathlib import Path

import lancedb
import numpy as np

import lib.corpus_config as config
from retrieval.parquet_embeddings import load_embeddings
from retrieval.lance_observation_index import LanceObservationIndex


SCALES = ("local", "medium", "broad")
DIMENSIONS = 768


def find_event_vector(
    store: Path,
    event_id: int,
    scale: str,
) -> tuple[int, np.ndarray]:
    """
    Locate one event in Tier 1 and return its year and canonical vector.

    Failure mode:
        The event must occur exactly once in the requested scale. A missing
        or duplicated event indicates a Tier 1 integrity problem rather than
        a Lance retrieval problem.
    """
    for year in range(1476, 1744):
        event_ids, vectors = load_embeddings(
            store,
            year_start=year,
            year_end=year,
            scale=scale,
            dimensions=DIMENSIONS,
        )

        matches = np.flatnonzero(event_ids == event_id)

        if len(matches) == 1:
            return year, np.asarray(
                vectors[matches[0]],
                dtype=np.float32,
            )

        if len(matches) > 1:
            raise RuntimeError(
                f"event_id={event_id} occurs more than once "
                f"in year={year}, scale={scale}"
            )

    raise KeyError(
        f"event_id={event_id} not found in Tier 1 scale={scale}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--event-id",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--scale",
        choices=SCALES,
        default="local",
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=config.EVENTSTORE_T1_PATH,
    )
    parser.add_argument(
        "--lance",
        type=Path,
        default=Path(config.LANCE_INDEXES_DIR),
    )
    args = parser.parse_args()

    print(
        f"Testing event_id={args.event_id} "
        f"scale={args.scale}"
    )

    year, canonical = find_event_vector(
        args.store,
        args.event_id,
        args.scale,
    )

    print(f"Tier 1 year: {year}")
    print(f"Tier 1 norm: {np.linalg.norm(canonical):.8f}")

    db = lancedb.connect(str(args.lance))
    table = db.open_table(args.scale)

    index = LanceObservationIndex(
        table,
        dimensions=DIMENSIONS,
        year_start=year,
        year_end=year,
    )

    reconstructed = index.reconstruct(args.event_id)

    difference = np.max(
        np.abs(canonical - reconstructed)
    )

    cosine = float(
        np.dot(canonical, reconstructed)
        / (
            np.linalg.norm(canonical)
            * np.linalg.norm(reconstructed)
        )
    )

    print(f"Lance norm: {np.linalg.norm(reconstructed):.8f}")
    print(f"Maximum absolute difference: {difference:.8g}")
    print(f"Cosine similarity: {cosine:.8f}")

    if not np.allclose(
        canonical,
        reconstructed,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise AssertionError(
            "Lance reconstruction does not match Tier 1"
        )

    print("PASS: reconstruction matches Tier 1")

    result = index.search(
        reconstructed,
        k=10,
    )

    print("\nNearest neighbours:")
    for rank, (found_id, score) in enumerate(
        zip(result.event_ids, result.distances),
        start=1,
    ):
        marker = " <-- TARGET" if int(found_id) == args.event_id else ""
        print(
            f"{rank:2d}  event_id={int(found_id)} "
            f"cosine={float(score):.6f}{marker}"
        )

    if args.event_id not in result.event_ids:
        raise AssertionError(
            "Target event was not returned by Lance search"
        )

    target_rank = int(
        np.flatnonzero(
            result.event_ids == args.event_id
        )[0]
    ) + 1

    print(
        f"\nPASS: target returned at rank {target_rank}"
    )

    ids = [
        int(event_id)
        for event_id in result.event_ids[:5]
    ]

    vectors = index.reconstruct_many(ids)

    if vectors.shape != (len(ids), DIMENSIONS):
        raise AssertionError(
            f"Unexpected reconstruct_many shape: {vectors.shape}"
        )

    for requested_id, vector in zip(ids, vectors):
        if not np.isfinite(vector).all():
            raise AssertionError(
                f"Non-finite vector for event_id={requested_id}"
            )

    print(
        f"PASS: reconstruct_many returned "
        f"{len(vectors)} vectors in requested order"
    )

    print("\nLance smoke test passed.")


if __name__ == "__main__":
    main()
