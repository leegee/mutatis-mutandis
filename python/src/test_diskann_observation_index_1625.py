from pathlib import Path
import time

import diskannpy
import numpy as np


INDEX_DIRECTORY = Path("out/test_diskann_diagnostic/diagnostic")
DIMENSIONS = 8
INDEX_PREFIX = "diagnostic"


def timed(label: str, fn):
    print()
    print("=" * 60)
    print(label)
    print("=" * 60)
    print("starting...", flush=True)

    start = time.perf_counter()

    try:
        result = fn()
    except Exception as exc:
        elapsed = time.perf_counter() - start
        print(f"FAILED after {elapsed:.3f}s", flush=True)
        raise exc

    elapsed = time.perf_counter() - start
    print(f"completed in {elapsed:.3f}s", flush=True)

    return result


def load_index(
    *,
    num_threads: int,
    num_nodes_to_cache: int,
):
    return diskannpy.StaticDiskIndex(
        index_directory=str(INDEX_DIRECTORY),
        num_threads=num_threads,
        num_nodes_to_cache=num_nodes_to_cache,
        cache_mechanism=0,
        distance_metric="l2",
        vector_dtype=np.float32,
        dimensions=DIMENSIONS,
        index_prefix=INDEX_PREFIX,
    )


def main() -> None:
    print(f"index directory: {INDEX_DIRECTORY}")
    print(f"exists: {INDEX_DIRECTORY.exists()}")
    print(f"dimensions: {DIMENSIONS}")
    print(f"index prefix: {INDEX_PREFIX}")

    if not INDEX_DIRECTORY.exists():
        raise RuntimeError(
            f"Index directory does not exist: {INDEX_DIRECTORY}"
        )

    print()
    print("Files:")
    for path in sorted(INDEX_DIRECTORY.iterdir()):
        print(f"  {path.name:40} {path.stat().st_size:,} bytes")

    # This deliberately tests only construction of the native DiskANN
    # StaticDiskIndex. No wrapper, mapping, query, or search is involved.
    for num_threads in [1]:
        for num_nodes_to_cache in [0]:
            index = timed(
                (
                    "STATIC INDEX LOAD "
                    f"(threads={num_threads}, "
                    f"nodes_to_cache={num_nodes_to_cache})"
                ),
                lambda: load_index(
                    num_threads=num_threads,
                    num_nodes_to_cache=num_nodes_to_cache,
                ),
            )

            print("StaticDiskIndex constructed successfully.", flush=True)
            print(f"type: {type(index)}", flush=True)


if __name__ == "__main__":
    main()