from pathlib import Path

import diskannpy
import numpy as np

from .models import Float32Array, UInt64Array


def build_diskann_index(
    vectors: Float32Array,
    event_ids: UInt64Array,
    *,
    index_directory: str | Path,
    dimensions: int,
    complexity: int = 100,
    graph_degree: int = 64,
    search_memory_gb: float = 4.0,
    build_memory_gb: float = 8.0,
    num_threads: int = 0,
    pq_disk_bytes: int = 0,
    index_prefix: str = "local",
) -> Path:
    """
    Build a DiskANN index and persist its observation-ID mapping.

    Input embeddings are normalised to unit length before indexing.
    DiskANN uses L2 distance, which is monotonic with cosine similarity
    for unit-normalised vectors.

    The Parquet observation store remains authoritative. DiskANN and its
    positional event-ID mapping are disposable derived artefacts.

    Failure mode:
        DiskANN writes an intermediate vector representation while building.
        That representation is not treated as corpus storage and can be
        regenerated from Parquet.
    """
    vector_array = np.asarray(vectors, dtype=np.float32)
    event_id_array = np.asarray(event_ids, dtype=np.uint64)

    if vector_array.ndim != 2:
        raise ValueError("vectors must be two-dimensional")

    if vector_array.shape[1] != dimensions:
        raise ValueError(
            f"vector dimension {vector_array.shape[1]} "
            f"does not match dimensions {dimensions}"
        )

    if event_id_array.ndim != 1:
        raise ValueError("event_ids must be one-dimensional")

    if len(event_id_array) != vector_array.shape[0]:
        raise ValueError(
            "event_ids and vectors must contain the same number of observations"
        )

    if len(event_id_array) == 0:
        raise ValueError("cannot build an index with no observations")

    if not np.all(np.isfinite(vector_array)):
        raise ValueError("vectors must contain only finite values")

    norms = np.linalg.norm(vector_array, axis=1)

    if np.any(norms == 0):
        raise ValueError("vectors must not contain zero vectors")

    vector_array = vector_array / norms[:, None]

    if len(np.unique(event_id_array)) != len(event_id_array):
        raise ValueError("event_ids must be unique")

    if complexity < graph_degree:
        raise ValueError(
            "complexity must be at least graph_degree"
        )

    if graph_degree <= 0:
        raise ValueError("graph_degree must be positive")

    if search_memory_gb <= 0:
        raise ValueError("search_memory_gb must be positive")

    if build_memory_gb <= 0:
        raise ValueError("build_memory_gb must be positive")

    if num_threads < 0:
        raise ValueError("num_threads must be non-negative")

    if pq_disk_bytes < 0:
        raise ValueError("pq_disk_bytes must be non-negative")

    output_directory = Path(index_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    event_ids_path = output_directory / f"{index_prefix}_event_ids.npy"
    np.save(event_ids_path, event_id_array)

    diskannpy.build_disk_index(
        data=vector_array,
        distance_metric="l2",
        index_directory=str(output_directory),
        complexity=complexity,
        graph_degree=graph_degree,
        search_memory_maximum=search_memory_gb,
        build_memory_maximum=build_memory_gb,
        num_threads=num_threads,
        pq_disk_bytes=pq_disk_bytes,
        index_prefix=index_prefix,
    )

    return event_ids_path
