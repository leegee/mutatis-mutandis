# retrieval/parquet_embeddings.py

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds

from .models import Float32Array, Int64Array


def load_embeddings(
    corpus_root: str | Path,
    *,
    year: int,
    scale: str,
    dimensions: int,
) -> tuple[Int64Array, Float32Array]:
    """
    Load one year's observation embeddings from the Parquet corpus.

    Returns
    -------
    event_ids
        Stable semantic observation ids.

    vectors
        (N, dimensions) float32 embedding matrix.

    Invariants
    ----------
    - event_ids are unique
    - vectors are finite
    - vectors and ids have identical length
    """

    embedding_column = f"emb_{scale}"

    dataset = ds.dataset(
        str(corpus_root),
        format="parquet",
        partitioning="hive",
    )

    fragments = sorted(
        dataset.get_fragments(
            filter=(ds.field("year") == year),
        ),
        key=str,
    )

    if not fragments:
        raise RuntimeError(
            f"No observations found for year={year}"
        )

    tables = [
        fragment.to_table(
            columns=[
                "event_id",
                embedding_column,
            ],
        )
        for fragment in fragments
    ]

    table = pa.concat_tables(tables)

    event_ids = np.asarray(
        table.column("event_id").to_numpy(),
        dtype=np.int64,
    )

    vectors = np.asarray(
        table.column(embedding_column).to_pylist(),
        dtype=np.float32,
    )

    if vectors.ndim != 2:
        raise ValueError(
            f"Expected 2D embedding matrix, got {vectors.shape}"
        )

    if vectors.shape[1] != dimensions:
        raise ValueError(
            f"Expected {dimensions} dimensions, got {vectors.shape[1]}"
        )

    if len(event_ids) != len(vectors):
        raise ValueError(
            "event_ids and vectors have different lengths"
        )

    if len(np.unique(event_ids)) != len(event_ids):
        raise ValueError(
            "Duplicate event_ids detected"
        )

    if not np.isfinite(vectors).all():
        raise ValueError(
            "Vectors contain non-finite values"
        )

    return event_ids, vectors
