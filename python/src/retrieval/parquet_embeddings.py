# retrieval/parquet_embeddings.py

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from .models import Float32Array, UInt64Array


def _normalise_embedding_column(
    table: pa.Table,
    column: str,
    dim: int,
) -> pa.Table:
    arr = table[column].combine_chunks()

    # if arr.null_count:
    #     mask = pc.is_valid(arr)
    #     table = table.filter(mask)
    #     arr = table[column].combine_chunks()

    # if len(arr) == 0:
    #     return table

    if pa.types.is_fixed_size_list(arr.type):
        if arr.type.list_size != dim:
            raise ValueError( f"{column} has fixed list size {arr.type.list_size}, expected {dim}" )

    elif pa.types.is_list(arr.type):
        lengths = pc.list_value_length(arr)
        if not pc.all(pc.equal(lengths, dim)).as_py():
            raise ValueError(
                f"{column} contains vectors with length != {dim}"
            )

        values = arr.values
        if values.type != pa.float32():
            values = pc.cast(values, pa.float32())

        arr = pa.FixedSizeListArray.from_arrays(values, dim)

    else:
        raise TypeError( f"Unsupported embedding type for {column}: {arr.type}" )

    return table.set_column(
        table.schema.get_field_index(column),
        column,
        arr,
    )


from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from .models import Float32Array, UInt64Array


def _fixed_size_list_to_numpy(
    column: pa.ChunkedArray,
    dimensions: int,
) -> Float32Array:
    """Convert Arrow fixed-size vectors without materialising Python lists."""

    arrays = []

    for chunk in column.chunks:
        if not pa.types.is_fixed_size_list(chunk.type):
            raise ValueError(
                f"Expected FixedSizeListArray, got {chunk.type}"
            )

        if chunk.type.list_size != dimensions:
            raise ValueError( f"Expected vector dimension {dimensions}, got {chunk.type.list_size}" )

        values = chunk.values.to_numpy(zero_copy_only=False)

        if values.dtype != np.float32:
            values = values.astype(np.float32, copy=False)

        arrays.append(
            values.reshape(-1, dimensions)
        )

    if not arrays:
        return np.empty(
            (0, dimensions),
            dtype=np.float32,
        )

    return np.concatenate(arrays, axis=0)


def load_embeddings(
    corpus_root: str | Path,
    *,
    year_start: int,
    year_end: int,
    scale: str,
    dimensions: int,
) -> tuple[UInt64Array, Float32Array]:
    """
    Load observations from an inclusive temporal bucket.

    The Parquet store remains the source of truth. This function materialises
    only the selected scale and year range for construction of a DiskANN
    derived index.

    Invariants:
        - event_ids are unique
        - vectors are finite
        - vectors have shape (N, dimensions)
        - event_ids and vectors have identical length

    Failure mode:
        Do not use Arrow's to_pylist() for embeddings. A large
        FixedSizeListArray becomes a Python object graph and can consume
        several times the memory of the underlying float32 vectors.
    """

    if year_start > year_end:
        raise ValueError(
            f"year_start {year_start} is after year_end {year_end}"
        )

    if dimensions <= 0:
        raise ValueError("dimensions must be positive")

    if not scale:
        raise ValueError("scale must not be empty")

    embedding_column = f"emb_{scale}"

    dataset = ds.dataset(
        str(corpus_root),
        format="parquet",
        partitioning="hive",
    )

    fragments = sorted(
        dataset.get_fragments(
            filter=(
                (ds.field("year") >= year_start)
                & (ds.field("year") <= year_end)
            )
        ),
        key=str,
    )

    if not fragments:
        raise RuntimeError(
            f"No observations found for years "
            f"{year_start}-{year_end}"
        )

    event_id_arrays = []
    vector_arrays = []

    for fragment in fragments:
        table = fragment.to_table(
            columns=[
                "event_id",
                embedding_column,
            ]
        )

        column = table.column(embedding_column)

        valid = pc.is_valid(column)

        if not pc.all(valid).as_py():
            table = table.filter(valid)

        if table.num_rows == 0:
            continue

        table = _normalise_embedding_column(
            table,
            embedding_column,
            dimensions,
        )

        event_id_arrays.append(
            np.asarray(
                table.column("event_id").to_numpy(),
                dtype=np.uint64,
            )
        )

        vector_arrays.append(
            _fixed_size_list_to_numpy(
                table.column(embedding_column),
                dimensions,
            )
        )

    if not event_id_arrays:
        return (
            np.empty(0, dtype=np.uint64),
            np.empty(
                (0, dimensions),
                dtype=np.float32,
            ),
        )

    event_ids = np.concatenate(event_id_arrays)
    vectors = np.concatenate(vector_arrays)

    if vectors.ndim != 2:
        raise ValueError(
            f"Expected 2D embedding matrix, got {vectors.shape}"
        )

    if vectors.shape[1] != dimensions:
        raise ValueError( f"Expected {dimensions} dimensions, got {vectors.shape[1]}" )

    if len(event_ids) != len(vectors):
        raise ValueError( "event_ids and vectors have different lengths" )

    if len(np.unique(event_ids)) != len(event_ids):
        raise ValueError( "Duplicate event_ids detected" )

    if not np.isfinite(vectors).all():
        raise ValueError( "Vectors contain non-finite values" )

    return event_ids, vectors
