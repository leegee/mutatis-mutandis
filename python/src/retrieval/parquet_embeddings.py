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
            raise ValueError(
                f"{column} has fixed list size {arr.type.list_size}, "
                f"expected {dim}"
            )

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
        raise TypeError(
            f"Unsupported embedding type for {column}: {arr.type}"
        )

    return table.set_column(
        table.schema.get_field_index(column),
        column,
        arr,
    )


def load_embeddings(
    corpus_root: str | Path,
    *,
    year: int,
    scale: str,
    dimensions: int,
) -> tuple[UInt64Array, Float32Array]:
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

    tables = []

    for fragment in fragments:
        table = fragment.to_table( columns=["event_id", embedding_column] )
        table = table.filter( pc.is_valid(table[embedding_column]) )
        table = _normalise_embedding_column( table, embedding_column, dimensions )
        tables.append(table)

    table = pa.concat_tables(tables)

    if table.num_rows == 0:
        return (
            np.empty(0, dtype=np.uint64),
            np.empty((0, dimensions), dtype=np.float32),
        )

    event_ids = np.asarray(
        table.column("event_id").to_numpy(),
        dtype=np.uint64,
    )

    vectors = np.asarray( table.column(embedding_column).to_pylist(), dtype=np.float32 )

    if vectors.ndim != 2:
        raise ValueError( f"Expected 2D embedding matrix, got {vectors.shape}" )

    if vectors.shape[1] != dimensions:
        raise ValueError( f"Expected {dimensions} dimensions, got {vectors.shape[1]}" )

    if len(event_ids) != len(vectors):
        raise ValueError( "event_ids and vectors have different lengths" )

    if len(np.unique(event_ids)) != len(event_ids):
        raise ValueError( "Duplicate event_ids detected" )

    if not np.isfinite(vectors).all():
        raise ValueError( "Vectors contain non-finite values" )

    return event_ids, vectors
