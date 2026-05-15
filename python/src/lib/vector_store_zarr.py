#!/usr/bin/env python
"""
vector_store_zarr.py

Append-only vector store backed by Zarr v2.

Stores a float32 embedding matrix (vecs) and a parallel int64 vector-id
array (ids).  The two arrays are always the same length; that invariant
is checked on every append.

truncate(n_rows) shrinks both arrays to n_rows, used by the resume logic
in tier1_corpus2zarr.py to discard any partially-written embeddings from
a previous interrupted run before restarting.
"""

import numpy as np
import zarr
from numcodecs import Blosc


class ZarrVectorStore:

    def __init__(self, path: str, dim: int):
        self.path = path
        self.dim = dim
        self.root = zarr.open_group(path, mode="a", zarr_version=2)

        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

        if "vecs" in self.root:
            self.vecs = self.root["vecs"]
            self.ids = self.root["ids"]
            return

        self.vecs = self.root.create_dataset(
            "vecs",
            shape=(0, dim),
            chunks=(4096, dim),
            dtype="float32",
            compressor=compressor,
        )

        self.ids = self.root.create_dataset(
            "ids",
            shape=(0,),
            chunks=(4096,),
            dtype="int64",
            compressor=compressor,
        )

    def __len__(self) -> int:
        return self.ids.shape[0]

    def append(self, vec_batch: np.ndarray, id_batch: np.ndarray) -> None:
        """
        Append a batch of embeddings and their vector ids.  Both arrays
        must have the same leading dimension.  vec_batch must be float32
        with shape (N, dim); id_batch must be int64 with shape (N,).
        """
        if len(vec_batch) == 0:
            return

        vec_batch = np.asarray(vec_batch, dtype=np.float32)
        id_batch = np.asarray(id_batch, dtype=np.int64)

        if vec_batch.shape[0] != id_batch.shape[0]:
            raise ValueError(
                f"vec_batch length {vec_batch.shape[0]} != "
                f"id_batch length {id_batch.shape[0]}"
            )

        if vec_batch.ndim != 2 or vec_batch.shape[1] != self.dim:
            raise ValueError(
                f"vec_batch has shape {vec_batch.shape}, expected (N, {self.dim})"
            )

        self.vecs.append(vec_batch, axis=0)
        self.ids.append(id_batch, axis=0)

    def truncate(self, n_rows: int) -> None:
        """
        Shrink both arrays to n_rows, discarding everything beyond that
        point.  Used at resume time to remove partially-written embeddings
        left by an interrupted run.

        Zarr v2 resize() supports shrinking in place without rewriting
        the retained data.
        """
        current = self.ids.shape[0]
        if n_rows > current:
            raise ValueError(
                f"Cannot truncate to {n_rows}: store only has {current} rows"
            )
        if n_rows == current:
            return

        self.vecs.resize(n_rows, self.dim)
        self.ids.resize(n_rows)
