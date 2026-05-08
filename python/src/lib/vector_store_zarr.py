import zarr
import numpy as np
from numcodecs import Blosc


class ZarrVectorStore:
    """
    Tier 1: Corpus field store

    Stores:
        - vecs: float32 [N, dim]
        - ids:  int64   [N]

    Invariant:
        - append-only
        - numeric only
    """

    def __init__(self, path: str, dim: int):
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

    def append(self, vec_batch: np.ndarray, id_batch: np.ndarray):
        if len(vec_batch) == 0:
            return

        vec_batch = np.asarray(vec_batch, dtype=np.float32)
        id_batch = np.asarray(id_batch, dtype=np.int64)

        if vec_batch.shape[0] != id_batch.shape[0]:
            raise ValueError("vec_batch and id_batch size mismatch")

        self.vecs.append(vec_batch, axis=0)
        self.ids.append(id_batch, axis=0)
