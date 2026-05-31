import zarr
import numpy as np
from numcodecs import Blosc


class ZarrConceptStore:
    """
    Tier 2: Concept probe store

    Stores:
        - vecs: float32 [N, dim]
        - concept: object [N] (string labels)
        - slice_id: int32 [N]

    Invariant:
        - separate from Tier 1
        - concept IDs allowed (strings)
    """

    def __init__(self, path: str, dim: int):
        self.root = zarr.open_group(path, mode="a", zarr_version=2)

        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

        if "vecs" in self.root:
            self.vecs = self.root["vecs"]
            self.concept = self.root["concept"]
            self.slice_id = self.root["slice_id"]
            return

        self.vecs = self.root.create_dataset(
            "vecs",
            shape=(0, dim),
            chunks=(2048, dim),
            dtype="float32",
            compressor=compressor,
        )

        # object dtype → stored as bytes/JSON internally
        self.concept = self.root.create_dataset(
            "concept",
            shape=(0,),
            chunks=(2048,),
            dtype=object,
        )

        self.slice_id = self.root.create_dataset(
            "slice_id",
            shape=(0,),
            chunks=(2048,),
            dtype="int32",
        )

    def append(self, vecs, concepts, slice_ids):
        if len(vecs) == 0:
            return

        vecs = np.asarray(vecs, dtype=np.float32)
        concepts = np.asarray(concepts, dtype=object)
        slice_ids = np.asarray(slice_ids, dtype=np.int32)

        n = vecs.shape[0]

        if not (len(concepts) == n and len(slice_ids) == n):
            raise ValueError("batch size mismatch")

        self.vecs.append(vecs, axis=0)
        self.concept.append(concepts, axis=0)
        self.slice_id.append(slice_ids, axis=0)
