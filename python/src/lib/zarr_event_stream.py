from pathlib import Path
import numpy as np
import zarr


class ZarrEventStream:
    """
    Cross-slice streaming abstraction over EEBO Zarr event logs.

    Role
    ----
    Provides a deterministic, memory-bounded view over:

        ZARR_ROOT/tier1/<slice>/events/*

    yielding (embeddings, vector_ids) batches suitable for FAISS ingestion.

    Invariant
    ---------
    - no full corpus materialisation
    - deterministic slice ordering
    - batch-level streaming only
    """

    def __init__(self, root: str):
        self.root = Path(root)

    def iter_embeddings(self, batch_size: int = 8192):
        """
        Yields:
            vecs: (batch, dim) float32
            ids:  (batch,) int64
        """

        for slice_dir in sorted(self.root.iterdir()):
            if not slice_dir.is_dir():
                continue

            g = zarr.open_group(str(slice_dir), mode="r")

            try:
                emb = g["events"]["emb_norm"]
                ids = g["events"]["vector_id"]
            except KeyError:
                continue

            n = emb.shape[0]
            if n == 0:
                continue

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)

                vecs = np.asarray(emb[start:end], dtype=np.float32)
                vids = np.asarray(ids[start:end], dtype=np.int64)

                yield vecs, vids
