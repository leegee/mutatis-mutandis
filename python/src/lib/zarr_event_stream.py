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
        # lazy-built lookup cache
        self._token_by_id = None
        self._doc_by_id = None


    def _build_lookup(self):
        if self._token_by_id is not None:
            return

        logger.info("[stream] building global event lookup")

        token_map = {}
        doc_map = {}

        for slice_dir in sorted(self.root.iterdir()):
            if not slice_dir.is_dir():
                continue

            g = zarr.open_group(str(slice_dir), mode="r")

            try:
                vids = g["events"]["vector_id"]
                docs = g["events"]["doc_id"]
                tokens = g["events"]["token_idx"]
            except KeyError:
                continue

            n = vids.shape[0]

            for i in range(n):
                vid = int(vids[i])
                doc_map[vid] = str(docs[i])
                token_map[vid] = int(tokens[i])

        self._token_by_id = token_map
        self._doc_by_id = doc_map
        logger.info(f"[stream] indexed events={len(token_map)}")


    def token(self, event_id: int) -> int:
        self._build_lookup()
        return self._token_by_id.get(int(event_id), None)


    def doc_id(self, event_id: int):
        self._build_lookup()
        return self._doc_by_id.get(int(event_id), None)


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
                emb = g["events"]["mb_raw"]
                vids = g["events"]["vector_id"]
            except KeyError:
                continue

            n = vids.shape[0]

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)

                yield (
                    np.asarray(emb[start:end], dtype=np.float32),
                    np.asarray(vids[start:end], dtype=np.int64),
                )