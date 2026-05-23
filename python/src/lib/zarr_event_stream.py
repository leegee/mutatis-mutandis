from pathlib import Path
import numpy as np
import zarr

from lib.eebo_logging import logger


class ZarrEventStream:
    """
    Cross-slice streaming abstraction over EEBO Tier1 Zarr event logs.

    This layer is intentionally *strict*:
        - schema mismatches fail loudly
        - missing datasets are not silently skipped
        - embeddings must be explicitly present

    Role
    ----
    Provides deterministic batch streaming of:
        (embeddings, event_ids)

    for FAISS ingestion.

    Invariant
    ---------
    - event_id is the stable, globally unique observation identity
    - vector_id is lexical identity only - NOT used as FAISS key
    - no corpus materialisation
    - no silent schema drift
    - batch-level streaming only
    """

    EXPECTED_GROUP = "events"
    EXPECTED_EMB_KEY = "emb_raw"
    EXPECTED_ID_KEY = "event_id"   # stable observation identity, not vector_id

    def __init__(self, root: str):
        self.root = Path(root)

        self._token_by_id = None
        self._doc_by_id = None

    # ------------------------------------------------------------
    # lookup index (optional, used outside FAISS path)
    # ------------------------------------------------------------

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

            if self.EXPECTED_GROUP not in g:
                continue

            group = g[self.EXPECTED_GROUP]

            if self.EXPECTED_ID_KEY not in group:
                raise KeyError(f"Missing event_id in {slice_dir}")

            # Read entire arrays in one chunk-aware call rather than
            # indexing element-by-element. The previous row-by-row loop
            # bypassed Zarr's chunked I/O and was O(n) Python overhead.
            eids = group[self.EXPECTED_ID_KEY][:]  # (n,) int64

            docs = group["doc_id"][:] if "doc_id" in group else None
            tokens = group["token"][:] if "token" in group else None

            for i, eid in enumerate(eids):
                eid = int(eid)

                if docs is not None:
                    doc_map[eid] = str(docs[i])

                if tokens is not None:
                    token_map[eid] = str(tokens[i])

        self._token_by_id = token_map
        self._doc_by_id = doc_map

        logger.info(f"[stream] indexed events={len(token_map)}")

    def token(self, event_id: int):
        self._build_lookup()
        return self._token_by_id.get(int(event_id))

    def doc_id(self, event_id: int):
        self._build_lookup()
        return self._doc_by_id.get(int(event_id))

    # core FAISS stream

    def iter_embeddings(self, batch_size: int = 8192):
        """
        Yields:
            vecs: (batch, dim) float32
            ids:  (batch,) int64  -- event_id, NOT vector_id
        """

        for slice_dir in sorted(self.root.iterdir()):
            if not slice_dir.is_dir():
                continue

            g = zarr.open_group(str(slice_dir), mode="r")

            if self.EXPECTED_GROUP not in g:
                raise KeyError(f"Missing 'events' group in {slice_dir}")

            group = g[self.EXPECTED_GROUP]

            if self.EXPECTED_EMB_KEY not in group:
                raise KeyError(
                    f"Missing embeddings key '{self.EXPECTED_EMB_KEY}' in {slice_dir}"
                )

            if self.EXPECTED_ID_KEY not in group:
                raise KeyError(
                    f"Missing event_id key in {slice_dir}"
                )

            emb = group[self.EXPECTED_EMB_KEY]
            eids = group[self.EXPECTED_ID_KEY]

            n = eids.shape[0]

            if n == 0:
                continue

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)

                vecs = np.asarray(emb[start:end], dtype=np.float32)
                ids = np.asarray(eids[start:end], dtype=np.int64)

                if len(vecs) != len(ids):
                    raise ValueError(
                        f"Embedding/id mismatch in {slice_dir}: "
                        f"{len(vecs)} vs {len(ids)}"
                    )

                yield vecs, ids