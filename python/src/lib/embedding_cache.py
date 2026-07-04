import numpy as np

class EmbeddingCache:
    """
    Single point of access for embeddings.

    Fetches each event's embedding from the lookup at most once and keeps
    it in memory as a row in a contiguous float32 matrix, so repeated
    np.stack([lookup.get_event(eid)["embedding"] for eid in ids]) calls
    across fit_umap_local / fit_cluster_local / build_depth_layers
    collapse into cheap array slicing.

    Does not change any numeric results — same embeddings, same dtype,
    same ordering semantics (callers still build their own X via
    `matrix(event_ids)`, which preserves the order of `event_ids`).
    """

    def __init__(self, lookup):
        self._lookup = lookup
        self._row_of = {}      # event_id -> row index in _mat
        self._mat = None       # (N, D) float32, grows as needed
        self._cap = 0

    def _ensure_capacity(self, extra):
        needed = len(self._row_of) + extra
        if self._mat is None:
            cap = max(needed, 1024)
            self._mat = np.empty((cap, self._dim), dtype=np.float32)
            self._cap = cap
            return
        if needed > self._cap:
            new_cap = max(needed, self._cap * 2)
            new_mat = np.empty((new_cap, self._mat.shape[1]), dtype=np.float32)
            new_mat[: self._mat.shape[0]] = self._mat
            self._mat = new_mat
            self._cap = new_cap

    def _fetch(self, eid):
        emb = self._lookup.get_event(eid)["embedding"]
        return np.asarray(emb, dtype=np.float32)

    def warm(self, event_ids):
        """Fetch and cache any embeddings not already cached."""
        missing = [eid for eid in event_ids if eid not in self._row_of]
        if not missing:
            return

        if self._mat is None:
            first = self._fetch(missing[0])
            self._dim = first.shape[0]
            self._ensure_capacity(len(missing))
            row = len(self._row_of)
            self._mat[row] = first
            self._row_of[missing[0]] = row
            missing = missing[1:]

        if missing:
            self._ensure_capacity(len(missing))
            for eid in missing:
                row = len(self._row_of)
                self._mat[row] = self._fetch(eid)
                self._row_of[eid] = row

    def matrix(self, event_ids):
        """
        Return an (len(event_ids), D) float32 array with rows in the same
        order as event_ids, fetching/caching as needed.
        """
        if not event_ids:
            raise ValueError("[EmbeddingCache] matrix() called with empty event_ids")
        self.warm(event_ids)
        idx = np.fromiter((self._row_of[eid] for eid in event_ids), dtype=np.int64, count=len(event_ids))
        return self._mat[idx]

    def vector(self, event_id):
        self.warm([event_id])
        return self._mat[self._row_of[event_id]]

