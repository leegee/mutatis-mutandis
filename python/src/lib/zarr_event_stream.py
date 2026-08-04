# lib/zarr_event_stream.py
from pathlib import Path
import numpy as np
import zarr

from lib.corpus_logging import logger
from lib.zarr_store_dirs import store_dirs

class ZarrEventStream:
    """
    Streaming abstraction over EEBO Tier1 Zarr event logs.

    Supports both a single observation store and a directory of stores
    (legacy slice layout).

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
    EXPECTED_EMB_KEY = "emb_medium"
    EXPECTED_ID_KEY = "event_id"   # stable observation identity, not vector_id

    def __init__(self, root: str):
        self.root = Path(root)
        self._token_by_id = None
        self._doc_by_id = None


def iter_multi_scale_embeddings(
    self,
    batch_size: int = 8192,
    year_filter: set[int] | None = None,
    year_manifest: dict[Path, np.ndarray] | None = None,
):
    """
    Yields tuples of:
        (emb_local, emb_medium, emb_broad, event_ids, pub_years)

    Invariant:
        All yielded arrays are aligned:
            emb_local[i]
            emb_medium[i]
            emb_broad[i]
            event_ids[i]
            pub_years[i]

        describe the same semantic observation event.

    year_filter:
        If given, only events whose pub_year is in the filter are yielded.
        Filtering must happen before yield so that embeddings and IDs remain
        aligned with years.
    """

    for store_dir in store_dirs(self.root):
        g = zarr.open_group(str(store_dir), mode="r")
        group = g["events"]

        if "pub_year" not in group:
            raise KeyError(
                f"Missing 'pub_year' in {store_dir} — this store predates "
                f"per-event pub_year and needs to be backfilled before "
                f"multi-scale streaming can proceed."
            )

        years_dataset = group["pub_year"]
        eids = group["event_id"]

        emb_l = group["emb_local"]
        emb_m = group["emb_medium"]
        emb_b = group["emb_broad"]

        n = years_dataset.shape[0]

        if n == 0:
            continue

        cached_years = (
            year_manifest.get(store_dir)
            if year_manifest is not None
            else None
        )

        if year_filter is not None:
            store_years_full = (
                cached_years
                if cached_years is not None
                else np.asarray(
                    years_dataset[:],
                    dtype=np.int16,
                )
            )

            store_lo = int(store_years_full.min())
            store_hi = int(store_years_full.max())

            filter_lo = min(year_filter)
            filter_hi = max(year_filter)

            if store_hi < filter_lo or store_lo > filter_hi:
                continue

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)

            if cached_years is not None:
                batch_years = cached_years[start:end]
            else:
                batch_years = np.asarray(
                    years_dataset[start:end],
                    dtype=np.int16,
                )

            batch_eids = np.asarray(
                eids[start:end],
                dtype=np.int64,
            )

            batch_emb_l = np.asarray(
                emb_l[start:end],
                dtype=np.float32,
            )

            batch_emb_m = np.asarray(
                emb_m[start:end],
                dtype=np.float32,
            )

            batch_emb_b = np.asarray(
                emb_b[start:end],
                dtype=np.float32,
            )

            if year_filter is not None:
                keep = np.isin(
                    batch_years,
                    list(year_filter),
                )

                if not keep.any():
                    continue

                batch_years = batch_years[keep]
                batch_eids = batch_eids[keep]

                batch_emb_l = batch_emb_l[keep]
                batch_emb_m = batch_emb_m[keep]
                batch_emb_b = batch_emb_b[keep]

                if 1734 in year_filter:
                    logger.info(
                        "[stream-debug] store=%s filtered batch years=%s count=%d",
                        store_dir,
                        {
                            int(y): int((batch_years == y).sum())
                            for y in np.unique(batch_years)
                        },
                        len(batch_eids),
                    )

            yield (
                batch_emb_l,
                batch_emb_m,
                batch_emb_b,
                batch_eids,
                batch_years,
            )


    def _build_lookup(self):
        if self._token_by_id is not None:
            return
        logger.info("[stream] building global event lookup")
        token_map = {}
        doc_map = {}

        for store_dir in store_dirs(self.root):
            g = zarr.open_group(str(store_dir), mode="r")
            if self.EXPECTED_GROUP not in g:
                continue

            group = g[self.EXPECTED_GROUP]
            if self.EXPECTED_ID_KEY not in group:
                raise KeyError(f"Missing event_id in {store_dir}")

            eids   = group[self.EXPECTED_ID_KEY][:]
            docs   = group["doc_id"][:] if "doc_id" in group else None
            tokens = group["token"][:]  if "token"  in group else None

            for i, eid in enumerate(eids):
                eid = int(eid)

                if docs is not None:
                    doc_map[eid] = str(docs[i])

                if tokens is not None:
                    token_map[eid] = str(tokens[i])

        self._token_by_id = token_map
        self._doc_by_id   = doc_map
        logger.info(f"[stream] indexed events={len(token_map)}")


    def token(self, event_id: int):
        self._build_lookup()
        return self._token_by_id.get(int(event_id))


    def doc_id(self, event_id: int):
        self._build_lookup()
        return self._doc_by_id.get(int(event_id))


    def iter_embeddings(self, batch_size: int = 8192):
        """
        Yields:
            vecs: (batch, dim) float32
            ids:  (batch,) int64  -- event_id, NOT vector_id
        """

        for store_dir in store_dirs(self.root):
            g = zarr.open_group(str(store_dir), mode="r")

            if self.EXPECTED_GROUP not in g:
                raise KeyError(f"Missing 'events' group in {store_dir}")

            group = g[self.EXPECTED_GROUP]

            if self.EXPECTED_EMB_KEY not in group:
                raise KeyError(
                    f"Missing embeddings key '{self.EXPECTED_EMB_KEY}' in {store_dir}"
                )

            if self.EXPECTED_ID_KEY not in group:
                raise KeyError(
                    f"Missing event_id key in {store_dir}"
                )

            emb  = group[self.EXPECTED_EMB_KEY]
            eids = group[self.EXPECTED_ID_KEY]

            n = eids.shape[0]

            if n == 0:
                continue

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)

                vecs = np.asarray(emb[start:end],  dtype=np.float32)
                ids  = np.asarray(eids[start:end], dtype=np.int64)

                if len(vecs) != len(ids):
                    raise ValueError(
                        f"Embedding/id mismatch in {store_dir}: "
                        f"{len(vecs)} vs {len(ids)}"
                    )

                yield vecs, ids


    def year_bounds(self) -> tuple[int, int]:
        lo, hi = None, None
        for store_dir in store_dirs(self.root):
            g = zarr.open_group(str(store_dir), mode="r")
            group = g["events"]
            years = np.asarray(group["pub_year"][:], dtype=np.int16)
            if years.size == 0:
                continue
            ymin, ymax = int(years.min()), int(years.max())
            lo = ymin if lo is None else min(lo, ymin)
            hi = ymax if hi is None else max(hi, ymax)
        if lo is None:
            raise RuntimeError("No pub_year data found in any store")
        return lo, hi


    def build_year_manifest(self) -> dict[Path, np.ndarray]:
        """One-time scan: store_dir -> full pub_year array, cached in memory
        (int16, so even a huge corpus is a few MB total)."""
        manifest = {}
        for store_dir in store_dirs(self.root):
            g = zarr.open_group(str(store_dir), mode="r")
            manifest[store_dir] = np.asarray(g["events"]["pub_year"][:], dtype=np.int16)
        return manifest
