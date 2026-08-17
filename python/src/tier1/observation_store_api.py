"""
observation_store_api.py — Tier 1 observation store abstraction

A backend-agnostic contract for Tier 1 observation storage. Zarr was the
original backend but did not scale to current Tier 1 observation volume;
Apache Parquet + DuckDB is now the only backend and the one these protocols
are written against.

Why this exists
---------------
Tier 1 observation volume has outgrown reliable full-corpus in-memory
consumption. Call sites must depend on a streaming / selective-load API
rather than concrete arrays. The protocols below encode exactly the
capabilities the pipeline uses:

    ObservationWriter  — append-only multi-scale event materialisation
    ObservationStream  — batch streaming of embeddings for FAISS ingestion
    ObservationLookup  — selective metadata + lazy embedding access for
                         neighbourhood analysis, clustering, and plots

Design rules
------------
1. event_id is the sole stable observation identity.
2. vector_id is lexical identity only; never a retrieval key.
3. Multi-scale embeddings (local / medium / broad) remain aligned per event.
4. Writers are append-only; concurrent writers are undefined.
5. Streaming never materialises the full corpus.
6. Lookup may keep compact metadata in memory; embeddings must be
   reconstructable on demand (or streamed) so peak RSS stays bounded.
7. Backends may partition by year, shard, or hive; the API does not expose
   partition layout except via optional year filters.
8. Scales (local / medium / broad) are independently loadable. Streaming
   and lookup methods take an optional `scales` argument; the backend must
   read/reconstruct only the requested scale(s), never the others. This
   lets a caller pull "local" for one pass and "medium" for a later pass
   without ever holding more than one scale's embedding matrix at a time.

Backend selection
-----------------
Use the factory helpers at the bottom of this module:

    writer = open_observation_writer("parquet", path, dim=768)
    stream = open_observation_stream("parquet", root)
    lookup = open_observation_lookup("parquet", root)

"parquet" (DuckDB-backed) is the only registered backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import (
    Any,
    Iterator,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from lib.corpus_config import MASKED_EVENTSTORE_T1_PATH, EVENTSTORE_T1_PATH

import numpy as np

# Canonical multi-scale names — order is significant for ensemble weights.
SCALES: tuple[str, ...] = ("local", "medium", "broad")

# Default ensemble weights used by FAISS build and neighbourhood search.
# local : medium : broad
DEFAULT_ENSEMBLE_WEIGHTS: tuple[float, float, float] = (0.25, 0.50, 0.25)

# Metadata columns that every backend must be able to surface.
# Embeddings are deliberately excluded — they have their own access path.
METADATA_FIELDS: tuple[str, ...] = (
    "event_id",          # int64  — unique observation identity
    "concept_id",        # int64  — stable (doc, token_idx) hash
    "vector_id",         # int64  — lexical identity from corpus
    "corpus",            # str    — corpus partition key
    "doc_id",            # str
    "token",             # str
    "token_idx",         # int64  — position in original document
    "window_id",         # int64  — transformer window start
    "window_token_pos",  # int64  — position inside the window (-1 = absent)
    "pub_year",          # int16
)

# Sentinel for missing window_token_pos. Never a valid token position.
NO_WINDOW_TOKEN_POS: int = -1


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

@runtime_checkable
class ObservationWriter(Protocol):
    """
    Append-only materialisation of multi-scale contextual observations.

    Implementations must keep all columns of a single append call aligned
    (same length, same row order). Partial writes of a batch are not allowed.
    """

    def append_events(
        self,
        *,
        event_id: np.ndarray,
        corpus: np.ndarray,
        doc_id: np.ndarray,
        token: np.ndarray,
        token_idx: np.ndarray,
        pub_year: np.ndarray,
        local_window_id: Optional[np.ndarray] = None,
        local_window_token_pos: Optional[np.ndarray] = None,
        medium_window_id: Optional[np.ndarray] = None,
        medium_window_token_pos: Optional[np.ndarray] = None,
        broad_window_id: Optional[np.ndarray] = None,
        broad_window_token_pos: Optional[np.ndarray] = None,
        emb_local: Optional[np.ndarray] = None,
        emb_medium: Optional[np.ndarray] = None,
        emb_broad: Optional[np.ndarray] = None,
    ) -> None:
        """
        Append one batch of aligned observations.

        event_id/corpus/doc_id/token/token_idx/pub_year are always
        required and must all have length n.

        Each scale's (emb_<scale>, <scale>_window_id,
        <scale>_window_token_pos) is an optional group: a caller that
        only computed a subset of scales for this run omits the other
        groups entirely rather than passing placeholder arrays. When
        supplied, emb_<scale> is (n, dim) float32 and the two window
        arrays are 1-D length n. At least one scale's group must be
        supplied. Backends must write null for a scale's columns on rows
        where that scale's group wasn't supplied, not zeros or a
        sentinel — later calls may add a previously-omitted scale to the
        same store.
        """
        ...

    @property
    def n_events(self) -> int:
        """Number of observations already written."""
        ...

    def get_doc_keys(self) -> set[tuple[str, str]]:
        """
        Set of (corpus, doc_id) pairs already present.

        Used by incremental Tier-1 runs to skip documents that have already
        been materialised.
        """
        ...

    def get_event_ids(self) -> set[int]:
        """
        Set of event_ids already present.

        Used by shard-merge and incremental writers to avoid duplicates.
        """
        ...

    def embedding_dim(self) -> int:
        """Dimensionality of each scale's embedding vectors."""
        ...

    def __len__(self) -> int:
        ...


# ---------------------------------------------------------------------------
# Stream
# ---------------------------------------------------------------------------

@runtime_checkable
class ObservationStream(Protocol):
    """
    Deterministic batch streaming of multi-scale embeddings + identities.

    Primary consumer: FAISS index construction (Tier 1.5).
    Must not load the full corpus into memory.
    """

    def iter_multi_scale_embeddings(
        self,
        batch_size: int = 8192,
        year_filter: Optional[set[int]] = None,
        year_manifest: Optional[Mapping[Any, np.ndarray]] = None,
        scales: Sequence[str] = SCALES,
    ) -> Iterator[
        tuple[
            Optional[np.ndarray],
            Optional[np.ndarray],
            Optional[np.ndarray],
            np.ndarray,
            np.ndarray,
        ]
    ]:
        """
        Yield aligned batches:

            (emb_local, emb_medium, emb_broad, event_ids, pub_years)

        event_ids and pub_years are always populated. The embedding
        position for any scale *not* in `scales` is None rather than an
        (n, dim) array — backends must not read or reconstruct embedding
        data for scales that weren't requested.

        This is the selective-load path: to work through scales one at a
        time (e.g. build a "local"-only FAISS index in one pass, then a
        "medium"-only index in a later pass) call this once per scale
        rather than requesting all three and discarding what isn't
        needed. Only one scale's embedding matrix need ever be resident
        at a time. The default, `scales=SCALES`, reproduces the original
        all-three-scales behaviour.

        Filtering by year_filter (when supplied) is applied *before* yield
        so alignment is preserved.

        year_manifest is an optional backend-specific optimisation to avoid
        re-reading pub_year arrays; the Parquet backend may ignore it if
        it isn't useful for its layout.
        """
        ...

    def year_bounds(self) -> tuple[int, int]:
        """(min_pub_year, max_pub_year) across the whole store."""
        ...



# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

@runtime_checkable
class ObservationLookup(Protocol):
    """
    Selective access to observation metadata and embeddings.

    Metadata may be held in compact columnar form. Embeddings must be
    obtainable without materialising the full (N, 3*dim) matrix.
    """

    # --- size / schema ---
    def __len__(self) -> int:
        ...

    @property
    def available_years(self) -> np.ndarray:
        """Sorted unique pub_year values present in the loaded metadata."""
        ...

    # --- identity resolution ---
    def get_pos(self, event_id: int) -> int:
        """
        event_id → dense row position in the loaded metadata tables.
        Raises KeyError if the event is not present.
        """
        ...

    def get_event_metadata(self, event_id: int) -> dict:
        """
        Provenance dict for one event (no embedding reconstruction).

        Keys match METADATA_FIELDS (window_token_pos may be None).
        """
        ...

    def get_event(self, event_id: int) -> dict:
        """
        Provenance + ensemble embedding for one event.
        Prefer get_event_metadata when the vector is not required.
        """
        ...

    # --- form / position queries ---
    def iter_matching_event_ids(
        self,
        forms: Sequence[str],
        false_positives: Optional[Sequence[str]] = None,
    ) -> Iterator[int]:
        """
        Yield distinct event_ids whose token (case-insensitive) is in forms
        and not in false_positives.
        """
        ...

    def find_matching_event_ids(
        self,
        forms: Sequence[str],
        false_positives: Optional[Sequence[str]] = None,
    ) -> list[int]:
        ...

    def find_event_ids_by_positions(
        self,
        positions: Sequence[tuple[str, str, int]],
    ) -> dict[tuple[str, str, int], list[int]]:
        """
        Map (corpus, doc_id, token_idx) → list of event_ids that observe
        that corpus occurrence (multiple windows → multiple events).
        """
        ...

    # --- embedding access ---

    # Selective single-scale primitives. Implementations must satisfy
    # these by reading/reconstructing *only* the requested scale — never
    # the other two. This is what makes "load 'local' now, load 'medium'
    # later, never hold both" possible: callers loop over scales one at a
    # time instead of calling the ensemble methods below (which touch
    # every scale named in `scales`/`weights`).
    def get_scale_embedding(self, pos: int, scale: str) -> np.ndarray:
        """
        Raw (unweighted) embedding for one row position, one scale.

        Raises ValueError if `scale` is not one of SCALES.
        """
        ...

    def get_scale_embeddings(
        self,
        event_ids: Sequence[int],
        scale: str,
    ) -> np.ndarray:
        """
        (n, dim) matrix for a single scale, aligned to event_ids.

        To switch scales, just call again with a different `scale` —
        the previous scale's data does not need to stay resident, and
        the backend must not eagerly load scales beyond the one asked
        for here.

        Raises ValueError if `scale` is not one of SCALES.
        """
        ...

    def get_ensemble_embedding(
        self,
        pos: int,
        weights: Sequence[float] = DEFAULT_ENSEMBLE_WEIGHTS,
        scales: Sequence[str] = SCALES,
    ) -> np.ndarray:
        """
        Weighted combination of embeddings for one row position.

        `scales` and `weights` are paired positionally (scales[i] gets
        weights[i]) and must be the same length. The default reproduces
        the original all-three-scale ensemble. Only the scales named in
        `scales` are read — e.g. scales=("local",), weights=(1.0,) is
        equivalent to get_scale_embedding(pos, "local") but expressed
        through the ensemble interface.
        """
        ...

    def get_embeddings(
        self,
        event_ids: Sequence[int],
        weights: Sequence[float] = DEFAULT_ENSEMBLE_WEIGHTS,
        scales: Sequence[str] = SCALES,
    ) -> np.ndarray:
        """
        (n, dim) ensemble matrix aligned to event_ids.

        `scales`/`weights` behave as in get_ensemble_embedding: only the
        named scales are read. Prefer get_scale_embeddings when you want
        one scale's raw vectors rather than a weighted combination.
        """
        ...

    def get_concatenated_embeddings(
        self,
        event_ids: Sequence[int],
        scales: Sequence[str] = SCALES,
    ) -> np.ndarray:
        """
        (n, len(scales)*dim) matrix: per-scale L2-normalised vectors
        concatenated in `scales` order. Used by clustering paths that
        want to keep scale structure. Only the named scales are read;
        pass a single scale (e.g. scales=("local",)) to get its raw
        L2-normalised (n, dim) block without reading the others.
        """
        ...

    # Optional: attach an external vector source (e.g. per-year FAISS).
    # Backends that store embeddings inline may implement this as a no-op.
    def attach_index(self, index: Any) -> None:
        ...


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, dict[str, Any]] = {}


def register_backend(
    name: str,
    *,
    writer: Optional[type] = None,
    stream: Optional[type] = None,
    lookup: Optional[type] = None,
) -> None:
    """Register concrete classes for a storage backend name."""
    entry = _BACKENDS.setdefault(name, {})
    if writer is not None:
        entry["writer"] = writer
    if stream is not None:
        entry["stream"] = stream
    if lookup is not None:
        entry["lookup"] = lookup


def open_observation_writer(
    backend: str,
    path: str | Path,
    *,
    dim: int,
    **kwargs: Any,
) -> ObservationWriter:
    """Open an ObservationWriter for the named backend."""
    cls = _require(backend, "writer")
    return cls(path, dim=dim, **kwargs)


def open_observation_stream(
    backend: str,
    root: str | Path,
    **kwargs: Any,
) -> ObservationStream:
    """Open an ObservationStream for the named backend."""
    cls = _require(backend, "stream")
    return cls(root, **kwargs)


def open_observation_lookup(
    backend: str,
    root: str | Path,
    **kwargs: Any,
) -> ObservationLookup:
    """Open an ObservationLookup for the named backend."""
    cls = _require(backend, "lookup")
    return cls(root, **kwargs)


def _require(backend: str, role: str) -> type:
    if backend not in _BACKENDS or role not in _BACKENDS[backend]:
        available = sorted(_BACKENDS.keys()) or ["(none registered)"]
        raise KeyError(
            f"No {role!r} registered for backend {backend!r}. "
            f"Known backends: {available}"
        )
    return _BACKENDS[backend][role]


def list_backends() -> list[str]:
    return sorted(_BACKENDS.keys())


def default_store_path(store_backend: str, masked: bool) -> Path:
    return (
        MASKED_EVENTSTORE_T1_PATH
        if masked
        else EVENTSTORE_T1_PATH
    )


def resolve_store_path(
    *,
    store_backend: str,
    masked: bool,
    store: str | Path | None = None,
    shard: int | None = None,
    num_shards: int = 1,
) -> Path:
    if store is None:
        path = default_store_path(store_backend, masked)
    else:
        path = Path(store)

    if num_shards > 1:
        path = path.parent / f"{path.name}_shard{shard}"

    return path


def configure_store_backend(store_backend: str, *, num_shards: int) -> None:
    """
    Apply backend-specific runtime configuration.

    This affects the current process only and must be called before the
    observation store is opened. No configuration is currently needed for
    the Parquet backend; kept as a hook for future backend-specific setup.
    """
    return
