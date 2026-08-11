"""
observation_store_api.py — Tier 1 observation store abstraction

Stage 1 of the storage refactor: a backend-agnostic contract that the
existing Zarr apparatus implements. Stage 2 will add a parallel Apache
Parquet + DuckDB backend that satisfies the same protocols.

Why this exists
---------------
Tier 1 observation volume has outgrown reliable full-corpus in-memory
consumption. Call sites must depend on a streaming / selective-load API
rather than concrete Zarr arrays. The protocols below encode exactly the
capabilities the pipeline already uses:

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

Backend selection
-----------------
Use the factory helpers at the bottom of this module:

    writer = open_observation_writer("zarr", path, dim=768)
    stream = open_observation_stream("zarr", root)
    lookup = open_observation_lookup("zarr", root)

Stage 2 will register "parquet" (DuckDB-backed) under the same names.
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
        event_id: np.ndarray,
        concept_id: np.ndarray,
        emb_local: np.ndarray,
        emb_medium: np.ndarray,
        emb_broad: np.ndarray,
        vector_id: np.ndarray,
        corpus: np.ndarray,
        doc_id: np.ndarray,
        pub_year: np.ndarray,
        token_idx: np.ndarray,
        token: np.ndarray,
        window_id: np.ndarray,
        window_token_pos: np.ndarray,
    ) -> None:
        """
        Append one batch of aligned observations.

        All array arguments must have length n. Embedding arrays are
        (n, dim) float32; identity / coordinate arrays are 1-D.
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
    ) -> Iterator[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]:
        """
        Yield aligned batches:

            (emb_local, emb_medium, emb_broad, event_ids, pub_years)

        All five arrays share the same leading dimension. Filtering by
        year_filter (when supplied) is applied *before* yield so alignment
        is preserved.

        year_manifest is an optional backend-specific optimisation (Zarr
        uses it to avoid re-reading pub_year arrays); other backends may
        ignore it.
        """
        ...

    def year_bounds(self) -> tuple[int, int]:
        """(min_pub_year, max_pub_year) across the whole store."""
        ...

    def build_year_manifest(self) -> Mapping[Any, np.ndarray]:
        """
        Optional one-shot scan that returns a structure later passed as
        year_manifest to iter_multi_scale_embeddings. Backends that have no
        use for it may return an empty mapping.
        """
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

    The protocol deliberately mirrors the historical ZarrEventLookup surface
    so existing Tier-2 / Tier-3 call sites need only type changes (or none).
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
    def get_ensemble_embedding(
        self,
        pos: int,
        weights: Sequence[float] = DEFAULT_ENSEMBLE_WEIGHTS,
    ) -> np.ndarray:
        """Weighted combination of the three scales for one row position."""
        ...

    def get_embeddings(
        self,
        event_ids: Sequence[int],
        weights: Sequence[float] = DEFAULT_ENSEMBLE_WEIGHTS,
    ) -> np.ndarray:
        """
        (n, dim) ensemble matrix aligned to event_ids.
        """
        ...

    def get_concatenated_embeddings(
        self,
        event_ids: Sequence[int],
    ) -> np.ndarray:
        """
        (n, 3*dim) matrix: per-scale L2-normalised vectors concatenated.
        Used by clustering paths that want to keep scale structure.
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
