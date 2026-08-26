"""
observation_store_api.py — Tier 1 observation store API

Apache Parquet + DuckDB is the sole Tier 1 observation store.

This module defines the storage contracts used by the pipeline and exposes
factory helpers for the concrete Parquet implementation.

Why this exists
---------------

Tier 1 observation volume has outgrown reliable full-corpus in-memory
consumption. Call sites must depend on a streaming / selective-load API
rather than concrete arrays.

Design rules

1. event_id is the sole stable observation identity.
2. vector_id is lexical identity only; never a retrieval key.
3. Multi-scale embeddings (local / medium / broad) remain aligned per event.
4. Writers are append-only; concurrent writers are undefined.
5. Streaming never materialises the full corpus.
6. Lookup may keep compact metadata in memory; embeddings must be
   reconstructable on demand (or streamed) so peak RSS stays bounded.
7. The Parquet backend may partition by year, shard, or hive; the API does
   not expose physical partition layout except via optional year filters.
8. Scales are independently loadable. Callers can request one scale without
   causing the backend to read the others.
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

from lib.corpus_config import (
    MASKED_EVENTSTORE_T1_PATH,
    EVENTSTORE_T1_PATH,
)


# Canonical multi-scale names. Order is significant for ensemble weights.
SCALES: tuple[str, ...] = ("local", "medium", "broad")

# Default ensemble weights used by FAISS build and neighbourhood search.
DEFAULT_ENSEMBLE_WEIGHTS: tuple[float, float, float] = (
    0.25,
    0.50,
    0.25,
)

# Metadata columns every observation store exposes.
# Embeddings have their own selective access path.
METADATA_FIELDS: tuple[str, ...] = (
    "event_id",
    "corpus",
    "doc_id",
    "token",
    "token_idx",
    "pub_year",
    "local_window_id",
    "local_window_token_pos",
    "medium_window_id",
    "medium_window_token_pos",
    "broad_window_id",
    "broad_window_token_pos",
)

# Sentinel for missing window_token_pos. Never a valid token position.
NO_WINDOW_TOKEN_POS: int = -1


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

        event_id/corpus/doc_id/token/token_idx/pub_year are always required
        and must all have length n.

        Each scale's embedding and window columns form an optional group.
        At least one scale must be supplied. Missing scales must be written
        as null rather than zeros or sentinels so later writes can add them.
        """
        ...

    @property
    def n_events(self) -> int:
        """Number of observations already written."""
        ...

    def get_doc_keys(self) -> set[tuple[str, str]]:
        """
        Return (corpus, doc_id) pairs already present.

        Used by incremental Tier 1 runs to skip materialised documents.
        """
        ...

    def get_event_ids(self) -> set[int]:
        """
        Return event_ids already present.

        Used by incremental writers and shard merging to avoid duplicates.
        """
        ...

    def embedding_dim(self) -> int:
        """Dimensionality of each scale's embedding vector."""
        ...

    def __len__(self) -> int:
        ...


@runtime_checkable
class ObservationStream(Protocol):
    """
    Deterministic batch streaming of multi-scale embeddings and identities.

    Primary consumer: FAISS index construction.

    The implementation must never materialise the complete corpus.
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

        Embedding positions for scales not included in `scales` are None.

        The backend must not read or reconstruct unrequested scales.

        `year_filter` is applied before yielding rows so alignment is
        preserved.

        `year_manifest` is an optional optimisation for callers that already
        have year metadata available. The Parquet implementation may ignore
        it when direct predicate filtering is cheaper.
        """
        ...

    def year_bounds(self) -> tuple[int, int]:
        """Return (min_pub_year, max_pub_year) across the store."""
        ...


@runtime_checkable
class ObservationLookup(Protocol):
    """
    Selective access to observation metadata and embeddings.

    Metadata may be held in compact columnar form. Embeddings must be
    obtainable without materialising the full multi-scale corpus.
    """

    def __len__(self) -> int:
        ...

    @property
    def available_years(self) -> np.ndarray:
        """Sorted unique publication years present in the lookup."""
        ...


    def get_scale_embeddings(
        self,
        event_ids: Sequence[int],
        scale: str,
    ) -> np.ndarray:
        ...

    def get_pos(self, event_id: int) -> int:
        """
        Map event_id to dense row position in loaded metadata.

        Raises KeyError if event_id is not present.
        """
        ...

    def get_event_metadata(self, event_id: int) -> dict:
        """
        Return provenance metadata without reconstructing an embedding.

        Keys correspond to METADATA_FIELDS.
        """
        ...

    def get_event(self, event_id: int) -> dict:
        """
        Return provenance plus the ensemble embedding for one event.
        """
        ...

    def iter_matching_event_ids(
        self,
        forms: Sequence[str],
        false_positives: Optional[Sequence[str]] = None,
    ) -> Iterator[int]:
        """
        Yield distinct event_ids whose token is in forms and not in
        false_positives.

        Token matching is case-insensitive.
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
        Map (corpus, doc_id, token_idx) to observing event_ids.

        Multiple windows may produce multiple events for one corpus position.
        """
        ...

    def get_scale_embedding(
        self,
        pos: int,
        scale: str,
    ) -> np.ndarray:
        """
        Return one raw embedding for one row and one scale.

        Only the requested scale may be read.
        """
        ...

    def get_scale_embeddings(
        self,
        event_ids: Sequence[int],
        scale: str,
    ) -> np.ndarray:
        """
        Return an (n, dim) matrix for one scale aligned to event_ids.

        Only the requested scale may be read.
        """
        ...

    def get_ensemble_embedding(
        self,
        pos: int,
        weights: Sequence[float] = DEFAULT_ENSEMBLE_WEIGHTS,
        scales: Sequence[str] = SCALES,
    ) -> np.ndarray:
        """
        Return a weighted combination of the requested scales.

        scales and weights are paired positionally and must have equal
        lengths.
        """
        ...

    def get_embeddings(
        self,
        event_ids: Sequence[int],
        weights: Sequence[float] = DEFAULT_ENSEMBLE_WEIGHTS,
        scales: Sequence[str] = SCALES,
    ) -> np.ndarray:
        """
        Return an (n, dim) ensemble matrix aligned to event_ids.

        Only scales named in `scales` may be read.
        """
        ...

    def get_concatenated_embeddings(
        self,
        event_ids: Sequence[int],
        scales: Sequence[str] = SCALES,
    ) -> np.ndarray:
        """
        Return per-scale L2-normalised vectors concatenated in `scales`
        order.

        Only requested scales may be read.
        """
        ...

    def attach_index(self, index: Any) -> None:
        """
        Attach an external vector source such as a per-year FAISS index.

        Inline Parquet implementations may treat this as a no-op.
        """
        ...


def open_observation_writer(
    path: str | Path,
    *,
    dim: int,
    **kwargs: Any,
) -> ObservationWriter:
    """
    Open the Parquet observation writer.

    The storage backend is fixed; callers do not select a backend.
    """
    from lib.parquet_observation_backend import ParquetObservationWriter

    return ParquetObservationWriter(
        path,
        dim=dim,
        **kwargs,
    )


def open_observation_stream(
    root: str | Path,
    **kwargs: Any,
) -> ObservationStream:
    """
    Open the Parquet observation stream.
    """
    from lib.parquet_observation_backend import ParquetObservationStream

    return ParquetObservationStream(
        root,
        **kwargs,
    )


def open_observation_lookup(
    root: str | Path,
    **kwargs: Any,
) -> ObservationLookup:
    """
    Open the Parquet observation lookup.
    """
    from lib.parquet_observation_backend import ParquetObservationLookup

    return ParquetObservationLookup(
        root,
        **kwargs,
    )


def default_store_path(masked: bool = False) -> Path:
    """
    Return the canonical Tier 1 Parquet root.
    """
    return (
        MASKED_EVENTSTORE_T1_PATH
        if masked
        else EVENTSTORE_T1_PATH
    )


def resolve_store_path(
    *,
    masked: bool,
    store: str | Path | None = None,
    shard: int | None = None,
    num_shards: int = 1,
) -> Path:
    """
    Resolve the canonical Parquet root, optionally applying shard naming.

    A supplied `store` always takes precedence over the configured default.
    """
    path = (
        default_store_path(masked)
        if store is None
        else Path(store)
    )

    if num_shards > 1:
        if shard is None:
            raise ValueError(
                "shard must be supplied when num_shards > 1"
            )

        path = path.parent / f"{path.name}_shard{shard}"

    return path
