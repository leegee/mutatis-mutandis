"""
zarr_observation_backend.py — Zarr concrete backend for the observation API

Stage 1: adapt the existing Zarr classes to the ObservationWriter /
ObservationStream / ObservationLookup protocols and register them under
the backend name "zarr".

Call sites can continue to import the historical class names; new code
should prefer the factory functions in observation_store_api:

    from observation_store_api import (
        open_observation_writer,
        open_observation_stream,
        open_observation_lookup,
    )
    writer = open_observation_writer("zarr", path, dim=768)
    stream = open_observation_stream("zarr", root)
    lookup = open_observation_lookup("zarr", root)

No behavioural change is introduced for the Zarr path. The adapters only
normalise constructor signatures and ensure protocol compliance so that a
future Parquet + DuckDB backend can be swapped in without touching Tier 1–3
call sites.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from observation_store_api import register_backend

# Historical implementations (kept under their original names for
# compatibility with existing imports in tier1 / tier1.5 / tier2 / tier3).
from lib.zarr_embedding_observation_store import ZarrEmbeddingObservationStore
from lib.zarr_event_stream import ZarrEventStream
from lib.zarr_event_lookup import ZarrEventLookup


# ---------------------------------------------------------------------------
# Thin constructor adapters
# ---------------------------------------------------------------------------
# The protocols / factories pass path-like + keyword arguments. Historical
# classes already accept these shapes; the adapters exist so we can evolve
# the factory signatures independently of the Zarr classes.

class ZarrObservationWriter(ZarrEmbeddingObservationStore):
    """ObservationWriter backed by a Zarr group on disk."""

    def __init__(self, path: str | Path, *, dim: int, **_kwargs: Any):
        super().__init__(path=str(path), dim=dim)


class ZarrObservationStream(ZarrEventStream):
    """ObservationStream over one or more Zarr event stores."""

    def __init__(self, root: str | Path, **_kwargs: Any):
        super().__init__(root=str(root))


class ZarrObservationLookup(ZarrEventLookup):
    """
    ObservationLookup over Tier-1 Zarr metadata.

    Embeddings remain lazy: call attach_index() with the per-year FAISS
    index dict before any embedding method is used. Metadata is loaded
    eagerly into compact NumPy columns (same as the historical class).
    """

    def __init__(self, root: str | Path, **_kwargs: Any):
        super().__init__(root=root)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_backend(
    "zarr",
    writer=ZarrObservationWriter,
    stream=ZarrObservationStream,
    lookup=ZarrObservationLookup,
)

# Re-export historical names so `from zarr_observation_backend import ...`
# remains a single import surface during the transition.
__all__ = [
    "ZarrObservationWriter",
    "ZarrObservationStream",
    "ZarrObservationLookup",
    # aliases of the original modules' public classes
    "ZarrEmbeddingObservationStore",
    "ZarrEventStream",
    "ZarrEventLookup",
]
