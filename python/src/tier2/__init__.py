"""
tier2 package

Tier 2 semantic neighbourhood construction.

Modules:

- analysis   : pure neighbourhood retrieval (run_tier2_core, analyse_concept)
- persistence: SQLite schema and materialisation of results
- resources  : FAISS index loading
- orchestrator: service + CLI entry points that wire the three together
"""

from tier2.analysis import (
    K,
    RRF_K,
    OVERSAMPLE,
    analyse_concept,
    run_tier2_core,
)
from tier2.persistence import (
    SCHEMA,
    sqlite_connection,
    initialise_database,
    ensure_documents,
    ensure_events,
    delete_concept,
    write_concept,
    enrich_documents,
)
from tier2.resources import load_indices
from tier2.orchestrator import (
    service,
    main,
)

__all__ = [
    "K",
    "RRF_K",
    "OVERSAMPLE",
    "analyse_concept",
    "run_tier2_core",
    "SCHEMA",
    "sqlite_connection",
    "initialise_database",
    "ensure_documents",
    "ensure_events",
    "delete_concept",
    "write_concept",
    "enrich_documents",
    "load_indices",
    "service",
    "main",
]
