#!/usr/bin/env python
"""
tier2_concept_events.py

Thin backward-compatible facade over the split Tier 2 modules:

    tier2_analyse.py          -- pure retrieval / neighbourhood analysis
    tier2_create_populate.py  -- schema + SQLite persistence

Existing callers that imported analysis or persistence helpers from this
module (run_tier2_core, run_tier2_service, write_concept, SCHEMA, etc.)
keep working unchanged. The CLI here is identical to the original
combined "analyse + populate" behaviour.
"""

from __future__ import annotations

from lib.corpus_logging import logger

# Re-exported for backward compatibility.
from tier2.tier2_analyse import (  # noqa: F401
    K,
    RRF_K,
    OVERSAMPLE,
    analyse_concept,
    run_tier2_core,
    build_resources,
)

from tier2.tier2_create_populate import (  # noqa: F401
    SCHEMA,
    sqlite_connection,
    initialise_database,
    ensure_documents,
    ensure_events,
    enrich_documents,
    delete_concept,
    write_concept,
    run_tier2_populate_service,
    run_tier2_service,
    main,
)


if __name__ == "__main__":
    main()
