#!/usr/bin/env python
"""
tier2_concept_events.py

Compatibility entry point. All behaviour lives in the tier2 package.

    Tier 1 Zarr observations
            |
        Parquet store
            |
    yearly DiskANN retrieval geometry
            |
    SQLite semantic neighbourhood store

Tier 2 does not define concepts. It records the empirical neighbourhood
around supplied lexical seeds and preserves provenance back to corpus
events.

Important invariants:

- event_id is the atomic corpus occurrence.
- RRF scores are ranking scores, not distances.
- Retrieval is publication-year scoped.
- Lexical forms are query provenance, not semantic membership.
"""

from tier2.orchestrator import main

if __name__ == "__main__":
    main()
