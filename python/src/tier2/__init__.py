"""
Tier 1 Parquet
    │
    │ event_id → embedding
    ▼
year-local DiskANN
    ├── local
    ├── medium
    └── broad
         │
         ▼
Tier 2 analysis
    ├── lexical seed resolution
    ├── seed embedding lookup
    ├── 3 × DiskANN search
    ├── RRF fusion
    └── event/provenance materialisation
         │
         ▼
Tier 2 JSON / SQLite
"""
