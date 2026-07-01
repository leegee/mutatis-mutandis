#!/usr/bin/env python
"""
backfill_zarr_event_type.py - one-time backfill for new Zarr columns: event_type + span_*
"""

import numpy as np
import zarr
from pathlib import Path

from lib.zarr_embedding_observation_store import ZarrEmbeddingObservationStore

ZARR_PATH = Path("out/zarr/tier1")   # Adjust if your path is different

def backfill():
    print("Opening Zarr store...")
    store = ZarrEmbeddingObservationStore(str(ZARR_PATH), dim=768)

    g = store.root["events"]
    n = g["event_id"].shape[0]

    print(f"Total events: {n:,}")

    # Create / resize arrays if needed
    for name, dtype, default_value in [
        ("event_type", "U32", "window"),
        ("span_start_token_idx", "int64", None),
        ("span_end_token_idx", "int64", None),
    ]:
        if name not in g or g[name].shape[0] == 0:
            print(f"Creating/backfilling {name}...")
            if name not in g:
                g.create_dataset(
                    name,
                    shape=(n,),
                    chunks=(4096,),
                    dtype=dtype,
                )

            arr = g[name]
            if arr.shape[0] < n:
                arr.resize(n)

            if default_value is not None:
                arr[:] = default_value
            else:
                # span = token_idx
                arr[:] = g["token_idx"][:]

            print(f"  → {name} filled with {n:,} entries")
        else:
            print(f"  {name} already has {g[name].shape[0]:,} entries")

    print("\nBackfill complete!")
    print("event_type sample:", g["event_type"][:5])
    print("span_start sample:", g["span_start_token_idx"][:5])

if __name__ == "__main__":
    backfill()
