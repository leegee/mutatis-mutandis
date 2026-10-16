from pathlib import Path
import zarr

from lib.corpus_config import CONCEPT_ZARR_PATH

ARRAYS = [
    "event_id",
    "concept_id",
    "emb_local",
    "emb_medium",
    "emb_broad",
    "vector_id",
    "token_idx",
    "token",
    "corpus",
    "doc_id",
    "pub_year",
    "window_id",
    "window_token_pos",
]

for i in range(3):
    path = CONCEPT_ZARR_PATH.parent / f"{CONCEPT_ZARR_PATH.name}_shard{i}"

    print(f"\n--- {path} ---")

    if not path.exists():
        print("MISSING")
        continue

    root = zarr.open_group(str(path), mode="r")

    if "events" not in root:
        print("No events group")
        continue

    events = root["events"]
    lengths = {
        name: events[name].shape[0]
        for name in ARRAYS
        if name in events
    }

    for name, length in lengths.items():
        print(f"{name:20s} {length:12,d}")

    print("Array lengths consistent:", len(set(lengths.values())) == 1)
    print("_COMPLETE marker:", (path / "_COMPLETE").exists())