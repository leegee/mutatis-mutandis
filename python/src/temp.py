# src/temp.py  (or run interactively)
from lib.eebo_config import ZARR_PATH
from lib.zarr_embedding_observation_store import ZarrEmbeddingObservationStore
from pathlib import Path
import zarr

print("=== Zarr Store Diagnostics ===")
root = zarr.open_group(ZARR_PATH, mode="a")
events = root["events"]

print("Existing arrays:")
for name in sorted(events.array_keys()):
    arr = events[name]
    print(f"  {name:25} shape={arr.shape} dtype={arr.dtype}")

print("\nFirst 5 event_types:", events.get("event_type", None))
print("First 5 span_start:  ", events.get("span_start_token_idx", None))

# lookup = ZarrEventLookup(ZARR_ROOT / "tier1")
# print("Event types distribution:", np.unique(lookup.event_type, return_counts=True))
