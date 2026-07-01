from lib.zarr_embedding_observation_store import ZarrEmbeddingObservationStore
import numpy as np

store = ZarrEmbeddingObservationStore("out/zarr/tier1", dim=768)
print("Total events in Zarr:", store.n_events)

eids = store.event_id[:]
unique_count = len(np.unique(eids))
print("Unique event_ids:", unique_count)
print("Duplicates exist:", unique_count < len(eids))

if unique_count < len(eids):
    from collections import Counter
    counts = Counter(eids)
    dups = {k: v for k, v in counts.items() if v > 1}
    print("Sample duplicates:", list(dups.items())[:5])