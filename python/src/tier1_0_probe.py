from lib.zarr_event_stream import ZarrEventStream
from lib.eebo_config import ZARR_ROOT

stream = ZarrEventStream(str(ZARR_ROOT / "tier1"))

count = 0

for vecs, obs_ids in stream.iter_embeddings(batch_size=1024):
    print(vecs.shape, obs_ids.shape)
    count += len(obs_ids)
    if count > 1000:
        break

print("TOTAL OBS:", count)