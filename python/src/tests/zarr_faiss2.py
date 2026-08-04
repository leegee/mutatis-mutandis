
from lib.zarr_event_stream import ZarrEventStream
from lib.eebo_config import ZARR_PATH
import numpy as np

stream = ZarrEventStream(str(ZARR_PATH))

counts = {}

for _, _, _, ids, years in stream.iter_multi_scale_embeddings():
    for y in years:
        y = int(y)
        counts[y] = counts.get(y, 0) + 1

print(sorted(counts.items()))
print("total", sum(counts.values()))