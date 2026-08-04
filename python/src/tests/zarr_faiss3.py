from lib.eebo_config import ZARR_PATH, faiss_index_paths
from lib.zarr_event_stream import ZarrEventStream
from lib.eebo_faiss import EeboFaissIndex
import numpy as np

stream = ZarrEventStream(str(ZARR_PATH))

count = 0
ids = set()

for _, _, _, eids, years in stream.iter_multi_scale_embeddings(
    batch_size=8192,
    year_filter={1734}
):
    mask = years == 1734
    count += mask.sum()
    ids.update(map(int, eids[mask]))

print("Zarr 1734:", count)

for scale, path in faiss_index_paths(False, year=1734).items():
    idx = EeboFaissIndex.load(path)
    print(scale, idx.ntotal, len(ids - idx.ids()))