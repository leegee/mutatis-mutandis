from lib.zarr_event_stream import ZarrEventStream
from lib.eebo_faiss import EeboFaissIndex
from lib.eebo_config import ZARR_PATH

stream = ZarrEventStream(str(ZARR_PATH))

zarr_ids = set()

for _, _, _, ids, years in stream.iter_multi_scale_embeddings():
    zarr_ids.update(map(int, ids))

print("Zarr:", len(zarr_ids))


faiss_ids = set()

indices = EeboFaissIndex.load_all()

for year, scales in indices.items():
    for scale, idx in scales.items():
        faiss_ids.update(idx.ids())

print("FAISS:", len(faiss_ids))

print("Missing in FAISS:", len(zarr_ids - faiss_ids))
print("Extra in FAISS:", len(faiss_ids - zarr_ids))
