import zarr
import numpy as np
from numcodecs import Blosc
from lib.corpus_db import get_connection
from lib.corpus_config import ZARR_PATH

root = zarr.open_group(str(ZARR_PATH), mode="a", zarr_version=2)
events = root["events"]

doc_ids = events["doc_id"][:]
token_idxs = events["token_idx"][:]

conn = get_connection()
cur = conn.cursor()
cur.execute("SELECT corpus, doc_id, token_idx FROM pamphlet_tokens")

lookup = {
    (doc_id, token_idx): corpus
    for corpus, doc_id, token_idx in cur
}

corpora = np.empty(len(doc_ids), dtype="U32")
missing = 0

for i, (doc_id, token_idx) in enumerate(zip(doc_ids, token_idxs)):
    key = (str(doc_id), int(token_idx))
    corpus = lookup.get(key)
    if corpus is None:
        missing += 1
        corpora[i] = ""
    else:
        corpora[i] = corpus

print("missing", missing)

if "corpus" in events:
    del events["corpus"]

compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

events.create_dataset(
    "corpus",
    data=corpora,
    shape=corpora.shape,
    dtype="U32",
    chunks=(4096,),
    compressor=compressor,
)

print("done")
