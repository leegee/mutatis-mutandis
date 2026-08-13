from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import glob
import pyarrow.dataset as ds


PARQUET_ROOT = Path(r"G:\corpus-out-parq\events")
YEAR = 1625
EMBEDDING_COLUMN = "emb_local"


paths = glob.glob(
    "G:/corpus-out-parq/events/tier1_shard*/year=1625/*.parquet"
)

print(f"Parquet files: {len(paths)}")
for path in paths:
    print(path)

dataset = ds.dataset(paths, format="parquet")

table = dataset.to_table(
    columns=["event_id", EMBEDDING_COLUMN],
)

event_ids = np.asarray(
    table["event_id"].to_numpy(),
    dtype=np.int64,
)

vectors = np.stack(
    table[EMBEDDING_COLUMN].to_pylist()
).astype(np.float32, copy=False)


print("events:", len(event_ids))
print("vectors:", vectors.shape)
print("dtype:", vectors.dtype)
print("event IDs unique:", len(np.unique(event_ids)) == len(event_ids))
print("finite:", np.isfinite(vectors).all())
print("norm range:", np.linalg.norm(vectors, axis=1).min(),
      np.linalg.norm(vectors, axis=1).max())
