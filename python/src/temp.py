import faiss, numpy as np
from lib.eebo_faiss import EeboFaissIndex
idx = EeboFaissIndex(dim=4, exact=True)
vecs = np.random.rand(3, 4).astype(np.float32)
ids = np.array([1,2,3], dtype=np.int64)
idx.add(vecs, ids)
print(idx.ntotal)
