import faiss
import numpy as np

base = faiss.IndexFlatIP(3)
idx = faiss.IndexIDMap2(base)

x = np.array([[1,0,0]], dtype="float32")
ids = np.array([123], dtype="int64")

idx.add_with_ids(x, ids)

print("before:")
print(type(idx).__name__)
print(type(idx.index).__name__)

faiss.write_index(idx, "test.faiss")

idx2 = faiss.read_index("test.faiss")

print("after:")
print(type(idx2).__name__)
print(type(idx2.index).__name__)

v = np.zeros(3, dtype="float32")
idx2.reconstruct(123, v)
print(v)