from glob import glob
import logging

from macberth_pipe.tei import load_tei
from macberth_pipe.normalization import normalize
from macberth_pipe.embedding import load_model, embed_documents, Embeddings
from macberth_pipe.semantic import SemanticIndex, search
from macberth_pipe.clustering import cluster_embeddings
from macberth_pipe.metadata import load_doc_meta

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

DEVICE = "cpu"

# CONFIG
FILES = glob("../../eebo_tei/*.xml")
CHUNK_SIZE = 512
AVERAGE_CHUNKS = True
K_CLUSTERS = 5
SQLITE_DB = "S:/src/pamphlets/eebo-data/eebo-tcp_metadata.sqlite"

# LOAD & NORMALIZE TEI
docs = [load_tei(p) for p in FILES]
texts = [normalize(d.text) for d in docs]

# LOAD DOC METADATA
doc_meta = load_doc_meta(FILES, SQLITE_DB)

# LOAD MODEL
model = load_model(device=DEVICE)

# EMBEDDING
emb = embed_documents(
    model,
    texts,
    device=DEVICE,
    chunk_size=CHUNK_SIZE,
    average_chunks=AVERAGE_CHUNKS,
    doc_meta=doc_meta
)

# SEMANTIC SEARCH
query = ["divine right of kings"]
query_emb = embed_documents(
    model,
    query,
    device=DEVICE,
    average_chunks=True
).vectors

index = SemanticIndex(emb)
results = search(index, query_emb, top_k=5)

print("\nTop snippet matches:\n")
for r in results:
    print(f"[{r['doc_id']} chunk {r['chunk_idx']}] {r['text'][:120]}...")
    print(f"Title: {r.get('title','')}, Author: {r.get('author','')}, Year: {r.get('year','')}")
    print(f"Permalink: {r.get('permalink','')}\n")

# CLUSTERING (safe)
if emb.vectors.shape[0] >= 2:
    safe_k = min(K_CLUSTERS, len(emb.ids))
    labels = cluster_embeddings(emb, k=safe_k)
    print("Cluster labels:")
    for doc_id, label in zip(emb.ids, labels):
        print(f"{doc_id}: cluster {label}")
