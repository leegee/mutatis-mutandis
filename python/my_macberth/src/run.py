import time
import logging
from glob import glob

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

start_time = time.perf_counter()

logging.info("Loading and normalizing TEI files...")
docs = [load_tei(p) for p in FILES]
texts = [normalize(d.text) for d in docs]

logging.info("Loading document metadata...")
doc_meta = load_doc_meta(FILES, SQLITE_DB)

logging.info("Loading MacBERTh model...")
model = load_model(device=DEVICE)

logging.info("Embedding documents...")
emb = embed_documents(
    model,
    texts,
    device=DEVICE,
    chunk_size=CHUNK_SIZE,
    average_chunks=AVERAGE_CHUNKS,
    doc_meta=doc_meta
)

logging.info("Performing semantic search for query: 'divine right of kings'")
query = ["divine right of kings"]
query_emb = embed_documents(
    model,
    query,
    device=DEVICE,
    average_chunks=True
).vectors

index = SemanticIndex(emb)
results = search(index, query_emb, top_k=5)

logging.info("Top snippet matches:")
for r in results:
    meta = next((m for m in emb.metas if m.doc_id == r['doc_id']), None)
    if meta:
        logging.info(
            f"[{r['doc_id']} chunk {r['chunk_idx']}] {r['text'][:120]}...\n"
            f"Title: {meta.title}, Author: {meta.author}, Year: {meta.year}\n"
            f"Permalink: {meta.permalink}"
        )
    else:
        logging.info(f"[{r['doc_id']} chunk {r['chunk_idx']}] {r['text'][:120]}...")

# CLUSTERING
if emb.vectors.shape[0] >= 2:
    safe_k = min(K_CLUSTERS, len(emb.ids))
    labels = cluster_embeddings(emb, k=safe_k)
    logging.info("Cluster labels:")
    for doc_id, label in zip(emb.ids, labels):
        logging.info(f"{doc_id}: cluster {label}")

end_time = time.perf_counter()
logging.info(f"Total processing time: {end_time - start_time:.2f} seconds")
