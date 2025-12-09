# run.py

import logging
from pathlib import Path
from macberth_pipe.pipeline import run_pipeline

TEI_PATH = Path("../../eebo-tei")  
FAISS_STORE_DIR = Path("faiss-cache/faiss-index")
DEVICE = "cpu"
CHUNK_SIZE = 512
AVERAGE_CHUNKS = True
K_CLUSTERS = 5
SQLITE_DB = Path("../../eebo-data/eebo-tcp_metadata.sqlite").resolve()

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

results = run_pipeline(
    tei_path=TEI_PATH,
    device=DEVICE,
    chunk_size=CHUNK_SIZE,
    average_chunks=AVERAGE_CHUNKS,
    k_clusters=K_CLUSTERS,
    sqlite_db=SQLITE_DB,
    faiss_store_dir=FAISS_STORE_DIR
)

logging.info(f"Pipeline completed in {results['time']:.2f} seconds")
logging.info(f"Top 5 search results:")
for r in results['results']:
    logging.info(f"[{r['doc_id']} chunk {r['chunk_idx']}] {r['text'][:120]}...")
