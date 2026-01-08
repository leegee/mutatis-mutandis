#!/usr/bin/env python

import logging
import logging.config
from pathlib import Path
from macberth_pipe.pipeline import run_pipeline

TEI_PATH = Path("../../eebo-tei")
FAISS_STORE_DIR = Path("../../../faiss-cache/faiss-index")
DEVICE = "cpu"
CHUNK_SIZE = 512
AVERAGE_CHUNKS = False
K_CLUSTERS = 5
SQLITE_DB = Path("../../eebo-data/eebo-tcp_metadata.sqlite").resolve()

logging.config.dictConfig({
    "version": 1,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "level": "DEBUG",
            "formatter": "std",
        }
    },
    "formatters": {
        "std": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console"]
    },
    "loggers": {
        "macberth_pipe": {
            "level": "DEBUG",
            "handlers": ["console"],
            "propagate": False
        }
    },
    "disable_existing_loggers": False
})

logger = logging.getLogger(__name__)
logger.debug("Configured logger.")

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
logging.info("Top 5 search results:")
for r in results['results']:
    logging.info(
        f"[{r['doc_id']} chunk {r['chunk_idx']}] {r['text'][:120]}..."
    )
