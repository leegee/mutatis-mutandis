# macberth_pipe/pipeline.py

import time
from pathlib import Path
from typing import Optional
import logging

from macberth_pipe.tei import load_tei
from macberth_pipe.normalization import normalize
from macberth_pipe.embedding import load_model, embed_documents, Embeddings
from macberth_pipe.semantic import SemanticIndex
from macberth_pipe.clustering import cluster_embeddings
from macberth_pipe.metadata import load_doc_meta

logger = logging.getLogger(__name__)

DEFAULT_DEVICE = "cpu"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_AVERAGE_CHUNKS = False
DEFAULT_K_CLUSTERS = 5
DEFAULT_SQLITE_DB = Path("../../../../eebo-data/eebo-tcp_metadata.sqlite").resolve()
DEFAULT_FAISS_STORE = Path("../../../../faiss-cache/faiss-index")


def gather_files(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if path.is_dir():
        return sorted([p for p in path.iterdir() if p.suffix.lower() == ".xml"])
    else:
        return [path]

def run_pipeline(
    tei_path: Path,
    device: str = DEFAULT_DEVICE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    average_chunks: bool = DEFAULT_AVERAGE_CHUNKS,
    k_clusters: int = DEFAULT_K_CLUSTERS,
    sqlite_db: Path = DEFAULT_SQLITE_DB,
    faiss_store_dir: Path = DEFAULT_FAISS_STORE
):
    start_time = time.perf_counter()

    files = gather_files(tei_path)

    docs = [load_tei(p) for p in files]
    texts = [normalize(d.text) for d in docs]

    logger.debug("Loading metadata")
    doc_meta = load_doc_meta(files, sqlite_db)

    logger.debug("Loading model")
    model = load_model(device=device)

    logger.debug("Getting embeddings")
    if (faiss_store_dir / "embeddings.npy").exists():
        emb = load_embeddings_from_store(faiss_store_dir)
    else:
        emb = embed_documents(
            model,
            texts,
            device=device,
            chunk_size=chunk_size,
            average_chunks=average_chunks,
            doc_meta=doc_meta,
            store_dir=faiss_store_dir,
            append_to_store=True
        )

    logger.debug("Getting semantic index")
    index = SemanticIndex(emb, store_dir=faiss_store_dir)

    logger.debug("Running semantic search test")
    query = ["divine right of kings"]
    query_emb = embed_documents(
        model,
        query,
        device=device,
        average_chunks=average_chunks,
    ).vectors
    results = index.search(query_emb, top_k=5)

    logger.debug("Running clustering")
    if emb.vectors.shape[0] >= 2:
        safe_k = min(k_clusters, len(emb.ids))
        labels = cluster_embeddings(emb, k=safe_k)
    else:
        labels = []

    end_time = time.perf_counter()
    rv = {
        'emb': emb,
        'index': index,
        'results': results,
        'labels': labels,
        'time': end_time - start_time
    }

    logger.debug("RV", rv)

    return rv