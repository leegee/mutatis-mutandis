#!/usr/bin/env python
# extract_neighbourhoods_faiss.py
"""
Phase 5: Neighbourhood Extraction (FAISS-enhanced)

- Uses canonical FAISS IndexIDMap to fetch nearest neighbours per slice
- Falls back to fastText if query vector not in FAISS
- Writes results to `neighbourhoods` table
- Handles ON CONFLICT with DO UPDATE
- Logs structured output via eebo_logging.logger
"""

from pathlib import Path
import numpy as np
import faiss
from tqdm import tqdm
from lib import eebo_db, eebo_config as config
from lib.eebo_logging import logger
import fasttext

def main():
    queries = list(config.KEYWORDS_TO_NORMALISE.keys())
    stopwords_path = Path(config.STOPWORD_FILE)
    stopwords = {line.strip().lower() for line in stopwords_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    logger.info("Loaded %d queries and %d stopwords", len(queries), len(stopwords))

    # Load canonical vectors
    canonical_vectors = {}
    with eebo_db.get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT canonical, vector FROM canonical_centroids")
        for canonical, blob in cur.fetchall():
            canonical_vectors[canonical] = np.array(blob, dtype=np.float32)
    logger.info("Loaded %d canonical vectors", len(canonical_vectors))

    # Load FAISS canonical index
    faiss_index_path = config.FAISS_CANONICAL_INDEX_PATH
    if not faiss_index_path.exists():
        logger.error("FAISS canonical index not found: %s", faiss_index_path)
        return
    faiss_index = faiss.read_index(str(faiss_index_path))
    logger.info("Loaded FAISS index from %s", faiss_index_path)

    # Build ID → token map from canonical vectors
    # Assumes faiss_index is an IndexIDMap with .id_map containing token IDs
    id_to_token = {i: token for i, token in enumerate(canonical_vectors.keys())}

    # Iterate over slice fastText models
    slice_models = sorted(Path(config.FASTTEXT_SLICE_MODEL_DIR).glob("*.bin"))
    if not slice_models:
        logger.error("No slice models found in %s", config.FASTTEXT_SLICE_MODEL_DIR)
        return

    for model_path in slice_models:
        slice_name = model_path.stem
        try:
            start_year, end_year = map(int, slice_name.split("-"))
        except ValueError:
            logger.warning("Skipping non-year slice '%s'", slice_name)
            continue

        logger.info("Processing slice %s (%d-%d)", slice_name, start_year, end_year)

        # Load fastText slice model for fallback
        slice_model = fasttext.load_model(str(model_path))
        slice_vocab = set(slice_model.get_words())

        with eebo_db.get_connection() as conn:
            cur = conn.cursor()

            for query in tqdm(queries, desc=f"{slice_name} queries"):
                if query not in canonical_vectors:
                    logger.warning("Query '%s' missing canonical vector, skipping FAISS", query)
                    continue

                query_vec = canonical_vectors[query].reshape(1, -1)

                # FAISS lookup
                try:
                    D, Idx = faiss_index.search(query_vec, config.TOP_K * 2)
                except Exception as e:
                    logger.warning("FAISS search failed for '%s': %s", query, e)
                    continue

                neighbours = []
                for dist, idx in zip(D[0], Idx[0], strict=True):
                    token = id_to_token.get(idx)
                    if not token:
                        logger.warning("FAISS returned unknown idx=%d for query '%s'", idx, query)
                        continue
                    if token == query or token in stopwords:
                        continue
                    neighbours.append((float(dist), token))
                    if len(neighbours) >= config.TOP_K:
                        break

                # Fallback to fastText if FAISS returned too few neighbours
                if len(neighbours) < config.TOP_K and query in slice_vocab:
                    ft_neighbors = slice_model.get_nearest_neighbors(query, config.TOP_K * 2)
                    for cosine, token in ft_neighbors:
                        token = token.lower()
                        if token == query or token in stopwords:
                            continue
                        if token in canonical_vectors:  # prefer canonical hits
                            continue
                        neighbours.append((float(cosine), token))
                        if len(neighbours) >= config.TOP_K:
                            break

                # Insert neighbours into DB
                for rank, (cosine, neighbour) in enumerate(neighbours, start=1):
                    cur.execute("""
                        INSERT INTO neighbourhoods
                        (slice_start, slice_end, query, neighbour, rank, cosine)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (slice_start, slice_end, query, rank)
                        DO UPDATE SET
                            neighbour = EXCLUDED.neighbour,
                            cosine    = EXCLUDED.cosine;
                    """, (start_year, end_year, query, neighbour, rank, cosine))

            conn.commit()

    logger.info("Neighbourhood extraction complete (FAISS-enhanced)")

if __name__ == "__main__":
    main()
