# lib/eebo_embeddings.py
from pathlib import Path
import fasttext
import numpy as np
from lib.eebo_logging import logger

def generate_embeddings_per_model(model_path: Path) -> dict[str, np.ndarray]:
    """Load a fastText slice model and generate embeddings for all words in its vocabulary."""
    logger.info(f"Loading model {model_path}")
    model = fasttext.load_model(str(model_path))

    tokens = model.get_words()
    embeddings: dict[str, np.ndarray] = {}
    for tok in tokens:
        embeddings[str(tok)] = model.get_word_vector(tok).astype(np.float32)
    logger.info(f"Generated embeddings for {len(tokens)} tokens in {model_path.name}")
    return embeddings

