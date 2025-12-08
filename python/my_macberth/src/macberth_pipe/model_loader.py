# python/my_macberth/src/macberth_pipe/model_loader.py

from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def get_local_macberth_path() -> Path:
    """
    Resolve and validate the local MacBERTh model directory.
    Used by all modules to avoid duplication.
    """
    base_dir = Path(__file__).resolve().parent
    model_dir = (base_dir / "../../../lib/macberth-huggingface").resolve()

    required = ["config.json", "pytorch_model.bin", "vocab.txt"]
    missing = [f for f in required if not (model_dir / f).exists()]

    logger.debug(f"Resolved MacBERTh directory: {model_dir}")
    logger.debug(f"Directory contents: {list(model_dir.iterdir())}")

    if missing:
        raise FileNotFoundError(
            f"Missing required model files: {missing}\n"
            f"Expected directory: {model_dir}"
        )

    return model_dir
