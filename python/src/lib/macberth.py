"""
lib/macberth.py

If you see an error like the below, make sure you are running from the `python/` dir:

    OSError: Repo id must use alphanumeric chars, '-', '_' or '.'. The name cannot start or end with '-' or '.' and the maximum length is 96: 'lib\macberth-huggingface'.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from lib.eebo_logging import logger
from lib.eebo_config import MACBERTH_MODEL_PATH


@dataclass
class MacberthModel:
    tokenizer: AutoTokenizer
    model: AutoModel
    device: str


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_macberth(local_files_only: bool = True) -> MacberthModel:
    """
    Loads tokenizer + model and moves model to correct device.
    """
    logger.info("Loading Macberth model...")

    tokenizer = AutoTokenizer.from_pretrained(
        MACBERTH_MODEL_PATH,
        local_files_only=local_files_only,
    )

    model = AutoModel.from_pretrained(
        MACBERTH_MODEL_PATH,
        local_files_only=local_files_only,
    )

    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("Tokenizer must be fast")

    device = get_device()
    model.to(device)
    model.eval()

    return MacberthModel(tokenizer=tokenizer, model=model, device=device)


def normalize(v: np.ndarray) -> Optional[np.ndarray]:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return None
    return v / n
