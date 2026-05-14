"""
lib/macberth.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from lib.eebo_logging import logger
from lib.eebo_config import EEBO_MODEL_NAME


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
        EEBO_MODEL_NAME,
        local_files_only=local_files_only,
    )

    model = AutoModel.from_pretrained(
        EEBO_MODEL_NAME,
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
