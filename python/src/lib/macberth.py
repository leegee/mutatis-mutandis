"""
lib/macberth.py

"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Tuple, Optional, List, Union
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
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




class MacBERThEmbedder:
    """
    BERTopic-compatible wrapper around your MacberthModel.
    Implements the .encode() interface expected by BERTopic / sentence-transformers.
    """

    def __init__(self, macberth: MacberthModel, pooling: str = "mean"):
        self.macberth = macberth
        self.pooling = pooling  # "mean", "cls", or "max"
        self.device = macberth.device

    def encode(self,
               texts: Union[str, List[str]],
               show_progress_bar: bool = False,
               convert_to_numpy: bool = True,
               **kwargs) -> np.ndarray:
        """
        Main method used by BERTopic.
        Returns document embeddings (shape: n_docs × hidden_size).
        """
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        with torch.no_grad():
            for text in texts:
                # Tokenize
                encoded = self.macberth.tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                # Forward pass
                outputs = self.macberth.model(**encoded)

                # Pooling
                if self.pooling == "cls":
                    emb = outputs.last_hidden_state[:, 0, :]          # CLS token
                elif self.pooling == "max":
                    emb = outputs.last_hidden_state.max(dim=1).values
                else:  # default: mean pooling
                    attention_mask = encoded["attention_mask"]
                    emb = self._mean_pooling(outputs.last_hidden_state, attention_mask)

                all_embeddings.append(emb.cpu().numpy().squeeze())

        embeddings = np.array(all_embeddings)

        if convert_to_numpy:
            return embeddings
        return torch.tensor(embeddings)

    @staticmethod
    def _mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling with attention mask (standard for BERT)."""
        token_embeddings = hidden_states
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask


def get_macberth_embedder(pooling: str = "mean") -> MacBERThEmbedder:
    """Convenience loader."""
    macberth_model = load_macberth()
    return MacBERThEmbedder(macberth_model, pooling=pooling)
