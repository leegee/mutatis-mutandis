"""
lib/macberth.py
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Union

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from lib.eebo_logging import logger


MACBERTH_MODEL_PATH = Path("./lib/macberth-huggingface")
MACBERTH_MODEL_NAME = "emanjavacas/MacBERTh"


@dataclass
class MacberthModel:
    tokenizer: AutoTokenizer
    model: AutoModelForMaskedLM
    device: str

    def encode(self, **kwargs):
        """
        Run the encoder only.

        Used by Tier 1:
            - normal tokens
            - masked tokens

        Returns hidden states.
        """
        return self.model.base_model(**kwargs)

    def predict_masked(self, **kwargs):
        """
        Run the full masked-language model.

        Used by Tier 1.4 substitute vectors.

        Returns logits over vocabulary.
        """
        return self.model(**kwargs)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_macberth(
    local_files_only: bool = True,
) -> MacberthModel:
    """
    Loads MacBERTh with encoder + MLM head.

    The encoder is accessed through model.base_model.
    The MLM head is used by Tier 1.4.
    """

    logger.info("Loading MacBERTh model...")

    tokenizer = AutoTokenizer.from_pretrained(
        MACBERTH_MODEL_PATH,
        local_files_only=local_files_only,
    )

    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("Tokenizer must be fast")

    model = AutoModelForMaskedLM.from_pretrained(
        MACBERTH_MODEL_PATH,
        local_files_only=local_files_only,
    )

    device = get_device()

    model.to(device)
    model.eval()

    return MacberthModel(
        tokenizer=tokenizer,
        model=model,
        device=device,
    )


def normalize(v: np.ndarray) -> Optional[np.ndarray]:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return None
    return v / n


class MacBERThEmbedder:
    """
    BERTopic-compatible wrapper around MacberthModel.
    """

    def __init__(self, macberth: MacberthModel, pooling: str = "mean"):
        self.macberth = macberth
        self.pooling = pooling
        self.device = macberth.device

    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        **kwargs,
    ) -> np.ndarray:

        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        with torch.no_grad():
            for text in texts:

                encoded = self.macberth.tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )

                encoded = {
                    k: v.to(self.device)
                    for k, v in encoded.items()
                }

                # Encoder only, not MLM head
                outputs = self.macberth.encode(**encoded)

                if self.pooling == "cls":
                    emb = outputs.last_hidden_state[:, 0, :]

                elif self.pooling == "max":
                    emb = outputs.last_hidden_state.max(dim=1).values

                else:
                    attention_mask = encoded["attention_mask"]
                    emb = self._mean_pooling(
                        outputs.last_hidden_state,
                        attention_mask,
                    )

                all_embeddings.append(
                    emb.cpu().numpy().squeeze()
                )

        embeddings = np.array(all_embeddings)

        if convert_to_numpy:
            return embeddings

        return torch.tensor(embeddings)

    def encode_normalized(
        self,
        texts: Union[str, List[str]],
    ) -> np.ndarray:

        embeddings = self.encode(texts)

        return np.array([
            normalize(v)
            for v in embeddings
        ])

    @staticmethod
    def _mean_pooling(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        input_mask_expanded = (
            attention_mask
            .unsqueeze(-1)
            .expand(hidden_states.size())
            .float()
        )

        sum_embeddings = torch.sum(
            hidden_states * input_mask_expanded,
            dim=1,
        )

        sum_mask = torch.clamp(
            input_mask_expanded.sum(dim=1),
            min=1e-9,
        )

        return sum_embeddings / sum_mask


def get_macberth_embedder(
    pooling: str = "mean",
) -> MacBERThEmbedder:

    macberth_model = load_macberth()

    return MacBERThEmbedder(
        macberth_model,
        pooling=pooling,
    )
