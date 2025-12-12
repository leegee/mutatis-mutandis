# python/my_macberth/src/macberth_pipe/macberth_model.py

from pathlib import Path
from typing import List, Union
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from .model_loader import get_local_macberth_path


class MacBERThModel:
    def __init__(self, model_path=None, device="cpu"):
        self.device = device

        # Use shared loader if not provided
        if model_path is None:
            model_path = get_local_macberth_path()
        else:
            model_path = Path(model_path).resolve()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True
        )
        self.model.to(device)
        self.model.eval()

    def split_into_chunks(self, text: str, chunk_size: int = 512) -> list[str]:
        """Split a text string into roughly equal chunks of given size."""
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    def _encode_text(self, text: str) -> np.ndarray:
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use the [CLS] token representation (or mean pooling)
        last_hidden_state = outputs.last_hidden_state  # (1, seq_len, hidden_dim)
        # Mean pooling
        vec = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # (hidden_dim,)
        return vec

    def embed_text(self, texts: Union[str, List[str]], batch_size: int = 8) -> List[np.ndarray]:
        """
        Embed a single string or a list of strings.
        Returns a list of numpy arrays, one per input text.
        Supports batching for speed.
        """
        single_input = False
        if isinstance(texts, str):
            texts = [texts]
            single_input = True

        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Mean pooling over sequence
            last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
            batch_vecs = last_hidden.mean(dim=1).cpu().numpy()  # (batch, hidden_dim)

            vectors.extend(batch_vecs)

        return vectors if not single_input else [vectors[0]]
