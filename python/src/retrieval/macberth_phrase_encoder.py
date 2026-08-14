from __future__ import annotations

import numpy as np

from lib.macberth import load_macberth_onnx
from retrieval.phrase_encoder import PhraseQueryEncoder

DEFAULT_CARRIER = "This refers to {}."


class MacBertMeanPhraseEncoder(PhraseQueryEncoder):
    """
    Encode a phrase by mean-pooling the MacBERTh representations of the
    phrase's own subword tokens inside a carrier sentence.

    The carrier supplies context for an unseen phrase; only the hidden
    states whose character offsets fall inside the phrase span contribute
    to the returned vector.

    MacBERTh is loaded internally using the project's default backend.
    """

    def __init__(
        self,
        carrier: str = DEFAULT_CARRIER,
    ):
        self.macberth = load_macberth_onnx()
        self.tokenizer = self.macberth.tokenizer
        self.carrier = carrier

    def encode(
        self,
        phrase: str,
    ) -> np.ndarray:

        if not phrase.strip():
            raise ValueError("Phrase must not be empty")

        sentence = self.carrier.format(phrase)

        phrase_start = sentence.index(phrase)
        phrase_end = phrase_start + len(phrase)

        encoded = self.tokenizer(
            sentence,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        offsets = encoded.pop("offset_mapping")[0]

        encoded = {
            key: value.to(self.macberth.device)
            for key, value in encoded.items()
        }

        outputs = self.macberth.encode(**encoded)
        hidden = outputs.last_hidden_state[0]

        phrase_vectors = []

        for vector, offset in zip(hidden, offsets):
            start, end = (
                int(offset[0]),
                int(offset[1]),
            )

            # Special tokens have (0, 0) offsets and therefore cannot
            # belong to the phrase span.
            if start == end:
                continue

            if start >= phrase_start and end <= phrase_end:
                phrase_vectors.append(vector)

        if not phrase_vectors:
            raise ValueError(
                f"No MacBERTh tokens found for phrase span: {phrase!r}"
            )

        vector = (
            np.stack([
                item.detach().cpu().numpy()
                for item in phrase_vectors
            ])
            .mean(axis=0)
            .astype(np.float32)
        )

        norm = np.linalg.norm(vector)

        if norm < 1e-12:
            raise ValueError(
                "Phrase encoder produced zero vector."
            )

        return (vector / norm).astype(np.float32)
