# retrieval/macberth_phrase_encoder2.py - no carriers

from __future__ import annotations

import numpy as np

from lib.macberth import load_macberth_onnx
from retrieval.phrase_encoder import PhraseQueryEncoder


class MacBertMeanPhraseEncoder(PhraseQueryEncoder):
    """
    Encode a phrase by mean-pooling the MacBERTh representations of the
    phrase's own subword tokens inside a caller-supplied carrier sentence.

    The carrier is supplied per call (not fixed at construction) so that
    callers can select a carrier matching the phrase's grammatical role
    and collocation pattern. This class only handles span extraction and
    pooling -- it has no opinion about which carrier is linguistically
    appropriate for a given term.
    """

    def __init__(self):
        self.macberth = load_macberth_onnx()
        self.tokenizer = self.macberth.tokenizer

    def encode(
        self,
        phrase: str,
        carrier: str,
    ) -> np.ndarray:

        if not phrase.strip():
            raise ValueError("Phrase must not be empty")

        if "{}" not in carrier:
            raise ValueError(f"Carrier must contain a {{}} placeholder: {carrier!r}")

        sentence = carrier.format(phrase)

        if sentence.count(phrase) > 1:
            raise ValueError(
                f"Phrase {phrase!r} occurs more than once in carrier "
                f"{carrier!r}; span extraction would be ambiguous."
            )

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
