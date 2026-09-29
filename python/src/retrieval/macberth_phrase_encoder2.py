# retrieval/macberth_phrase_encoder2.py

from __future__ import annotations

import numpy as np

from lib.macberth import load_macberth_onnx
from retrieval.phrase_encoder import PhraseQueryEncoder

class MacBertMeanPhraseEncoder(PhraseQueryEncoder):
    """
    Encode text using mean-pooled MacBERTh representations.

    Public APIs
    -----------
    encode(phrase, carrier)
        Backwards-compatible API. Inserts `phrase` into `carrier` and pools
        only the MacBERTh subword tokens belonging to the phrase.

    encode_text(text)
        Encodes a complete text directly and mean-pools all of its
        non-special-token MacBERTh representations.

    `_encode()` contains the common MacBERTh encoding and pooling machinery.
    """

    def __init__(self):
        self.macberth = load_macberth_onnx()
        self.tokenizer = self.macberth.tokenizer

    def encode(
        self,
        phrase: str,
        carrier: str,
    ) -> np.ndarray:
        """
        Backwards-compatible phrase-in-carrier API.
        """

        if not phrase.strip():
            raise ValueError("Phrase must not be empty")

        if "{}" not in carrier:
            raise ValueError(
                f"Carrier must contain a {{}} placeholder: {carrier!r}"
            )

        sentence = carrier.format(phrase)

        if sentence.count(phrase) > 1:
            raise ValueError(
                f"Phrase {phrase!r} occurs more than once in carrier "
                f"{carrier!r}; span extraction would be ambiguous."
            )

        phrase_start = sentence.index(phrase)
        phrase_end = phrase_start + len(phrase)

        return self._encode(
            sentence,
            span=(phrase_start, phrase_end),
        )

    def encode_text(
        self,
        text: str,
    ) -> np.ndarray:
        """
        Encode a complete piece of text directly, without a carrier.

        All non-special MacBERTh subword tokens in `text` are mean-pooled.
        """

        if not text.strip():
            raise ValueError("Text must not be empty")

        return self._encode(text)

    def _encode(
        self,
        text: str,
        span: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """
        Common MacBERTh encoding and mean-pooling implementation.

        If `span` is supplied, only tokens wholly contained within the span
        are pooled. Otherwise all non-special tokens are pooled.
        """

        encoded = self.tokenizer(
            text,
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

        vectors = []

        for vector, offset in zip(hidden, offsets):
            start, end = (
                int(offset[0]),
                int(offset[1]),
            )

            # Skip special tokens and other zero-width offsets.
            if start == end:
                continue

            if span is not None:
                span_start, span_end = span

                if not (
                    start >= span_start
                    and end <= span_end
                ):
                    continue

            vectors.append(vector)

        if not vectors:
            if span is not None:
                raise ValueError(
                    f"No MacBERTh tokens found for phrase span."
                )

            raise ValueError(
                f"No MacBERTh tokens found for text: {text!r}"
            )

        vector = (
            np.stack([
                item.detach().cpu().numpy()
                for item in vectors
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
