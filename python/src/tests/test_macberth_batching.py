#!/usr/bin/env python

from __future__ import annotations

import time

import numpy as np

from lib.macberth import (
    MacBERThEmbedder,
    load_macberth_onnx,
)


TEXTS = [
    "The king did graciously receive the petition of his loyal subjects.",
    "It is reported that the plague hath spread through the parish.",
    "Concerning the nature of witchcraft, many learned men have written.",
    "The merchant sold his wares at the market on Thursday last.",
    "Let all men know that this covenant is binding before God.",
    "The soldiers marched upon the town at break of day.",
] * 8


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.dot(a, b)
        / (np.linalg.norm(a) * np.linalg.norm(b))
    )


def main() -> None:
    print("Loading FP32 ONNX MacBERTh...")
    model = load_macberth_onnx(quantize=False)
    embedder = MacBERThEmbedder(model, pooling="mean")

    print(f"Testing {len(TEXTS)} texts...\n")

    # Warm up ONNX Runtime so model/session initialisation is not included
    # in either timing measurement.
    embedder.encode(TEXTS[:32])

    # The old implementation is reproduced here explicitly so we have a
    # genuine batch-1 baseline rather than relying on an older checkout.
    def encode_batch_1(texts):
        embeddings = []

        import torch

        with torch.no_grad():
            for text in texts:
                encoded = embedder.macberth.tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )

                outputs = embedder.macberth.encode(**{
                    k: v.to(embedder.device)
                    for k, v in encoded.items()
                })

                attention_mask = encoded["attention_mask"]
                emb = embedder._mean_pooling(
                    outputs.last_hidden_state,
                    attention_mask,
                )

                embeddings.append(emb.cpu().numpy().squeeze())

        return np.array(embeddings)

    t0 = time.perf_counter()
    baseline = encode_batch_1(TEXTS)
    baseline_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    batched = embedder.encode(TEXTS)
    batched_time = time.perf_counter() - t0

    similarities = [
        cosine(a, b)
        for a, b in zip(baseline, batched)
    ]

    print(f"Batch 1:  {baseline_time:.2f}s")
    print(f"Batch 32: {batched_time:.2f}s")
    print()

    print(
        f"Batch 1:  {len(TEXTS) / baseline_time:.2f} texts/sec"
    )
    print(
        f"Batch 32: {len(TEXTS) / batched_time:.2f} texts/sec"
    )
    print(
        f"Speedup:   {baseline_time / batched_time:.2f}x"
    )
    print()

    print(
        f"Vector cosine similarity: "
        f"min={min(similarities):.8f} "
        f"mean={np.mean(similarities):.8f} "
        f"max={max(similarities):.8f}"
    )


if __name__ == "__main__":
    main()
