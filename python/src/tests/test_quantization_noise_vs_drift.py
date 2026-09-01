#!/usr/bin/env python
"""
test_quantization_noise.py

No-DB diagnostic: embed a handful of strings with both the fp32 and int8
ONNX MacBERTh models and report how much quantization moves the embedding.

This does NOT measure real diachronic (across-time) semantic drift --
that needs actual corpus occurrences across years, which requires your
DB or corpus files. What this DOES give you is a same-input-different-model
comparison (the true quantization noise floor), plus, as a rough scale
reference, the fp32-only distance between clearly different sentences/
contexts. If quantization noise is much smaller than that reference gap,
it's a good sign int8 isn't destroying signal -- but it's not proof for
the specific diachronic-drift use case, since real drift is usually far
subtler than "these are different sentences."

Usage:
    python test_quantization_noise.py
    python test_quantization_noise.py --text "Custom sentence one." --text "Custom sentence two."
"""

from __future__ import annotations

import argparse
import numpy as np

from lib.corpus_logging import logger
from lib.macberth import load_macberth_onnx, MacBERThEmbedder


DEFAULT_TEXTS = [
    "The king did graciously receive the petition of his loyal subjects.",
    "It is reported that the plague hath spread through the parish.",
    "Concerning the nature of witchcraft, many learned men have written.",
    "The merchant sold his wares at the market on Thursday last.",
    "Let all men know that this covenant is binding before God.",
    "The soldiers marched upon the town at break of day.",
]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def summarize(name: str, values: list[float]) -> None:
    arr = np.array([v for v in values if not np.isnan(v)])
    if arr.size == 0:
        print(f"{name}: no valid samples")
        return
    pct = np.percentile(arr, [5, 50, 95]) if arr.size > 1 else [arr[0]] * 3
    print(
        f"{name}: n={arr.size}  mean={arr.mean():.5f}  "
        f"p5={pct[0]:.5f}  median={pct[1]:.5f}  p95={pct[2]:.5f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--text",
        action="append",
        default=None,
        help="Sentence to test (repeatable). If omitted, uses a small "
             "built-in set of sample sentences.",
    )
    args = ap.parse_args()

    texts = args.text if args.text else DEFAULT_TEXTS

    if len(texts) < 2:
        raise SystemExit(
            "Need at least 2 sentences to compute a reference distance "
            "(pass --text twice, or omit --text to use the defaults)."
        )

    logger.info("[quant-test] Loading fp32 and int8 ONNX models...")
    model_fp32 = load_macberth_onnx(quantize=False)
    model_int8 = load_macberth_onnx(quantize=True)

    embedder_fp32 = MacBERThEmbedder(model_fp32, pooling="mean")
    embedder_int8 = MacBERThEmbedder(model_int8, pooling="mean")

    print(f"\nEmbedding {len(texts)} sentence(s) with both models...\n")

    fp32_vecs = []
    noise_similarities = []

    for text in texts:
        vec_fp32 = embedder_fp32.encode(text)[0]
        vec_int8 = embedder_int8.encode(text)[0]

        sim = cosine(vec_fp32, vec_int8)
        noise_similarities.append(sim)
        fp32_vecs.append(vec_fp32)

        print(f"  [{sim:.5f}] {text[:70]}")

    # Reference scale: fp32-only distance between different sentences.
    # This is NOT diachronic drift -- it's just "how far apart do
    # obviously different sentences land," as a sanity-scale reference.
    reference_similarities = []
    for i in range(len(fp32_vecs)):
        for j in range(i + 1, len(fp32_vecs)):
            reference_similarities.append(cosine(fp32_vecs[i], fp32_vecs[j]))

    noise_distances = [1.0 - s for s in noise_similarities]
    reference_distances = [1.0 - s for s in reference_similarities]

    print("\n=== Summary (as cosine distance, 1 - similarity) ===\n")
    summarize("NOISE      (fp32 vs int8, same sentence)", noise_distances)
    summarize("REFERENCE  (fp32 vs fp32, different sentences)", reference_distances)

    if noise_distances and reference_distances:
        noise_median = float(np.median(noise_distances))
        ref_median = float(np.median(reference_distances))
        ratio = noise_median / ref_median if ref_median > 0 else float("inf")

        print(f"\nnoise_median / reference_median = {ratio:.4f}")
        print(
            "\nNote: 'reference' here is the gap between semantically "
            "different sentences, which is a coarse scale check, not a "
            "measurement of real diachronic drift. A small ratio is "
            "reassuring but doesn't by itself confirm int8 is safe for "
            "subtle across-year semantic-change analysis -- that "
            "specifically needs real corpus occurrences across years."
        )


if __name__ == "__main__":
    main()