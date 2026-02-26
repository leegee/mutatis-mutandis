#!/usr/bin/env python
"""
compute_liberty_poles.py

Uses FAISS slice indexes and PCA to attempt to compute conceptual poles.

Outputs HTML to `semantic_poles_liberty.html`
"""

from __future__ import annotations
from typing import Any, List, Tuple
import numpy as np
import faiss
from sklearn.decomposition import PCA
import pandas as pd
from collections import Counter

import lib.eebo_config as config
from lib.faiss_slices import faiss_slice_path, vocab_slice_path
from lib.eebo_logging import logger
from lib.wordlist import STOPWORDS

TOP_N_WORDS = 15
WINDOW = 5
LIBERTY_FORMS = config.CONCEPT_SETS["LIBERTY"]["forms"]
FALSE_POSITIVES = config.CONCEPT_SETS["LIBERTY"]["false_positives"]


def get_liberty_context_vectors(index, vocab: List[str]) -> Tuple[List[str], np.ndarray, Counter[str]]:
    """
    Returns:
        - list of context tokens (actual variant, not canonicalized)
        - stacked vectors
        - counts of each token in context
    """
    token_to_idx = {token.lower(): i for i, token in enumerate(vocab)}
    context_tokens: List[str] = []
    vectors: List[np.ndarray] = []
    counts: Counter[str] = Counter()

    for form in LIBERTY_FORMS:
        idx = token_to_idx.get(form.lower())
        if idx is None:
            continue

        start = max(idx - WINDOW, 0)
        end = min(idx + WINDOW + 1, len(vocab))

        for i in range(start, end):
            token = vocab[i].strip()  # preserve original casing/variant
            token_lc = token.lower()

            if token_lc in STOPWORDS or token_lc in FALSE_POSITIVES:
                continue

            # No longer  canonicalized!
            vec = index.reconstruct(i)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm

            vectors.append(vec)
            context_tokens.append(token)
            counts[token] += 1

    if not vectors:
        raise RuntimeError("No LIBERTY-context tokens found in slice")

    return context_tokens, np.stack(vectors).astype(np.float32), counts



def compute_poles(word_vectors: np.ndarray, words: List[str], top_n=TOP_N_WORDS):
    centroid = word_vectors.mean(axis=0)
    X = word_vectors - centroid

    pca = PCA(n_components=1)
    pca.fit(X)

    pc1 = pca.components_[0]
    pc1 /= np.linalg.norm(pc1)

    scores = X @ pc1

    top_idx = np.argsort(scores)[-top_n:][::-1]
    bottom_idx = np.argsort(scores)[:top_n]

    return (
        [words[i] for i in top_idx],
        [words[i] for i in bottom_idx],
        float(pca.explained_variance_ratio_[0])
    )


results: List[Tuple[str, float, List[str], List[str], Counter[str]]] = []

for slice_range in config.SLICES:
    index_path = faiss_slice_path(slice_range)
    vocab_path = vocab_slice_path(slice_range)

    if not index_path.exists() or not vocab_path.exists():
        continue

    logger.debug(f"Slice {slice_range[0]}–{slice_range[1]} ===")

    index = faiss.read_index(str(index_path))
    vocab = [line.strip() for line in open(vocab_path, encoding="utf-8")]

    words, vectors, counts = get_liberty_context_vectors(index, vocab)
    if len(words) < 10:
        continue

    pos_words, neg_words, var = compute_poles(vectors, words)
    slice_label = f"{slice_range[0]}–{slice_range[1]}"

    results.append((slice_label, var, pos_words, neg_words, counts))


# HTML output
columns: List[str] = ["Slice", "PC1 Var"] + [f"Rank {i+1}" for i in range(TOP_N_WORDS)]
table_rows: List[List[Any]] = []

for slice_label, var, pos_words, neg_words, counts in results:
    row: List[Any] = [slice_label, f"{var:.3f}"]

    for i in range(TOP_N_WORDS):
        pos = pos_words[i] if i < len(pos_words) else ""
        neg = neg_words[i] if i < len(neg_words) else ""

        # display actual counts
        pos_count = counts.get(pos, 0) if pos else 0
        neg_count = counts.get(neg, 0) if neg else 0

        cell_html = (
            f'<span class="pos">{pos}&nbsp;({pos_count})</span><br>'
            f'<span class="neg">{neg}&nbsp;({neg_count})</span>'
        )
        row.append(cell_html)

    table_rows.append(row)

df = pd.DataFrame(table_rows)
df.columns = columns
html_table = df.to_html(index=False, escape=False)

css = """
<style>
body { background:#111; color:#ddd; font-family:sans; font-size:12pt; }
table { border-collapse: collapse; width: 100%; table-layout: fixed; }
th, td {
    border: 1px solid #444;
    padding: 1em;
    vertical-align: top;
    word-wrap: no-wrap;
    white-space: no-wrap;
}
th { background:#222; }
.pos { color:#7dd3fc; }
.neg { color:#fca5a5; margin-top:0.3em; display:block; }
</style>
"""

output_path = config.OUT_DIR / "semantic_poles_liberty_aligned.html"
output_path.write_text(css + html_table, encoding="utf-8")
logger.info(f"Wrote {output_path}")
