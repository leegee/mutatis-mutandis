"""
lib.js_divergence.py - Jensen-Shannon Divergence

A method for measuring the similarity or difference between two probability distributions.

Unlike some other measures like Kullback-Leibler (KL) divergence, JSD is symmetric
and always produces a finite value, which makes it more stable and interpretable.

0: The distributions are identical.
Higher value: The distributions are more different.

"""

import numpy as np

def js_divergence(p_counts, q_counts):
    vocab = set(p_counts) | set(q_counts)

    p = np.array([p_counts.get(t, 0) for t in vocab], dtype=float)
    q = np.array([q_counts.get(t, 0) for t in vocab], dtype=float)

    if p.sum() == 0 or q.sum() == 0:
        return 0.0

    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)

    mask_p = p > 0
    mask_q = q > 0

    kl = lambda a, b, mask: np.sum(a[mask] * np.log(a[mask] / b[mask]))
    return float(0.5 * kl(p, m, mask_p) + 0.5 * kl(q, m, mask_q))
