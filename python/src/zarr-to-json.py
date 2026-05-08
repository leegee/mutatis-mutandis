#!/usr/bin/env python
"""
zarr-to-json.py
"""

import zarr
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

from lib.eebo_config import OUT_DIR, ZARR_ROOT, SLICES
from lib.eebo_db import get_connection


TOKENS = ["church", "state", "king", "liberty", "man", "house"]

OUT_JSON = OUT_DIR / "drift.json"



def slice_id_str(s):
    return f"{s[0]}-{s[1]}"


def load_slice(slice_id):
    root = zarr.open_group(str(ZARR_ROOT / slice_id), mode="r")
    return {
        "mean": root["mean"][:],
        "var": root["var"][:],
        "count": root["count"][:],
        "ids": root["ids"][:],
    }


def get_vector_ids(conn, tokens, slice_range):
    """
    token → vector_ids for this slice
    """
    mapping = defaultdict(list)

    with conn.cursor() as cur:
        cur.execute("""
            SELECT vector_id, lower(token)
            FROM pamphlet_tokens
            WHERE lower(token) = ANY(%s)
              AND pub_year BETWEEN %s AND %s
        """, (tokens, slice_range[0], slice_range[1]))

        for vid, tok in cur:
            mapping[tok].append(int(vid))

    return mapping


def cosine_distance(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return 1.0 - np.dot(a, b) / (na * nb)


def export_subset():
    conn = get_connection()

    slice_ids = [slice_id_str(s) for s in SLICES]

    # token → slice → vector
    token_means = defaultdict(dict)
    token_vars = defaultdict(dict)
    token_counts = defaultdict(dict)

    for s, slice_range in zip(slice_ids, SLICES):
        print(f"Processing slice {s}")

        data = load_slice(s)
        id_index = {int(v): i for i, v in enumerate(data["ids"])}

        vid_map = get_vector_ids(conn, TOKENS, slice_range)

        centroid = np.mean(data["mean"], axis=0)
        mean = data["mean"] - centroid

        for tok, vids in vid_map.items():
            vecs = []
            vars_ = []
            counts = []

            for vid in vids:
                idx = id_index.get(vid)
                if idx is None:
                    continue

                vecs.append(mean[idx] * data["count"][idx])
                vars_.append(data["var"][idx] * data["count"][idx])
                counts.append(data["count"][idx])

            if not vecs:
                continue

            total_n = np.sum(counts)

            token_means[tok][s] = np.sum(vecs, axis=0) / total_n
            token_vars[tok][s] = np.sum(vars_, axis=0) / total_n
            token_counts[tok][s] = int(total_n)

    conn.close()

    # build output
    output = {
        "slices": slice_ids,
        "tokens": {}
    }

    baseline = slice_ids[0]

    for tok in TOKENS:
        drift = []
        variance = []
        count = []

        if baseline not in token_means[tok]:
            continue

        base_vec = token_means[tok][baseline]

        for s in slice_ids:
            if s not in token_means[tok]:
                drift.append(None)
                variance.append(None)
                count.append(0)
                continue

            vec = token_means[tok][s]

            drift.append(float(cosine_distance(base_vec, vec)))
            variance.append(float(np.mean(token_vars[tok][s])))
            count.append(token_counts[tok][s])

        output["tokens"][tok] = {
            "drift": drift,
            "variance": variance,
            "count": count,
        }

    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    export_subset()
