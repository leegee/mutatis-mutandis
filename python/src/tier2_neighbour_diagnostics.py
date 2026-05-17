#!/usr/bin/env python
"""
tier2_neighbour_diagnostics.py

Probe semantic neighbours around LIBERTY.

Purpose:
    Discover:
        - OCR variants
        - orthographic variants
        - semantic aliases
        - contamination / false positives

Method:
    1. Build centroid from known LIBERTY forms
    2. Build token -> mean embedding map
    3. Compute cosine similarity to all tokens
    4. Output ranked nearest neighbours


### Methodology

The initial Tier 2 approach attempted slice-local semantic clustering using DBSCAN over contextual token embeddings derived from the EEBO pamphlet corpus. Clustering was performed independently within chronological slices in order to identify local semantic microstructures and possible semantic drift over time.

During testing, most concepts (eg LIBERTY, LAW, RELIGION) produced either no stable clusters or extremely sparse local structures. Investigation revealed that many slice-level distributions contained very few occurrences, often spread across multiple documents and rhetorical contexts, making density-based clustering unstable.

To diagnose this, a corpus-wide semantic neighbour analysis was implemented for LIBERTY. A centroid embedding was constructed from known orthographic and OCR-normalised forms (eg liberty, liberties, libertie), and nearest semantic neighbours were retrieved across the entire embedding space.

### Findings

The neighbour analysis demonstrated that the embeddings themselves are semantically coherent. The nearest neighbours to LIBERTY included:

* lexical variants: liberties, libertie
* semantic associates: freedomes, freedome, priviledges
* political/discourse terms: authority, monarchy, burthens

This suggests that the embedding space captures a meaningful Early Modern political-semantic field rather than random lexical similarity.

Importantly, many high-similarity neighbours were not synonyms but recurring discourse companions (eg disorders, terrour, contentment), indicating that the model is detecting rhetorical and ideological co-positioning within pamphlet literature.

The results imply that the earlier clustering failures were caused primarily by slice-level sparsity and fragmentation rather than absence of semantic structure.

The experiments also suggest that subtracting a "global bias centroid" may currently be methodologically premature, since this global structure may itself encode historically meaningful discourse organisation.

### Next Step

The project will shift from attempting primarily slice-local semantic clustering toward modelling:

* semantic neighbourhoods
* discourse fields
* temporal neighbour drift
* changing conceptual associations over time

The next stage will therefore focus on tracking how the semantic environment surrounding concepts such as LIBERTY changes across the EEBO corpus, rather than assuming that stable local clusters necessarily exist within individual temporal slices.

"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import zarr

from lib.eebo_db import get_connection
from lib.eebo_config import (
    ZARR_ROOT,
    OUT_DIR,
    CONCEPT_SETS
)

from lib.eebo_logging import logger


NEIGHBOURS_OUTPUT_PATH = OUT_DIR / "liberty_neighbours.json"


def load_all_embeddings():
    logger.info("[diag] loading embeddings")

    all_vecs = []
    all_ids = []

    tier1_root = ZARR_ROOT / "tier1"

    for path in sorted(tier1_root.iterdir()):

        if not path.is_dir():
            continue

        root = zarr.open(path, mode="r")

        vecs = root["vecs"][:]
        ids = root["ids"][:]

        all_vecs.append(vecs)
        all_ids.append(ids)

        logger.info(
            f"[diag] loaded_slice={path.name} "
            f"vecs={len(vecs)}"
        )

    vecs = np.concatenate(all_vecs, axis=0)
    ids = np.concatenate(all_ids, axis=0)

    logger.info(
        f"[diag] total_embeddings={len(vecs)}"
    )

    return vecs, ids


# Load token occurrences from DB
def load_token_rows():
    logger.info("[diag] loading token rows")

    conn = get_connection()

    with conn.cursor() as cur:

        cur.execute("""
            SELECT
                vector_id,
                token,
                doc_id
            FROM pamphlet_tokens
        """)

        rows = list(cur)

    conn.close()
    logger.info( f"[diag] token_rows={len(rows)}" )
    return rows


def build_maps(rows):
    vector_to_token = {}
    vector_to_doc = {}

    token_counts = defaultdict(int)
    token_docs = defaultdict(set)

    for vector_id, token, doc_id in rows:
        vid = int(vector_id)
        token = str(token)
        vector_to_token[vid] = token
        vector_to_doc[vid] = doc_id
        token_counts[token] += 1
        token_docs[token].add(doc_id)

    logger.info( f"[diag] unique_tokens={len(token_counts)}" )

    return (
        vector_to_token,
        vector_to_doc,
        token_counts,
        token_docs
    )


def build_liberty_centroid(
    vecs,
    ids,
    vector_to_token,
    liberty_forms
):
    logger.info( f"[diag] liberty_forms={len(liberty_forms)}" )

    mask = np.array([
        vector_to_token.get(int(v)) in liberty_forms
        for v in ids
    ])

    liberty_vecs = vecs[mask]

    logger.info( f"[diag] liberty_vectors={len(liberty_vecs)}" )

    if len(liberty_vecs) == 0:
        raise RuntimeError(
            "No LIBERTY vectors found"
        )

    centroid = liberty_vecs.mean(axis=0)

    centroid = centroid / (
        np.linalg.norm(centroid) + 1e-12
    )

    return centroid, len(liberty_vecs)


def build_token_centroids(
    vecs,
    ids,
    vector_to_token
):
    logger.info( "[diag] building token centroids" )

    bucket = defaultdict(list)

    for i, vid in enumerate(ids):
        token = vector_to_token.get(int(vid))
        if token is None:
            continue
        bucket[token].append(vecs[i])

    token_centroids = {}

    for token, members in bucket.items():
        arr = np.stack(members)
        centroid = arr.mean(axis=0)
        centroid = centroid / (
            np.linalg.norm(centroid) + 1e-12
        )
        token_centroids[token] = centroid

    logger.info( f"[diag] token_centroids={len(token_centroids)}" )
    return token_centroids

def cosine(a, b):
    return float(
        np.dot(a, b) /
        (
            (np.linalg.norm(a) *
             np.linalg.norm(b)) + 1e-12
        )
    )


def neighbours_of():
    liberty_forms = {
        x.lower()
        for x in CONCEPT_SETS["LIBERTY"]["forms"]
    }

    vecs, ids = load_all_embeddings()
    rows = load_token_rows()

    (
        vector_to_token,
        vector_to_doc,
        token_counts,
        token_docs
    ) = build_maps(rows)

    liberty_centroid, n_vectors = (
        build_liberty_centroid(
            vecs,
            ids,
            vector_to_token,
            liberty_forms
        )
    )

    token_centroids = build_token_centroids(
        vecs,
        ids,
        vector_to_token
    )

    logger.info( "[diag] computing neighbours" )
    neighbours = []

    for token, centroid in token_centroids.items():
        sim = cosine(
            liberty_centroid,
            centroid
        )

        neighbours.append({
            "token": token,
            "similarity": sim,
            "occurrences": int(
                token_counts[token]
            ),
            "doc_count": int( len(token_docs[token]) ),
            "known_form": token in liberty_forms
        })

    neighbours.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    output = {
        "query": "LIBERTY",
        "n_vectors": n_vectors,
        "nearest": neighbours[:500]
    }

    with open(
        NEIGHBOURS_OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(output, f, indent=2)

    logger.info( f"[diag] wrote={NEIGHBOURS_OUTPUT_PATH}" )


if __name__ == "__main__":
    neighbours_of()
