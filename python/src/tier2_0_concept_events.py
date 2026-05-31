#!/usr/bin/env python
"""
tier2_0_concept_events.py - Tier2: neighbourhood analysis over event-space substrate

Core invariant
--------------

Tier 1:
    token x window -> embedding event (Zarr), keyed by event_id

Tier 2:
    event -> geometric neighbourhood (FAISS) -> contextual analysis

FAISS is a geometric operator only.
FAISS ids are event_ids (stable, globally unique observation identity).
vector_id is lexical identity only and is NOT used as a lookup key.

No aggregation or centroid reconstruction is performed.
Events remain atomic observations with full provenance.

Performance model
-----------------

ZarrEventLookup._build reads each Zarr array slice-by-slice in batches,
materialising one numpy array per dataset per batch before iterating in
Python. This avoids the element-by-element Zarr reads that would otherwise
pay decompression overhead on every scalar access.

Embeddings are currently stored in ZarrEventLookup alongside metadata so
that FAISS queries can be issued using the canonical Zarr vector rather than
relying on FAISS internal storage. This is correct and safe for IndexFlatIP,
which stores vectors verbatim, but it has two consequences:

    1. Memory: all embeddings for the full corpus are resident in the lookup
       dict. For a large corpus this can be several GB.

    2. Coupling: the lookup is responsible for both metadata and vector
       storage, which conflates two concerns.

A cleaner long-term approach is to drop the "embedding" field from
ZarrEventLookup.by_event_id entirely, and instead reconstruct vectors
from the FAISS index at query time via EeboFaissIndex.reconstruct().
See eebo_faiss.py for the reconstruct() method and usage notes.

This migration is deferred until the index type (exact vs. approximate)
is confirmed stable, because IndexHNSWFlat does not support vector
reconstruction.

window_id scoping invariant
---------------------------

window_counter keys are (doc_id, window_id) because window_id is defined
as a document-local coordinate in the Tier 1 store (it is the token-space
start offset of the transformer window within that document). Treating
window_id as globally unique across documents would silently merge windows
from different documents that happen to share the same offset. This
invariant is enforced in Tier 1 but is not re-checked here; if Tier 1
were ever rebuilt with a global window_id scheme this counter would
become incorrect without raising an error.
"""

from __future__ import annotations

from collections import Counter
import json
import numpy as np
import zarr
import argparse
import random
import numpy as np
from itertools import combinations
from collections import Counter

from lib.eebo_config import (
    CONCEPT_SETS,
    INDEXES_DIR,
    ZARR_ROOT,
)
from lib.eebo_faiss import EeboFaissIndex
from lib.eebo_logging import logger
from lib.concept_resolve import resolve_concepts
from lib.eebo_db import get_connection

K = 25
BATCH_SIZE = 8192
OUTPUT_PATH = INDEXES_DIR / "tier2_concept_neighbours.json"



# Event metadata + embedding index
class ZarrEventLookup:
    """
    In-memory index of all Tier 1 observation events.

    Key invariant:
        Keyed by event_id (stable, globally unique observation identity).
        vector_id is stored as metadata only - NOT used as key.

    Embeddings are loaded alongside metadata so that FAISS queries
    can be issued using the canonical Zarr vector rather than relying
    on FAISS internal vector storage. See module docstring for the
    trade-offs and the deferred migration path to EeboFaissIndex.reconstruct().

    Read strategy:
        Each Zarr dataset is read as a contiguous numpy slice per batch.
        Iterating over pre-loaded numpy arrays avoids per-element Zarr
        decompression overhead, which is significant at corpus scale.
    """

    def __init__(self, root):
        self.root = root
        self.by_event_id = {}
        self._build()

    def _build(self):
        logger.info("[tier2] building event lookup")

        for slice_dir in sorted(self.root.iterdir()):
            if not slice_dir.is_dir():
                continue

            g = zarr.open_group(str(slice_dir), mode="r")

            if "events" not in g:
                continue

            e = g["events"]

            if "event_id" not in e:
                raise KeyError(f"Missing event_id in {slice_dir} - rebuild Tier 1")

            eids = e["event_id"]
            vids = e["vector_id"]
            docs = e["doc_id"]
            toks = e["token"]
            idxs = e["token_idx"]
            wins = e["window_id"]
            embs = e["emb_raw"]
            wpos = e["window_token_pos"] if "window_token_pos" in e else None

            n = eids.shape[0]

            for start in range(0, n, BATCH_SIZE):
                end = min(start + BATCH_SIZE, n)

                # Read each dataset once per batch as a contiguous numpy
                # array. Indexing into these numpy arrays in the inner loop
                # is cheap; indexing into the Zarr datasets directly would
                # trigger one decompressed read per element.
                b_eids = eids[start:end]
                b_vids = vids[start:end]
                b_docs = docs[start:end]
                b_toks = toks[start:end]
                b_idxs = idxs[start:end]
                b_wins = wins[start:end]
                b_embs = embs[start:end]
                b_wpos = wpos[start:end] if wpos is not None else None

                for i in range(end - start):
                    eid = int(b_eids[i])

                    self.by_event_id[eid] = {
                        "event_id": eid,
                        "vector_id": int(b_vids[i]),
                        "doc_id": str(b_docs[i]),
                        "token": str(b_toks[i]),
                        "token_idx": int(b_idxs[i]),
                        "window_id": int(b_wins[i]),
                        "window_token_pos": int(b_wpos[i]) if b_wpos is not None else None,
                        # NOTE: storing embeddings here holds the full corpus
                        # embedding matrix in memory. See module docstring for
                        # the deferred migration to EeboFaissIndex.reconstruct().
                        "embedding": np.asarray(b_embs[i], dtype=np.float32),
                    }

        logger.info(f"[tier2] events={len(self.by_event_id)}")

    def get_event(self, event_id: int):
        return self.by_event_id[int(event_id)]

    def iter_matching_event_ids(self, forms):
        forms = {f.lower() for f in forms}

        for eid, event in self.by_event_id.items():
            if event["token"].lower() in forms:
                yield eid


# Additional metadata
def load_doc_metadata(conn) -> dict:
    """
    Build a doc_id -> metadata mapping from pamphlet_tokens.

    pub_year, are stable per doc_id so
    we take the first occurrence of each.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT ON (doc_id)
            doc_id,
            pub_year,
            title
        FROM pamphlet_tokens
        ORDER BY doc_id
    """)

    return {
        row[0]: {
            "pub_year": row[1],
            "title": row[2],
        }
        for row in cur.fetchall()
    }

# Concept analysis
def analyse_concept(doc_meta, index, lookup, concept_name, concept, top_n=25):
    """
    Compute neighbourhood structure for all events matching a concept.
    """

    forms = set(concept["forms"])
    logger.info(f"[tier2] concept={concept_name}")

    event_ids = list(lookup.iter_matching_event_ids(forms))

    if not event_ids:
        return {"concept": concept_name, "empty": True}

    # Build query matrix
    query_vecs = np.stack(
        [lookup.get_event(eid)["embedding"] for eid in event_ids]
    )

    logger.info(f"[tier2] query_events={len(event_ids)}")
    logger.info(f"[tier2] sample_event_id={event_ids[0] if event_ids else None}")
    logger.info(f"[tier2] sample_embedding_shape={query_vecs.shape}")

    # EMBEDDING DIAGNOSTIC (correct cosine computation)
    logger.info("[tier2] EMBEDDING DIVERSITY AUDIT START")

    sample_n = min(50, len(query_vecs))
    sample = query_vecs[:sample_n]

    # L2 norm stats
    norms = np.linalg.norm(sample, axis=1)
    logger.info(
        f"[tier2] norms: mean={norms.mean():.6f} std={norms.std():.6f} "
        f"min={norms.min():.6f} max={norms.max():.6f}"
    )

    # true cosine similarity (normalised)
    normed = sample / (np.linalg.norm(sample, axis=1, keepdims=True) + 1e-12)
    sim_matrix = normed @ normed.T

    mask = ~np.eye(sample_n, dtype=bool)
    off_diag = sim_matrix[mask]

    logger.info(
        f"[tier2] cosine: mean={off_diag.mean():.6f} "
        f"std={off_diag.std():.6f} "
        f"p95={np.percentile(off_diag, 95):.6f} "
        f"max={off_diag.max():.6f}"
    )

    collapse_score = off_diag.std()
    if collapse_score < 1e-3:
        logger.warning(
            "[tier2] EMBEDDING COLLAPSE SUSPECTED "
            "(near-constant semantic neighbourhood geometry)"
        )

    # FAISS SEARCH
    all_scores, all_neigh_ids = index.search(query_vecs, K)

    token_counter = Counter()
    doc_counter = Counter()
    window_counter = Counter()

    results = []

    # DEBUG: neighbour identity entropy (cheap + very informative)
    flat_ids = all_neigh_ids.flatten()
    if len(flat_ids):
        from collections import Counter as _C
        freq = _C(flat_ids)
        top10 = freq.most_common(10)

        logger.info("[tier2] TOP NEIGHBOUR IDS (frequency)")
        for k, v in top10:
            logger.info(f"[tier2] id={k} freq={v}")

    # ------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------
    for i, eid in enumerate(event_ids):
        event = lookup.get_event(eid)

        neighbours = []

        for nid, score in zip(all_neigh_ids[i], all_scores[i]):

            if nid == -1:
                continue

            if int(nid) == int(eid):
                continue

            n_event = lookup.get_event(int(nid))

            token_counter[n_event["token"]] += 1
            doc_counter[n_event["doc_id"]] += 1
            window_counter[(n_event["doc_id"], n_event["window_id"])] += 1

            neighbours.append({
                "event_id": int(nid),
                "vector_id": n_event["vector_id"],
                "token": n_event["token"],
                "doc_id": n_event["doc_id"],
                "pub_year": doc_meta.get(event["doc_id"], {}).get("pub_year"),
                "token_idx": n_event["token_idx"],
                "window_id": n_event["window_id"],
                "window_token_pos": n_event["window_token_pos"],
                "score": float(score),
            })

        results.append({
            "event_id": int(eid),
            "vector_id": event["vector_id"],
            "token": event["token"],
            "doc_id": event["doc_id"],
            "pub_year": doc_meta.get(event["doc_id"], {}).get("pub_year"),
            "token_idx": event["token_idx"],
            "window_id": event["window_id"],
            "window_token_pos": event["window_token_pos"],
            "neighbours": neighbours,
        })

    return {
        "concept": concept_name,
        "n_events": len(event_ids),
        "aggregate": {
            "top_tokens": token_counter.most_common(top_n),
            "top_docs": doc_counter.most_common(top_n),
            "top_windows": window_counter.most_common(top_n),
        },
        "events": results,
    }

def knn_diagnostics(lookup, index, concept_forms, sample_n=25, k=25):
    forms = {f.lower() for f in concept_forms}

    # collect event ids
    event_ids = list(lookup.iter_matching_event_ids(forms))
    if len(event_ids) < 5:
        print("Too few events")
        return

    event_ids = event_ids[:sample_n]

    # get embeddings
    vecs = np.stack([lookup.get_event(eid)["embedding"] for eid in event_ids])

    _, nn_ids = index.search(vecs, k)

    knn_sets = [set(map(int, row)) for row in nn_ids]

    overlaps = []
    jaccards = []
    entropies = []

    for i, j in combinations(range(len(knn_sets)), 2):
        a, b = knn_sets[i], knn_sets[j]

        inter = len(a & b)
        union = len(a | b)

        overlaps.append(inter)
        jaccards.append(inter / union if union else 0)

    for s in knn_sets:
        flat = list(s)
        freq = Counter(flat)
        p = np.array(list(freq.values())) / len(flat)
        entropy = -(p * np.log(p + 1e-9)).sum()
        entropies.append(entropy)

    print("\n--- KNN DIAGNOSTICS ---")
    print(f"events sampled: {len(event_ids)}")
    print(f"mean overlap: {np.mean(overlaps):.3f} ± {np.std(overlaps):.3f}")
    print(f"mean jaccard: {np.mean(jaccards):.3f} ± {np.std(jaccards):.3f}")
    print(f"mean entropy: {np.mean(entropies):.3f}")

    print("\noverlap quantiles:", np.percentile(overlaps, [0, 25, 50, 75, 100]))
    print("jaccard quantiles:", np.percentile(jaccards, [0, 25, 50, 75, 100]))

def main():
    logger.info("[tier2] init")

    args = argparse.ArgumentParser()
    args.add_argument("--concept", type=str, default=None)
    args = args.parse_args()

    faiss_index_path = INDEXES_DIR / "faiss" / "tier1.index"

    logger.info('--------------------------------------------------------')

    index = EeboFaissIndex.load(faiss_index_path)
    lookup = ZarrEventLookup(ZARR_ROOT / "tier1")

    knn_diagnostics(lookup, index, CONCEPT_SETS["PREROGATIVE"]["forms"])
    knn_diagnostics(lookup, index, CONCEPT_SETS["LAW"]["forms"])

    logger.info('--------------------------------------------------------')

    conn = get_connection()
    doc_meta = load_doc_metadata(conn)

    output = {}

    for concept_name, concept in resolve_concepts(args):
        output[concept_name] = analyse_concept(
            doc_meta, index, lookup, concept_name, concept
        )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info(f"[tier2] written -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
