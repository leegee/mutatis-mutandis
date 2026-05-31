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
doc_id, token_idx binds to the source text in the db

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

import argparse
import json
import random
from collections import Counter
from itertools import combinations

import numpy as np
import zarr

from lib.eebo_config import CONCEPT_SETS, FAISS_INDEX_DIR, INDEXES_DIR, ZARR_ROOT
from lib.eebo_faiss import EeboFaissIndex
from lib.eebo_logging import logger
from lib.concept_resolve import resolve_concepts
from lib.eebo_db import get_connection


K          = 25
BATCH_SIZE = 8192
OUTPUT_PATH = INDEXES_DIR / "tier2_concept_neighbours.json"

#
# TODO Move Zarr routines into lib
#
class ZarrEventLookup:
    """
    In-memory index of all Tier 1 observation events, keyed by event_id.

    vector_id is stored as metadata only — NOT used as a lookup key.

    Embeddings are loaded alongside metadata so that FAISS queries can be
    issued using the canonical Zarr vector rather than relying on FAISS
    internal vector storage. See module docstring for trade-offs and the
    deferred migration path to EeboFaissIndex.reconstruct().
    """

    def __init__(self, root):
        self.root        = root
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

            self._load_slice(g["events"], slice_dir)

        logger.info(f"[tier2] events={len(self.by_event_id)}")

    def _load_slice(self, e, slice_dir):
        if "event_id" not in e:
            raise KeyError(f"Missing event_id in {slice_dir} - rebuild Tier 1")

        wpos = e["window_token_pos"] if "window_token_pos" in e else None
        n    = e["event_id"].shape[0]

        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)

            # Read each dataset as a contiguous numpy array per batch.
            # Indexing into numpy in the inner loop is cheap; indexing
            # directly into Zarr would trigger per-element decompression.
            b_eids = e["event_id"][start:end]
            b_vids = e["vector_id"][start:end]
            b_docs = e["doc_id"][start:end]
            b_toks = e["token"][start:end]
            b_idxs = e["token_idx"][start:end]
            b_wins = e["window_id"][start:end]
            b_embs = e["emb_raw"][start:end]
            b_wpos = wpos[start:end] if wpos is not None else None

            for i in range(end - start):
                eid = int(b_eids[i])
                self.by_event_id[eid] = {
                    "event_id":         eid,
                    "vector_id":        int(b_vids[i]),
                    "doc_id":           str(b_docs[i]),
                    "token":            str(b_toks[i]),
                    "token_idx":        int(b_idxs[i]),
                    "window_id":        int(b_wins[i]),
                    "window_token_pos": int(b_wpos[i]) if b_wpos is not None else None,
                    # NOTE: storing embeddings here holds the full corpus
                    # embedding matrix in memory. See module docstring for
                    # the deferred migration to EeboFaissIndex.reconstruct().
                    "embedding":        np.asarray(b_embs[i], dtype=np.float32),
                }

    def get_event(self, event_id: int) -> dict:
        return self.by_event_id[int(event_id)]

    def iter_matching_event_ids(self, forms):
        forms = {f.lower() for f in forms}
        for eid, event in self.by_event_id.items():
            if event["token"].lower() in forms:
                yield eid



def load_doc_metadata(conn) -> dict:
    """
    Build a doc_id -> metadata mapping from pamphlet_tokens.
    pub_year and title are stable per doc_id; we take the first occurrence.
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
        row[0]: {"pub_year": row[1], "title": row[2]}
        for row in cur.fetchall()
    }

#
# Diagnostics - extrapolate
#
def _audit_embedding_diversity(concept_name, query_vecs):
    """Log embedding norm and cosine diversity stats; warn on collapse."""
    logger.info("[tier2] EMBEDDING DIVERSITY AUDIT START")

    sample   = query_vecs[:min(50, len(query_vecs))]
    sample_n = len(sample)

    norms = np.linalg.norm(sample, axis=1)
    logger.info(
        f"[tier2] norms: mean={norms.mean():.6f} std={norms.std():.6f} "
        f"min={norms.min():.6f} max={norms.max():.6f}"
    )

    normed     = sample / (np.linalg.norm(sample, axis=1, keepdims=True) + 1e-12)
    sim_matrix = normed @ normed.T
    off_diag   = sim_matrix[~np.eye(sample_n, dtype=bool)]

    logger.info(
        f"[tier2] cosine: mean={off_diag.mean():.6f} "
        f"std={off_diag.std():.6f} "
        f"p95={np.percentile(off_diag, 95):.6f} "
        f"max={off_diag.max():.6f}"
    )

    if off_diag.std() < 1e-3:
        logger.warning(
            "[tier2] EMBEDDING COLLAPSE SUSPECTED "
            "(near-constant semantic neighbourhood geometry)"
        )


def _audit_neighbour_identity(all_neigh_ids):
    """Log the most frequently returned neighbour ids across all queries."""
    flat_ids = all_neigh_ids.flatten()
    if not len(flat_ids):
        return

    freq   = Counter(flat_ids)
    top10  = freq.most_common(10)

    logger.info("[tier2] TOP NEIGHBOUR IDS (frequency)")
    for k, v in top10:
        logger.info(f"[tier2] id={k} freq={v}")


def knn_diagnostics(lookup, index, concept_forms, sample_n=25, k=25):
    """Dev utility: print kNN overlap and Jaccard stats for a concept's events."""
    forms     = {f.lower() for f in concept_forms}
    event_ids = list(lookup.iter_matching_event_ids(forms))

    if len(event_ids) < 5:
        print("Too few events")
        return

    event_ids = event_ids[:sample_n]
    vecs      = np.stack([lookup.get_event(eid)["embedding"] for eid in event_ids])
    _, nn_ids = index.search(vecs, k)
    knn_sets  = [set(map(int, row)) for row in nn_ids]

    overlaps  = []
    jaccards  = []
    entropies = []

    for i, j in combinations(range(len(knn_sets)), 2):
        a, b  = knn_sets[i], knn_sets[j]
        inter = len(a & b)
        union = len(a | b)
        overlaps.append(inter)
        jaccards.append(inter / union if union else 0)

    for s in knn_sets:
        flat    = list(s)
        freq    = Counter(flat)
        p       = np.array(list(freq.values())) / len(flat)
        entropy = -(p * np.log(p + 1e-9)).sum()
        entropies.append(entropy)

    print("\n--- KNN DIAGNOSTICS ---")
    print(f"events sampled: {len(event_ids)}")
    print(f"mean overlap: {np.mean(overlaps):.3f} ± {np.std(overlaps):.3f}")
    print(f"mean jaccard: {np.mean(jaccards):.3f} ± {np.std(jaccards):.3f}")
    print(f"mean entropy: {np.mean(entropies):.3f}")
    print("\noverlap quantiles:", np.percentile(overlaps, [0, 25, 50, 75, 100]))
    print("jaccard quantiles:", np.percentile(jaccards, [0, 25, 50, 75, 100]))

#
# Concept analysis
#
def _event_record(event, doc_meta):
    """Serialisable dict for one query event (without neighbours)."""
    return {
        "event_id":         int(event["event_id"]),
        "vector_id":        event["vector_id"],
        "token":            event["token"],
        "doc_id":           event["doc_id"],
        "pub_year":         doc_meta.get(event["doc_id"], {}).get("pub_year"),
        "token_idx":        event["token_idx"],
        "window_id":        event["window_id"],
        "window_token_pos": event["window_token_pos"],
    }


def _neighbour_record(n_event, query_event, doc_meta, score):
    """Serialisable dict for one neighbour of a query event."""
    return {
        "event_id":         int(n_event["event_id"]),
        "vector_id":        n_event["vector_id"],
        "token":            n_event["token"],
        "doc_id":           n_event["doc_id"],
        "pub_year":         doc_meta.get(query_event["doc_id"], {}).get("pub_year"),
        "token_idx":        n_event["token_idx"],
        "window_id":        n_event["window_id"],
        "window_token_pos": n_event["window_token_pos"],
        "score":            float(score),
    }


def analyse_concept(doc_meta, index, lookup, concept_name, concept, top_n=K):
    """Compute neighbourhood structure for all events matching a concept."""
    forms     = set(concept["forms"])
    event_ids = list(lookup.iter_matching_event_ids(forms))

    logger.info(f"[tier2] concept={concept_name}")

    if not event_ids:
        return {"concept": concept_name, "empty": True}

    query_vecs = np.stack([lookup.get_event(eid)["embedding"] for eid in event_ids])

    logger.info(f"[tier2] query_events={len(event_ids)}")
    logger.info(f"[tier2] sample_event_id={event_ids[0]}")
    logger.info(f"[tier2] sample_embedding_shape={query_vecs.shape}")

    _audit_embedding_diversity(concept_name, query_vecs)

    all_scores, all_neigh_ids = index.search(query_vecs, K)

    _audit_neighbour_identity(all_neigh_ids)

    token_counter  = Counter()
    doc_counter    = Counter()
    window_counter = Counter()
    results        = []

    for i, eid in enumerate(event_ids):
        event      = lookup.get_event(eid)
        neighbours = []

        for nid, score in zip(all_neigh_ids[i], all_scores[i]):
            if nid == -1 or int(nid) == int(eid):
                continue

            n_event = lookup.get_event(int(nid))

            token_counter[n_event["token"]]                              += 1
            doc_counter[n_event["doc_id"]]                               += 1
            window_counter[(n_event["doc_id"], n_event["window_id"])]    += 1

            neighbours.append(_neighbour_record(n_event, event, doc_meta, score))

        results.append({**_event_record(event, doc_meta), "neighbours": neighbours})

    return {
        "concept":   concept_name,
        "n_events":  len(event_ids),
        "aggregate": {
            "top_tokens":  token_counter.most_common(top_n),
            "top_docs":    doc_counter.most_common(top_n),
            "top_windows": window_counter.most_common(top_n),
        },
        "events": results,
    }



def main():
    logger.info("[tier2] init")

    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", type=str, default=None)
    args = parser.parse_args()

    faiss_index_path = FAISS_INDEX_DIR / "tier1.index"

    index  = EeboFaissIndex.load(faiss_index_path)
    lookup = ZarrEventLookup(ZARR_ROOT / "tier1")

    logger.info("--------------------------------------------------------")
    knn_diagnostics(lookup, index, CONCEPT_SETS["PREROGATIVE"]["forms"])
    knn_diagnostics(lookup, index, CONCEPT_SETS["LAW"]["forms"])
    logger.info("--------------------------------------------------------")

    conn     = get_connection()
    doc_meta = load_doc_metadata(conn)
    output   = {}

    for concept_name, concept in resolve_concepts(args):
        output[concept_name] = analyse_concept(
            doc_meta, index, lookup, concept_name, concept
        )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info(f"[tier2] written -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
