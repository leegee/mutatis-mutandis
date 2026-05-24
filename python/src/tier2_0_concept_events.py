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

    pub_year, slice_start, slice_end are stable per doc_id so
    we take the first occurrence of each.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT ON (doc_id)
            doc_id,
            pub_year,
            slice_start,
            slice_end,
            title
        FROM pamphlet_tokens
        ORDER BY doc_id
    """)
    return {
        row[0]: {
            "pub_year":    row[1],
            "slice_start": row[2],
            "slice_end":   row[3],
            "title":       row[4],
        }
        for row in cur.fetchall()
    }

# Concept analysis
def analyse_concept(doc_meta, index, lookup, concept_name, concept, top_n=25):
    """
    Compute neighbourhood structure for all events matching a concept.

    FAISS search is batched over all matching events: a single index.search()
    call is issued with all query vectors stacked, rather than one call per
    event. For concepts with many matching events this is materially faster
    because FAISS parallelises multi-query search internally.

    Vectors come from ZarrEventLookup (Zarr is the canonical source of truth),
    not from FAISS internal storage. See module docstring re: the deferred
    migration to EeboFaissIndex.reconstruct().

    window_counter keys are (doc_id, window_id). window_id is doc-local;
    see module docstring for the scoping invariant.
    """

    forms = set(concept["forms"])
    logger.info(f"[tier2] concept={concept_name}")

    event_ids = list(lookup.iter_matching_event_ids(forms))

    if not event_ids:
        return {"concept": concept_name, "empty": True}

    # Stack all query vectors and issue a single batched FAISS search.
    # Previously this was one index.search() call per event; FAISS
    # parallelises multi-query search so the batched form is significantly
    # faster for concepts with many matching events.
    query_vecs = np.stack(
        [lookup.get_event(eid)["embedding"] for eid in event_ids]
    )  # (n_events, dim)

    all_scores, all_neigh_ids = index.search(query_vecs, K)
    # all_scores:    (n_events, K)
    # all_neigh_ids: (n_events, K)

    token_counter = Counter()
    doc_counter = Counter()
    window_counter = Counter()

    results = []

    for i, eid in enumerate(event_ids):
        event = lookup.get_event(eid)

        neighbours = []

        for nid, score in zip(all_neigh_ids[i], all_scores[i]):

            if nid == -1:
                continue

            # exclude query event itself from neighbourhood
            if int(nid) == int(eid):
                continue

            n_event = lookup.get_event(int(nid))

            token_counter[n_event["token"]] += 1
            doc_counter[n_event["doc_id"]] += 1
            # window_id is doc-local: key must include doc_id to avoid
            # merging windows from different documents that share an offset.
            window_counter[(n_event["doc_id"], n_event["window_id"])] += 1

            neighbours.append({
                "event_id":         int(nid),
                "vector_id":        n_event["vector_id"],
                "token":            n_event["token"],
                "doc_id":           n_event["doc_id"],
                "pub_year":         doc_meta.get(event["doc_id"], {}).get("pub_year"),
                "slice_start":      doc_meta.get(event["doc_id"], {}).get("slice_start"),
                "slice_end":        doc_meta.get(event["doc_id"], {}).get("slice_end"),
                "token_idx":        n_event["token_idx"],
                "window_id":        n_event["window_id"],
                "window_token_pos": n_event["window_token_pos"],
                "score":            float(score),
            })

        results.append({
            "event_id":         int(eid),
            "vector_id":        event["vector_id"],
            "token":            event["token"],
            "doc_id":           event["doc_id"],
            "pub_year":         doc_meta.get(event["doc_id"], {}).get("pub_year"),
            "slice_start":      doc_meta.get(event["doc_id"], {}).get("slice_start"),
            "slice_end":        doc_meta.get(event["doc_id"], {}).get("slice_end"),
            "token_idx":        event["token_idx"],
            "window_id":        event["window_id"],
            "window_token_pos": event["window_token_pos"],
            "neighbours":       neighbours,
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


def main():
    logger.info("[tier2] init")

    args = argparse.ArgumentParser()
    args.add_argument("--concept", type=str, default=None)
    args = args.parse_args()

    index = EeboFaissIndex.load(INDEXES_DIR / "faiss" / "tier1.index")
    lookup = ZarrEventLookup(ZARR_ROOT / "tier1")

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
