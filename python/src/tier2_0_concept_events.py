#!/usr/bin/env python
"""
Tier2 v3 (event-consistent): neighbourhood analysis over event-space substrate

Core invariant
--------------

Tier 1:
    token x window -> embedding event (Zarr)

Tier 2:
    event -> geometric neighbourhood (FAISS) -> contextual analysis (distributional statistics)

FAISS remains a geometric operator only.

No aggregation or centroid reconstruction is performed.
Events remain atomic observations with full provenance.
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


K = 25
BATCH_SIZE = 8192

OUTPUT_PATH = INDEXES_DIR / "tier2_concept_neighbours.json"

# Event-space resolver
class ZarrEventLookup:
    """
    Materialises an in-memory view of the event-space.

    Each entry corresponds to a single observation event:

        (vector_id, doc_id, token_idx, window_id) -> embedding + metadata
    """

    def __init__(self, root):
        self.root = root
        self.by_vector_id = {}
        self._build()

    def _build(self):
        logger.info("[tier2] building event lookup")

        for slice_dir in sorted(self.root.iterdir()):
            if not slice_dir.is_dir():
                continue

            g = zarr.open_group(str(slice_dir), mode="r")

            try:
                events = g["events"]
            except KeyError:
                continue

            vids = events["vector_id"]
            docs = events["doc_id"]
            toks = events["token"]
            idxs = events["token_idx"]
            wins = events["window_id"]

            n = vids.shape[0]

            for start in range(0, n, BATCH_SIZE):
                end = min(n, start + BATCH_SIZE)

                for vid, doc, tok, idx, win, vec in zip(
                    vids[start:end],
                    docs[start:end],
                    toks[start:end],
                    idxs[start:end],
                    wins[start:end],
                    vecs[start:end],
                ):
                    self.by_vector_id[int(vid)] = {
                        "vector_id": int(vid),
                        "doc_id": str(doc),
                        "token": str(tok),
                        "token_idx": int(idx),
                        "window_id": int(win),
                    }

        logger.debug(f"[tier2] lookup size={len(self.by_vector_id)}")

    def get_event(self, vector_id: int):
        return self.by_vector_id[int(vector_id)]

    def iter_matching_event_ids(self, forms):
        forms = {x.lower() for x in forms}
        for vid, event in self.by_vector_id.items():
            if event["token"].lower() in forms:
                yield vid


# Concept-level neighbourhood analysis
def analyse_concept(index, lookup, concept_name, concept, top_n=25):
    forms = set(concept["forms"])
    logger.info(f"[tier2] concept={concept_name} forms={len(forms)}")

    event_ids = list(lookup.iter_matching_event_ids(forms))

    if not event_ids:
        logger.warning(f"[tier2] empty concept={concept_name}")
        return {"concept": concept_name, "empty": True}

    token_counter = Counter()
    doc_counter = Counter()
    window_counter = Counter()
    results = []

    for i, eid in enumerate(event_ids):
        if i % 50 == 0:
            logger.debug(
                f"[tier2] {concept_name} event={i}/{len(event_ids)}"
            )

        event = lookup.get_event(eid)
        vec = event["emb_norm"][None, :]

        scores, neigh_ids = index.search(vec, K)

        scores = scores[0]
        neigh_ids = neigh_ids[0]

        neighbours = []

        for nid, score in zip(neigh_ids, scores):

            neighbour = lookup.get_event(int(nid))

            token_counter[neighbour["token"]] += 1
            doc_counter[neighbour["doc_id"]] += 1
            window_counter[neighbour["window_id"]] += 1

            neighbours.append({
                "event_id": int(nid),
                "vector_id": neighbour["vector_id"],
                "token": neighbour["token"],
                "doc_id": neighbour["doc_id"],
                "token_idx": neighbour["token_idx"],
                "window_id": neighbour["window_id"],
                "score": float(score),
            })

        results.append({
            "event_id": int(eid),
            "vector_id": event["vector_id"],
            "token": event["token"],
            "doc_id": event["doc_id"],
            "token_idx": event["token_idx"],
            "window_id": event["window_id"],
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--concept",
        type=str,
        default=None,
        help="Concept to process (e.g. PREROGATIVE, LAW)"
    )
    return parser.parse_args()


def main():
    logger.info("[tier2] init")
    args = parse_args()

    logger.info("[tier2] loading FAISS index")
    index = EeboFaissIndex.load(
        INDEXES_DIR / "faiss" / "tier1.index"
    )

    logger.info("[tier2] building event lookup")
    lookup = ZarrEventLookup(ZARR_ROOT / "tier1")

    output = {}

    concepts = resolve_concepts(args)

    for concept_name, concept in concepts:
        logger.debug(f"[tier2] START concept={concept_name}")

        output[concept_name] = analyse_concept(
            index=index,
            lookup=lookup,
            concept_name=concept_name,
            concept=concept,
        )

        logger.debug(f"[tier2] DONE concept={concept_name}")

    logger.info(f"[tier2] writing output={OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info("[tier2] complete")


if __name__ == "__main__":
    main()
