#!/usr/bin/env python
"""
Tier2: neighbourhood analysis over event-space substrate

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


# ------------------------------------------------------------
# Event metadata + embedding index
# ------------------------------------------------------------

class ZarrEventLookup:
    """
    In-memory index of all Tier 1 observation events.

    Key invariant:
        Keyed by event_id (stable, globally unique observation identity).
        vector_id is stored as metadata only - NOT used as key.

    Embeddings are loaded alongside metadata so that FAISS queries
    can be issued without reaching back into private index internals.
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

                for i in range(end - start):
                    eid = int(eids[start + i])

                    self.by_event_id[eid] = {
                        "event_id": eid,
                        "vector_id": int(vids[start + i]),
                        "doc_id": str(docs[start + i]),
                        "token": str(toks[start + i]),
                        "token_idx": int(idxs[start + i]),
                        "window_id": int(wins[start + i]),
                        "window_token_pos": int(wpos[start + i]) if wpos is not None else None,
                        "embedding": np.asarray(embs[start + i], dtype=np.float32),
                    }

        logger.info(f"[tier2] events={len(self.by_event_id)}")

    def get_event(self, event_id: int):
        return self.by_event_id[int(event_id)]

    def iter_matching_event_ids(self, forms):
        forms = {f.lower() for f in forms}

        for eid, event in self.by_event_id.items():
            if event["token"].lower() in forms:
                yield eid


# ------------------------------------------------------------
# Concept analysis
# ------------------------------------------------------------

def analyse_concept(index, lookup, concept_name, concept, top_n=25):
    forms = set(concept["forms"])
    logger.info(f"[tier2] concept={concept_name}")

    event_ids = list(lookup.iter_matching_event_ids(forms))

    if not event_ids:
        return {"concept": concept_name, "empty": True}

    token_counter = Counter()
    doc_counter = Counter()
    window_counter = Counter()

    results = []

    for i, eid in enumerate(event_ids):
        if i % 50 == 0:
            logger.debug(f"[tier2] {concept_name} {i}/{len(event_ids)}")

        event = lookup.get_event(eid)

        # Vector comes from Zarr (source of truth), not FAISS internals
        vec = event["embedding"][None, :]

        scores, neigh_ids = index.search(vec, K)

        neighbours = []

        for nid, score in zip(neigh_ids[0], scores[0]):

            if nid == -1:
                continue

            # exclude query event itself from neighbourhood
            if int(nid) == int(eid):
                continue

            n_event = lookup.get_event(int(nid))

            token_counter[n_event["token"]] += 1
            doc_counter[n_event["doc_id"]] += 1
            # window_counter scoped to (doc_id, window_id) - window_id is doc-local
            window_counter[(n_event["doc_id"], n_event["window_id"])] += 1

            neighbours.append({
                "event_id": int(nid),
                "vector_id": n_event["vector_id"],
                "token": n_event["token"],
                "doc_id": n_event["doc_id"],
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


def main():
    logger.info("[tier2] init")

    args = argparse.ArgumentParser()
    args.add_argument("--concept", type=str, default=None)
    args = args.parse_args()

    index = EeboFaissIndex.load(INDEXES_DIR / "faiss" / "tier1.index")
    lookup = ZarrEventLookup(ZARR_ROOT / "tier1")

    output = {}

    for concept_name, concept in resolve_concepts(args):
        output[concept_name] = analyse_concept(
            index, lookup, concept_name, concept
        )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info(f"[tier2] written -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()