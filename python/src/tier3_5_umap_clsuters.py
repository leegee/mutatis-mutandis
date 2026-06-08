#!/usr/bin/env python
"""
tier3_5_clustering.py - Tier 3: UMAP + HDBSCAN clustering → JSON files
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np

import umap
import hdbscan

from lib.eebo_config import SQLITE_DB_PATH, ZARR_ROOT, CLUSTER_JSON_PATH
from lib.eebo_logging import logger
from lib.concept_resolve import resolve_concepts
from tier2_0_concept_events import ZarrEventLookup


def cluster_concept_events(query_vecs: np.ndarray):
    """UMAP + HDBSCAN → returns labels and 2D projection."""
    if len(query_vecs) < 10:
        return {
            "labels": [-1] * len(query_vecs),
            "umap_2d": None
        }

    reducer = umap.UMAP(
        n_components=5,
        n_neighbors=15,
        min_dist=0.0,
        metric='cosine',
        random_state=42,
    )
    reduced = reducer.fit_transform(query_vecs)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=3,
        metric='euclidean',
        cluster_selection_method='eom',
    )
    labels = clusterer.fit_predict(reduced)

    return {
        "labels": labels.tolist(),
        "umap_2d": reduced[:, :2].tolist()
    }


def compute_cluster_aggregates(event_ids, cluster_labels, lookup, top_n=25):
    from collections import defaultdict, Counter
    token_counters = defaultdict(Counter)
    doc_counters = defaultdict(Counter)

    for eid, cid in zip(event_ids, cluster_labels):
        if cid == -1:
            continue
        event = lookup.get_event(eid)
        token_counters[cid][event["token"]] += 1
        doc_counters[cid][event["doc_id"]] += 1

    aggregates = {}
    for cid in token_counters:
        aggregates[str(cid)] = {
            "top_tokens": token_counters[cid].most_common(top_n),
            "top_docs": doc_counters[cid].most_common(top_n),
        }
    return aggregates


def build_cluster_json(concept_name, event_ids, cluster_data, aggregates, lookup):
    label_map = {}
    unique_clusters = sorted(c for c in set(cluster_data["labels"]) if c != -1)
    for i, cid in enumerate(unique_clusters):
        label_map[cid] = chr(65 + i)  # A, B, C...

    events = []
    for i, eid in enumerate(event_ids):
        cid = cluster_data["labels"][i]
        event = lookup.get_event(eid)
        events.append({
            "event_id": int(eid),
            "token": event["token"],
            "doc_id": event["doc_id"],
            "pub_year": event.get("pub_year"),
            "token_idx": event["token_idx"],
            "cluster_id": int(cid),
            "cluster_label": label_map.get(cid),
            "umap_x": float(cluster_data["umap_2d"][i][0]) if cluster_data["umap_2d"] else None,
            "umap_y": float(cluster_data["umap_2d"][i][1]) if cluster_data["umap_2d"] else None,
        })

    return {
        "concept": concept_name,
        "n_events": len(event_ids),
        "generated_at": datetime.now().isoformat(),
        "clusters": {
            "labels": cluster_data["labels"],
            "umap_2d": cluster_data["umap_2d"],
            "label_map": label_map,
            "aggregates": aggregates
        },
        "events": events
    }


def main():
    logger.info("[tier3] JSON clustering pipeline started")

    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", type=str, help="Single concept to cluster")
    parser.add_argument("--clear", action="store_true", help="Clear existing JSON files")
    args = parser.parse_args()

    CLUSTER_JSON_PATH.mkdir(parents=True, exist_ok=True)

    if args.clear:
        for f in CLUSTER_JSON_PATH.glob("*.json"):
            f.unlink()
        logger.info(f"[tier3] Cleared all files in {CLUSTER_JSON_PATH}")

    lookup = ZarrEventLookup(ZARR_ROOT / "tier1", forms=None)

    concepts_to_run = resolve_concepts(args)

    for concept_name, concept in concepts_to_run:
        logger.info(f"[tier3] Processing {concept_name}")

        forms = set(concept["forms"])
        fps = set(concept.get("false_positives", []))

        event_ids = list(lookup.iter_matching_event_ids(forms, fps))
        if len(event_ids) < 10:
            logger.info(f"[tier3] Skipping {concept_name} — too few events ({len(event_ids)})")
            continue

        query_vecs = np.stack([lookup.get_event(eid)["embedding"] for eid in event_ids])

        cluster_data = cluster_concept_events(query_vecs)
        aggregates = compute_cluster_aggregates(event_ids, cluster_data["labels"], lookup)

        cluster_json = build_cluster_json(concept_name, event_ids, cluster_data, aggregates, lookup)

        # Write to JSON file
        out_path = CLUSTER_JSON_PATH / f"{concept_name.lower()}.json"
        out_path.write_text(json.dumps(cluster_json, indent=2))

        logger.info(f"[tier3] Wrote {len(event_ids)} events → {out_path}")

    logger.info("[tier3] Clustering pipeline finished")


if __name__ == "__main__":
    main()
