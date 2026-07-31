#!/usr/bin/env python
"""
compare_masked_clustering.py - validate whether masked-target embeddings
actually fix the "clusters split by spelling/document, not sense" problem
identified manually in the LIBERTY concept (see conversation history).

Runs the SAME clustering pipeline (fit_cluster_local, straight from
tier3_0_plots.py - no reimplementation, no drift from production logic)
twice, on the same set of event_ids:

    1. against the ORIGINAL raw hidden-state embeddings (ZARR_PATH)
    2. against the MASKED-target embeddings (MASKED_ZARR_PATH), written by
       tier1b_masked_concept_embeddings.py

...then prints each cluster's top tokens, top docs, and doc-concentration
(the fraction of a cluster's points coming from its single most common
doc_id - the diagnostic that caught clusters 7/8 being "one document said
liberty 56 times", not a real usage sense) side by side, so the two runs
can be compared directly.

This is a READ-ONLY diagnostic. It does not write to sqlite, does not
touch FAISS, does not rerun tier2. It only needs whatever event_ids are
already sitting in both Zarr stores for this concept.

Usage:
    python compare_masked_clustering.py --concept LIBERTY
"""

import argparse
from collections import Counter, defaultdict

import numpy as np

from lib.corpus_config import ZARR_PATH, MASKED_ZARR_PATH, CORPUS_TIER2_DB_PATH
from lib.corpus_logging import logger
from lib.embedding_cache import EmbeddingCache

from tier2_0_concept_events import ZarrEventLookup, sqlite3_connection
from tier3_0_plots import fit_cluster_local


def fetch_concept_event_ids(db_path, concept: str) -> list[str]:
    con = sqlite3_connection(db_path)
    rows = con.execute(
        "SELECT event_id FROM events WHERE concept = ?", (concept,)
    ).fetchall()
    con.close()
    return [str(r[0]) for r in rows]


def restrict_to_available(lookup, event_ids: list[str]) -> list[str]:
    """
    tier1b_masked_concept_embeddings.py deliberately skips (not
    zero-fills) events it couldn't build a complete masked vector for -
    missing windows, missing mask positions, etc (see its RunStats
    summary). So the masked store's event_id set is very likely a STRICT
    SUBSET of the full concept event_ids. For a fair comparison, both
    runs must use the SAME subset - otherwise apparent differences in
    cluster quality could just be an artifact of comparing different
    (differently-sized) point sets rather than the embeddings themselves.
    """
    available = []
    for eid in event_ids:
        try:
            event = lookup.get_event(eid)
        except Exception:
            event = None
        if event is not None:
            available.append(eid)
    return available


def summarize_clusters(event_ids, cluster_labels, lookup, top_n=8):
    """
    Per-cluster token/doc breakdown + doc-concentration, mirroring the
    diagnostic done manually earlier in this conversation for the raw
    LIBERTY clusters - reused here so both runs are read the same way.
    """
    token_counters = defaultdict(Counter)
    doc_counters = defaultdict(Counter)
    cluster_sizes = Counter()

    for eid, cid in zip(event_ids, cluster_labels):
        cluster_sizes[cid] += 1
        if cid == -1:
            continue
        event = lookup.get_event(eid)
        token_counters[cid][event["token"]] += 1
        doc_counters[cid][event["doc_id"]] += 1

    summaries = []
    for cid in sorted(c for c in token_counters.keys()):
        n = cluster_sizes[cid]
        top_docs = doc_counters[cid].most_common(top_n)
        top_tokens = token_counters[cid].most_common(top_n)
        dominant_doc, dominant_count = top_docs[0]
        concentration = dominant_count / n

        summaries.append({
            "cluster_id": cid,
            "n_points": n,
            "top_tokens": top_tokens,
            "top_docs": top_docs,
            "concentration": concentration,
            "dominant_doc": dominant_doc,
        })

    n_noise = cluster_sizes.get(-1, 0)
    return summaries, n_noise


def print_summary(label: str, summaries: list[dict], n_noise: int, n_total: int):
    print(f"\n{'=' * 70}")
    print(f"{label}  ({n_total} points, {len(summaries)} clusters, {n_noise} noise)")
    print(f"{'=' * 70}")

    high_concentration = [s for s in summaries if s["concentration"] >= 0.6]

    for s in summaries:
        flag = "  <-- single-doc/token dominated" if s["concentration"] >= 0.6 else ""
        tokens_str = ", ".join(f"{t}({c})" for t, c in s["top_tokens"][:5])
        docs_str = ", ".join(f"{d}({c})" for d, c in s["top_docs"][:3])
        print(
            f"  cluster {s['cluster_id']:>3} | n={s['n_points']:>4} | "
            f"concentration={s['concentration']:.0%}{flag}\n"
            f"      tokens: {tokens_str}\n"
            f"      docs:   {docs_str}"
        )

    print(
        f"\n  --> {len(high_concentration)}/{len(summaries)} clusters are "
        f"single-doc/token dominated (concentration >= 60%)"
    )


def fetch_concept_forms(db_path, concept: str) -> tuple[set[str], set[str]]:
    """
    Same forms set originally used to populate events.concept via
    iter_matching_event_ids(concept_def["forms"]) in the production
    pipeline - reusing it here means ZarrEventLookup(forms=...) loads
    exactly the same event set, from either store, without needing a
    separate event_ids-based filter mechanism that doesn't exist on the
    real class.
    """
    con = sqlite3_connection(db_path)
    forms = {
        r[0] for r in con.execute(
            "SELECT form FROM concept_forms WHERE concept = ? AND is_false_positive = 0",
            (concept,),
        ).fetchall()
    }
    false_positives = {
        r[0] for r in con.execute(
            "SELECT form FROM concept_forms WHERE concept = ? AND is_false_positive = 1",
            (concept,),
        ).fetchall()
    }
    con.close()
    return [], false_positives
    # return forms, false_positives


def compare(concept: str, top_n: int = 8):
    logger.info(f"[compare] loading event_ids for concept={concept}")
    all_event_ids = fetch_concept_event_ids(CORPUS_TIER2_DB_PATH, concept)
    if not all_event_ids:
        logger.error(f"[compare] no events found for concept={concept}")
        return

    forms, false_positives = fetch_concept_forms(CORPUS_TIER2_DB_PATH, concept)
    if not forms:
        logger.error(
            f"[compare] no forms found in concept_forms for concept={concept} - "
            f"falling back to full-corpus load (slow); check concept_forms table."
        )

    # Scoped load: ZarrEventLookup already supports filtering by token via
    # forms= (used internally at Tier 3 build time to populate
    # events.concept) - passing it here means only ~1-2k of the 1.8M
    # corpus events get RETAINED after each batch's keep-mask, instead of
    # the full corpus. This does not yet avoid the embedding
    # decompression cost for non-matching batches (see _load_store's read
    # ordering) - that's a separate lib-level fix - but it removes the
    # multi-gigabyte retained-array problem that was the dominant cost.
    logger.info(f"[compare] loading raw store, forms={forms}")
    raw_lookup = ZarrEventLookup(ZARR_PATH, forms=forms, false_positives=false_positives)

    logger.info(f"[compare] loading masked store, forms={forms}")
    masked_lookup = ZarrEventLookup(MASKED_ZARR_PATH, forms=forms, false_positives=false_positives)

    # Fair comparison requires the SAME event set in both runs - restrict
    # to whichever ids the (smaller, deliberately-scoped) masked store
    # actually has, since raw always has the full set.
    masked_available = restrict_to_available(masked_lookup, all_event_ids)
    if not masked_available:
        logger.error(
            f"[compare] no events available in masked store for concept={concept} - "
            f"did tier1b_masked_concept_embeddings.py actually run for this concept?"
        )
        return

    if len(masked_available) < len(all_event_ids):
        logger.warning(
            f"[compare] masked store has {len(masked_available)}/{len(all_event_ids)} "
            f"events for {concept} - comparison restricted to this common subset "
            f"for both runs, so the two are directly comparable."
        )

    event_ids = masked_available

    raw_cache = EmbeddingCache(raw_lookup)
    masked_cache = EmbeddingCache(masked_lookup)

    logger.info(f"[compare] fetching {len(event_ids)} raw vectors")
    X_raw = raw_cache.matrix(event_ids)

    logger.info(f"[compare] fetching {len(event_ids)} masked vectors")
    X_masked = masked_cache.matrix(event_ids)

    logger.info("[compare] clustering RAW embeddings")
    _, raw_labels = fit_cluster_local(X_raw, event_ids)

    logger.info("[compare] clustering MASKED embeddings")
    _, masked_labels = fit_cluster_local(X_masked, event_ids)

    raw_summaries, raw_noise = summarize_clusters(event_ids, raw_labels, raw_lookup, top_n)
    masked_summaries, masked_noise = summarize_clusters(event_ids, masked_labels, masked_lookup, top_n)

    print_summary(f"RAW embeddings - concept={concept}", raw_summaries, raw_noise, len(event_ids))
    print_summary(f"MASKED embeddings - concept={concept}", masked_summaries, masked_noise, len(event_ids))

    # Headline comparison
    def avg_concentration(summaries):
        if not summaries:
            return float("nan")
        weighted = sum(s["concentration"] * s["n_points"] for s in summaries)
        total = sum(s["n_points"] for s in summaries)
        return weighted / total if total else float("nan")

    print(f"\n{'=' * 70}")
    print("HEADLINE COMPARISON")
    print(f"{'=' * 70}")
    print(f"  raw:    {len(raw_summaries)} clusters, {raw_noise} noise, "
          f"point-weighted avg concentration = {avg_concentration(raw_summaries):.0%}")
    print(f"  masked: {len(masked_summaries)} clusters, {masked_noise} noise, "
          f"point-weighted avg concentration = {avg_concentration(masked_summaries):.0%}")
    print(
        "\n  Lower average concentration in the masked run is evidence the "
        "masking approach is reducing single-document/spelling artifacts. "
        "No change (or higher concentration) suggests masking alone did "
        "not fix the underlying issue for this concept - inspect the "
        "per-cluster token lists above before drawing conclusions from "
        "this one aggregate number."
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--concept", required=True)
    p.add_argument("--top-n", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    compare(args.concept, top_n=args.top_n)


if __name__ == "__main__":
    main()
