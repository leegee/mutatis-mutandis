#!/usr/bin/env python
"""
tier2/tier2_analyse.py

Tier 2 semantic neighbourhood construction — analysis only.

Responsibilities:

    Tier 1 Zarr observations
            |
            v
    yearly FAISS retrieval geometry
            |
            v
    in-memory neighbourhood result (no persistence)

This module is pure analysis. It never opens a SQLite connection and
never writes anything to disk except, optionally, a JSON dump when run
from its own CLI for inspection/debugging. Persistence lives in
tier2_create_populate.py.

Important invariants:

- event_id is the atomic corpus occurrence.
- FAISS ids are retrieval mechanisms only.
- RRF scores are ranking scores, not distances.
- Retrieval is publication-year scoped.
- Lexical forms are query provenance, not semantic membership.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from lib.corpus_config import (
    ZARR_PATH,
    MASKED_ZARR_PATH,
)

from lib.corpus_faiss import (
    CorpusFaissIndex,
    multiscale_search,
)

from lib.zarr_event_lookup import ZarrEventLookup
from lib.corpus_logging import logger, setEmit
from lib.concept_resolve import resolve_concepts


K = 60
RRF_K = 60
OVERSAMPLE = 5
_NO_WPOS = -1


def analyse_concept(
    *,
    concept_name,
    concept,
    lookup,
    indexes,
    top_n=K,
    rrf_k=RRF_K,
    oversample=OVERSAMPLE,
    false_positives=None,
):
    forms = {
        f.lower()
        for f in concept.get("forms", [])
    }

    false_positives = {
        x.lower()
        for x in (false_positives or [])
    }

    logger.info( f"[tier2] {concept_name} forms: {sorted(forms)[:50]}" )

    event_ids = lookup.find_matching_event_ids(
        forms,
        false_positives,
    )

    logger.info( f"[tier2] {concept_name}: {len(event_ids)} events" )

    if not event_ids:
        return {
            "concept": concept_name,
            "empty": True,
        }


    positions = np.asarray(
        [
            lookup.get_pos(eid)
            for eid in event_ids
        ],
        dtype=np.int64,
    )

    seed_positions = {
        (
            str(lookup.corpus[pos]),
            str(lookup.doc_id[pos]),
            int(lookup.token_idx[pos]),
        )
        for pos in positions
    }

    # Group by publication year.
    #
    # This is deliberately based on Tier 1 Zarr metadata.
    # Document metadata may be incomplete or normalised differently.

    by_year = {}
    for pos in positions:
        year = int( lookup.pub_year[pos] )

        by_year.setdefault(
            year,
            []
        ).append(pos)

    fused = {}
    for year, year_positions in by_year.items():
        result = multiscale_search(
            indexes,
            lookup,
            np.asarray(
                year_positions,
                dtype=np.int64,
            ),
            top_n,
            pub_year=year,
            rrf_k=rrf_k,
            oversample=oversample,
        )

        for pos, neighbours in zip(
            year_positions,
            result,
        ):
            fused[ int(lookup.event_id[pos]) ] = neighbours

    token_counts = Counter()
    doc_counts = Counter()
    window_counts = Counter()

    output_events = []
    for pos in positions:
        event_id = int( lookup.event_id[pos] )


        neighbours_out = []
        for item in fused[event_id]:
            neighbour_id = item["event_id"]
            if neighbour_id == event_id:
                continue

            npos = lookup.get_pos(neighbour_id)

            candidate_key = (
                str(lookup.corpus[npos]),
                str(lookup.doc_id[npos]),
                int(lookup.token_idx[npos]),
            )

            # Do not allow overlapping-window observations of the seed occurrence
            if candidate_key in seed_positions:
                continue

            token = str(lookup.token[npos])
            token_lower = token.lower()

            if token_lower in false_positives:
                continue

            # Do not allow the seed lexical forms as semantic neighbours
            if token_lower in forms:
                continue

            doc_id = str( lookup.doc_id[npos] )
            window_id = int( lookup.window_id[npos] )
            wpos = int( lookup.window_token_pos[npos] )
            token_counts[token] += 1
            doc_counts[doc_id] += 1

            window_counts[
                (
                    doc_id,
                    window_id,
                )
            ] += 1

            neighbours_out.append(
                {
                    "event_id": neighbour_id,
                    "vector_id": int( lookup.vector_id[npos] ),
                    "token": token,
                    "corpus": str(lookup.corpus[npos]),
                    "doc_id": doc_id,
                    "pub_year": int( lookup.pub_year[npos] ),
                    "token_idx": int( lookup.token_idx[npos] ),
                    "window_id": window_id,
                    "window_token_pos":
                        None
                        if wpos == _NO_WPOS
                        else wpos,
                    "score": item["rrf_score"],
                    "score_local": item.get( "score_local" ),
                    "score_medium": item.get( "score_medium" ),
                    "score_broad": item.get( "score_broad" ),
                    "depth": 1,
                    "via_event_id": None,
                }
            )

        output_events.append(
            {
                "event_id": event_id,
                "vector_id": int( lookup.vector_id[pos] ),
                "token": str( lookup.token[pos] ),
                "corpus": str(lookup.corpus[pos]),
                "doc_id": str( lookup.doc_id[pos] ),
                "pub_year": int( lookup.pub_year[pos] ),
                "token_idx": int( lookup.token_idx[pos] ),
                "window_id": int( lookup.window_id[pos] ),
                "window_token_pos":
                    None
                    if int(
                        lookup.window_token_pos[pos]
                    ) == _NO_WPOS
                    else int(
                        lookup.window_token_pos[pos]
                    ),
                "neighbours": neighbours_out, }
        )

    return {
        "concept": concept_name,
        "forms": forms,
        "n_events": len(event_ids),
        "events": output_events,
        "aggregate":
            {
                "top_tokens": token_counts.most_common(top_n),
                "top_docs": doc_counts.most_common(top_n),
                "top_windows": window_counts.most_common(top_n),
            },
    }


# Core is pure analysis, no I/O.
#
# This is the service-style entrypoint: it accepts already-constructed
# resources (lookup, indexes, concepts) and returns a pure result dict.
# No database writes, no index loading, no side effects beyond logging.
# Long-lived processes (UI, FastAPI, the populate service, etc.) should
# call this directly rather than going through the CLI below.
def run_tier2_core(
    *,
    lookup,
    indexes,
    concepts_to_run,
    top_n: int = K,
    rrf_k: int = RRF_K,
    oversample: int = OVERSAMPLE,
    false_positives=None,
    emit=None,
):
    """
    Heart of Tier 2: neighbourhood retrieval for every requested concept.
    """
    logger.info("[tier2.run_tier2_core] Enter")
    output = {}

    for concept_name, concept in concepts_to_run:
        if emit:
            emit("concept_start", {"concept": concept_name})

        output[concept_name] = analyse_concept(
            concept_name=concept_name,
            concept=concept,
            lookup=lookup,
            indexes=indexes,
            top_n=top_n,
            rrf_k=rrf_k,
            oversample=oversample,
            false_positives=false_positives,
        )

        if emit:
            emit("concept_done", {"concept": concept_name})

    logger.info("[tier2.run_tier2_core] Leave")
    return output


def build_resources(*, concept=None, mask=False, false_positives=None, max_load_workers=6):
    """
    Shared resource construction used by this module's CLI and by callers
    (e.g. the create-populate CLI) that want to run analysis end-to-end
    from raw args without duplicating index/lookup setup.
    """
    zarr_path = MASKED_ZARR_PATH if mask else ZARR_PATH

    indexes = CorpusFaissIndex.load_all(
        masked=mask,
        workers=max_load_workers,
    )

    if not indexes:
        raise RuntimeError("No FAISS indices found")

    target_fps = None
    if concept:
        concept_name = concept.upper()
        resolved = dict(resolve_concepts(
            concept=concept_name,
            false_positives=false_positives,
        ))
        target_fps = set(resolved[concept_name].get("false_positives", []))

    lookup = ZarrEventLookup(zarr_path)

    concepts = list(resolve_concepts(
        concept=concept,
        false_positives=false_positives,
    ))

    return lookup, indexes, concepts, target_fps


def _json_default(obj):
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)!r}")


# CLI — builds resources and runs analysis only. Useful standalone for
# inspecting/testing retrieval geometry without touching any database.
# Optionally dumps the raw result to JSON via --out for later consumption
# by tier2_create_populate.py's CLI (--input).
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--concept", default=None)
    parser.add_argument("-m", "--mask", action="store_true")
    parser.add_argument("-k", "--k", type=int, default=K)
    parser.add_argument("--rrf-k", type=int, default=RRF_K)
    parser.add_argument("--oversample", type=int, default=OVERSAMPLE)
    parser.add_argument("-w", "--max-load-workers", type=int, default=6)
    parser.add_argument("-fp", "--false-positives", type=str, default=None)
    parser.add_argument("-o", "--out", type=str, default=None,
                         help="Optional path to dump raw analysis output as JSON")
    args = parser.parse_args()

    lookup, indexes, concepts, target_fps = build_resources(
        concept=args.concept,
        mask=args.mask,
        false_positives=args.false_positives,
        max_load_workers=args.max_load_workers,
    )

    logger.info(
        "[tier2_analyse] resolved concepts: %d %s",
        len(concepts),
        [c[0] for c in concepts[:20]],
    )

    output = run_tier2_core(
        lookup=lookup,
        indexes=indexes,
        concepts_to_run=concepts,
        top_n=args.k,
        rrf_k=args.rrf_k,
        oversample=args.oversample,
        false_positives=target_fps,
    )

    for concept_name, data in output.items():
        if data.get("empty"):
            logger.info(f"[tier2_analyse] {concept_name}: no matching events")
        else:
            logger.info(f"[tier2_analyse] {concept_name}: {data['n_events']} events analysed")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(output, f, default=_json_default, indent=2)
        logger.info(f"[tier2_analyse] wrote raw analysis output → {args.out}")


if __name__ == "__main__":
    main()
