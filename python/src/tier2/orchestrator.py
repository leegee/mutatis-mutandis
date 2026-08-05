"""
tier2.orchestrator

Orchestrates Tier 2 resource construction, pure analysis, and persistence.

Preserves the original service and CLI entry points:

- service: accepts already-built lookup + indexes
- main / CLI: discovers paths, loads resources, then hands them to the service
"""

from __future__ import annotations

import argparse
import time

from lib.eebo_config import (
    CORPUS_TIER2_DB_PATH,
    CORPUS_TIER2_MASKED_DB_PATH,
    ZARR_PATH,
    MASKED_ZARR_PATH,
    faiss_index_paths,
    discover_index_years,
)
from lib.eebo_db import get_connection
from lib.zarr_event_lookup import ZarrEventLookup
from lib.eebo_logging import logger, setEmit
from lib.concept_resolve import resolve_concepts
from lib.get_processed_concepts import get_processed_concepts

from collections import Counter

from tier2.analysis import (
    K,
    RRF_K,
    OVERSAMPLE,
    BATCH_SIZE,
    build_year_schedule,
    resolve_concept_positions,
    iter_year_concept_batches,
)
from tier2.persistence import (
    initialise_database,
    start_concept,
    write_concept_batch,
    finish_concept,
    enrich_documents,
)
from tier2.resources import LazyYearIndices


def service(
    *,
    lookup,
    indexes,
    concepts_to_run,
    db_path,
    clear: bool = False,
    top_n: int = K,
    rrf_k: int = RRF_K,
    oversample: int = OVERSAMPLE,
    false_positives=None,
    emit=None,
    batch_size: int = BATCH_SIZE,
    commit_every_batch: bool = True,
):
    """
    Reusable entry point for long-lived processes (UI, FastAPI, etc.).

    Expects already-built lookup and FAISS indexes.

    Year-major processing: outer loop is publication year, inner loop is
    the concepts that touch that year. This guarantees at most one
    year's FAISS indices (local/medium/broad) are ever resident,
    regardless of how much concepts' year coverage overlaps.

    Within each (year, concept) pair, seed positions are still streamed
    in batches of `batch_size` so peak memory for neighbour payloads
    stays proportional to `batch_size * top_n` even for high-frequency
    concepts that match hundreds of thousands of events.
    """
    started = time.perf_counter()
    concept_names = [name for name, _ in concepts_to_run]
    logger = setEmit(emit, "[tier2]", {"concepts": concept_names})
    logger.info("[tier2.service] Enter")

    # Attach indexes so any lookup helpers that need them can see them
    if hasattr(lookup, "attach_index"):
        lookup.attach_index(indexes)

    con = initialise_database(db_path, clear=clear)

    # Cheap FAISS-free pass: only year sets, no position payloads.
    years_by_concept, concepts_by_year = build_year_schedule(
        lookup=lookup,
        concepts_to_run=concepts_to_run,
        false_positives=false_positives,
    )

    # Per-concept state that must survive across years (small: counters
    # + metadata). Full position lists are never retained here.
    concept_state = {}
    for concept_name, concept in concepts_to_run:
        if emit:
            emit("concept_start", {"concept": concept_name})
        start_concept(con, concept_name)
        concept_state[concept_name] = {
            "concept": concept,
            "token_counts": Counter(),
            "doc_counts": Counter(),
            "window_counts": Counter(),
            "forms": None,
            "n_events": 0,
            "seed_ids": None,
            "has_events": False,
            "years_done": 0,
            "years_total": len(years_by_concept.get(concept_name, ())),
        }

    # Year-major: load → process every concept that needs this year →
    # evict. At most one year resident at any time.
    for year in sorted(concepts_by_year):
        for concept_name, concept in concepts_by_year[year]:
            state = concept_state[concept_name]

            # Re-resolve on demand (lookup-only, cheap). Only the
            # current year's position list is held while we chunk it.
            resolved = resolve_concept_positions(
                concept_name=concept_name,
                concept=concept,
                lookup=lookup,
                false_positives=false_positives,
            )

            if state["forms"] is None:
                state["forms"] = resolved["forms"]
                state["n_events"] = len(resolved["event_ids"])
                state["seed_ids"] = resolved.get("event_ids_set") or set()

            for item in iter_year_concept_batches(
                concept_name=concept_name,
                concept=concept,
                lookup=lookup,
                indexes=indexes,
                year=year,
                top_n=top_n,
                rrf_k=rrf_k,
                oversample=oversample,
                false_positives=false_positives,
                resolved=resolved,
                batch_size=batch_size,
                token_counts=state["token_counts"],
                doc_counts=state["doc_counts"],
                window_counts=state["window_counts"],
            ):
                state["has_events"] = True
                write_concept_batch(
                    con,
                    concept_name,
                    lookup,
                    item["events"],
                    seed_ids=item["seed_ids"],
                )
                if commit_every_batch:
                    con.commit()

            state["years_done"] += 1
            # resolved / year positions go out of scope here

        # Finished every concept that needed this year → drop indices.
        if hasattr(indexes, "evict"):
            indexes.evict(year)

    # Finalise each concept once all of its years are done.
    processed = 0
    written = 0
    empty_concepts = []

    for concept_name, _ in concepts_to_run:
        state = concept_state[concept_name]
        processed += 1

        if state["has_events"]:
            finish_concept(
                con,
                concept_name,
                state["forms"],
                state["n_events"],
                {
                    "top_tokens": state["token_counts"].most_common(top_n),
                    "top_docs": state["doc_counts"].most_common(top_n),
                    "top_windows": state["window_counts"].most_common(top_n),
                },
            )
            con.commit()
            written += 1
        else:
            empty_concepts.append(concept_name)

        if emit:
            emit("concept_done", {"concept": concept_name})

    logger.info("[tier2.service] Enriching documents")

    # Enrich any newly-inserted document stubs
    try:
        pg = get_connection()
        try:
            enrich_documents(con, pg)
        finally:
            pg.close()
    except Exception as exc:
        logger.warning(f"[tier2] document enrichment skipped: {exc}")

    con.commit()
    con.close()
    elapsed = time.perf_counter() - started

    logger.info("[tier2.service] Done")
    return {
        "generated": "tier2_concept_neighbours",
        "summary": {
            "concepts_requested": len(concepts_to_run),
            "concepts_processed": processed,
            "concepts_written": written,
            "concepts_empty": empty_concepts,
        },
        "elapsed_seconds": round(elapsed, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--concept", default=None)
    parser.add_argument("-m", "--mask", action="store_true")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("-k", "--k", type=int, default=K)
    parser.add_argument("--rrf-k", type=int, default=RRF_K)
    parser.add_argument("--oversample", type=int, default=OVERSAMPLE)
    parser.add_argument("-w", "--max-load-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("-fp", "--false-positives", type=str, default=None)
    parser.add_argument("--json", default=None, help="Path at which to write JSON if required")
    args = parser.parse_args()

    # Paths
    if args.mask:
        zarr_path = MASKED_ZARR_PATH
        db_path = CORPUS_TIER2_MASKED_DB_PATH
    else:
        zarr_path = ZARR_PATH
        db_path = CORPUS_TIER2_DB_PATH

    # Discover FAISS index paths, but don't load them yet.
    #
    # Indices are loaded lazily, per publication year, the first time a
    # concept's events actually touch that year (see LazyYearIndices in
    # tier2.resources). A single-concept or small-batch run then only
    # pays the memory/IO cost for the years it needs, instead of every
    # year present in the corpus.
    years = discover_index_years(args.mask)
    if not years:
        raise RuntimeError("No FAISS indices found")

    index_paths = {
        year: faiss_index_paths(masked=args.mask, year=year)
        for year in years
    }
    indexes = LazyYearIndices(index_paths, workers=args.max_load_workers)

    # Build the Zarr event lookup
    # Restrict forms only when a single concept is requested (keeps memory proportional)
    target_forms = None
    target_fps = None
    if args.concept:
        concept_name = args.concept.upper()
        resolved = dict(resolve_concepts(
            concept=concept_name,
            false_positives=args.false_positives,
        ))
        concept = resolved[concept_name]
        target_forms = set(concept.get("forms", []))
        target_fps = set(concept.get("false_positives", []))

    lookup = ZarrEventLookup(
        zarr_path,
        # forms=target_forms,
        # false_positives=target_fps,
    )

    # Resolve which concepts still need work
    concepts = list(resolve_concepts(
        concept=args.concept,
        false_positives=args.false_positives,
    ))
    logger.info(
        "[tier2] resolved concepts: %d %s",
        len(concepts),
        [c[0] for c in concepts[:20]],
    )

    processed = set() if args.clear else get_processed_concepts(db_path)
    concepts_to_run = [c for c in concepts if c[0] not in processed]

    if not concepts_to_run:
        logger.info("[tier2.main] nothing to write — all concepts already processed")
        return

    # Hand live resources to the service. Per-year FAISS index eviction
    # is now handled automatically inside service(), scheduled against
    # the whole batch, so no per-run flag is needed here.
    output = service(
        lookup=lookup,
        indexes=indexes,
        concepts_to_run=concepts_to_run,
        db_path=db_path,
        clear=args.clear,
        top_n=args.k,
        rrf_k=args.rrf_k,
        oversample=args.oversample,
        false_positives=target_fps,
        emit=None,
    )


    if args.json:
        json.dump( result, open(json, "w") )

    logger.info(f"[tier2.main] Complete → {db_path}")


if __name__ == "__main__":
    main()
