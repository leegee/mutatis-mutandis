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

from tier2.analysis import (
    K,
    RRF_K,
    OVERSAMPLE,
    BATCH_SIZE,
    build_eviction_schedule,
    iter_concept_batches,
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

    Streams each concept's analysis in bounded-size chunks, writing and
    committing each chunk to SQLite as it's produced (see
    tier2.analysis.iter_concept_batches). Peak memory stays roughly
    proportional to `batch_size * top_n`, regardless of how many events
    an individual concept matches or how many concepts are in the batch
    — this matters because common words (KING, LAW, PARLIAMENT...) can
    match hundreds of thousands of events in a large corpus, and would
    otherwise dominate memory on their own, independent of batch size.
    """
    started = time.perf_counter()
    concept_names = [name for name, _ in concepts_to_run]
    logger = setEmit(emit, "[tier2]", {"concepts": concept_names})
    logger.info("[tier2.service] Enter")

    # Attach indexes so any lookup helpers that need them can see them
    if hasattr(lookup, "attach_index"):
        lookup.attach_index(indexes)

    con = initialise_database(db_path, clear=clear)

    # One cheap, FAISS-free pass over the whole batch: find out which
    # publication years each concept touches, and the last concept in
    # the run that needs each year. Deliberately lightweight — it does
    # NOT retain each concept's positions/event_ids, only the small set
    # of years each one touches (see build_eviction_schedule). This lets
    # the loop below evict a year's FAISS indices the moment nothing
    # left in the batch needs them, without holding every concept's
    # full matched-event data in memory for the whole run.
    years_by_concept, last_use = build_eviction_schedule(
        lookup=lookup,
        concepts_to_run=concepts_to_run,
        false_positives=false_positives,
    )

    processed = 0
    written = 0
    empty_concepts = []

    for i, (concept_name, concept) in enumerate(concepts_to_run):
        if emit:
            emit("concept_start", {"concept": concept_name})

        evict_after_years = {
            year
            for year in years_by_concept[concept_name]
            if last_use.get(year) == i
        }

        start_concept(con, concept_name)

        has_events = False
        final_meta = None

        # resolved=None (the default) — iter_concept_batches resolves
        # this concept's positions/event_ids itself, on demand. It's a
        # cheap, lookup-only recomputation (already done once, lightly,
        # in build_eviction_schedule above), and keeping it scoped to
        # a single concept at a time is the point: only one concept's
        # matched-event data is ever alive in memory, not the whole
        # batch's.
        for item in iter_concept_batches(
            concept_name=concept_name,
            concept=concept,
            lookup=lookup,
            indexes=indexes,
            top_n=top_n,
            rrf_k=rrf_k,
            oversample=oversample,
            false_positives=false_positives,
            batch_size=batch_size,
            evict_after_years=evict_after_years,
        ):
            kind = item["type"]

            if kind == "batch":
                has_events = True
                write_concept_batch(
                    con,
                    concept_name,
                    lookup,
                    item["events"],
                    seed_ids=item["seed_ids"],
                )
                if commit_every_batch:
                    con.commit()
                # item["events"] (and everything it references) goes
                # out of scope here — nothing from this chunk persists
                # past this iteration.

            elif kind == "final":
                final_meta = item

            elif kind == "empty":
                pass

        processed += 1

        if has_events and final_meta is not None:
            finish_concept(
                con,
                concept_name,
                final_meta["forms"],
                final_meta["n_events"],
                final_meta["aggregate"],
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
