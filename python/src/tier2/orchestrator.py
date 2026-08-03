"""
tier2.orchestrator

Orchestrates Tier 2 resource construction, pure analysis, and persistence.

Preserves the original service and CLI entry points:

- run_tier2_service: accepts already-built lookup + indexes
- main / CLI: discovers paths, loads resources, then hands them to the service
"""

from __future__ import annotations

import argparse

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
    run_tier2_core,
)
from tier2.persistence import (
    initialise_database,
    write_concept,
    enrich_documents,
)
from tier2.resources import load_indices


def run_tier2_service(
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
):
    """
    Reusable entry point for long-lived processes (UI, FastAPI, etc.).

    Expects already-built lookup and FAISS indexes.
    Calls core, then writes results to SQLite.
    """
    concept_names = [name for name, _ in concepts_to_run]
    logger = setEmit(emit, "[tier2]", {"concepts": concept_names})
    logger.info("[tier2.run_tier2_service] Enter")

    # Attach indexes so any lookup helpers that need them can see them
    if hasattr(lookup, "attach_index"):
        lookup.attach_index(indexes)

    con = initialise_database(db_path, clear=clear)

    output = run_tier2_core(
        lookup=lookup,
        indexes=indexes,
        concepts_to_run=concepts_to_run,
        top_n=top_n,
        rrf_k=rrf_k,
        oversample=oversample,
        false_positives=false_positives,
        emit=emit,
    )

    logger.info("[tier2.run_tier2_service] Writing results")
    for concept_name, data in output.items():
        if data.get("empty"):
            continue
        write_concept(con, data, lookup)

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

    logger.info("[tier2.run_tier2_service] Done")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--concept", default=None)
    parser.add_argument("-m", "--mask", action="store_true")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("-k", "--k", type=int, default=K)
    parser.add_argument("--rrf-k", type=int, default=RRF_K)
    parser.add_argument("--oversample", type=int, default=OVERSAMPLE)
    parser.add_argument("-w", "--max-load-workers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("-fp", "--false-positives", type=str, default=None)
    args = parser.parse_args()

    # Paths
    if args.mask:
        zarr_path = MASKED_ZARR_PATH
        db_path = CORPUS_TIER2_MASKED_DB_PATH
    else:
        zarr_path = ZARR_PATH
        db_path = CORPUS_TIER2_DB_PATH

    # Discover & load FAISS indexes (resource construction lives here)
    years = discover_index_years(args.mask)
    if not years:
        raise RuntimeError("No FAISS indices found")

    index_paths = {
        year: faiss_index_paths(masked=args.mask, year=year)
        for year in years
    }
    indexes = load_indices(index_paths, workers=args.max_load_workers)

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

    # Hand live resources to the service
    run_tier2_service(
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

    logger.info(f"[tier2.main] Complete → {db_path}")


if __name__ == "__main__":
    main()
