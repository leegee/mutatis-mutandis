"""
tier2.orchestrator

Orchestrates Tier 2 resource construction, pure analysis, and persistence.

Preserves the original service and CLI entry points:

- service: accepts already-built lookup + indexes (backend-agnostic)
- main / CLI: opens an ObservationLookup via the observation-store factory
  (--store-backend zarr|parquet, --store PATH), loads FAISS indexes, then
  hands them to the service

Analysis and persistence do not depend on Zarr; only resource loading does.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from collections import Counter

from lib.corpus_config import (
    CORPUS_TIER2_DB_PATH,
    CORPUS_TIER2_MASKED_DB_PATH,
    EVENTSTORE_T1_PATH,
    MASKED_EVENTSTORE_T1_PATH,
    faiss_index_paths,
    discover_index_years,
)
from lib.corpus_db import get_connection
from lib.corpus_logging import logger, setEmit
from lib.concept_resolve import resolve_concepts
from lib.get_processed_concepts import get_processed_concepts

from tier1.observation_store_api import (
    open_observation_lookup,
    default_store_path
)

# Register observation-store backends (import side-effect).
import lib.zarr_observation_backend  # noqa: F401
import lib.parquet_observation_backend  # noqa: F401

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
    create_indexes,
    restore_reader_pragmas,
    start_concept,
    write_concept_batch,
    finish_concept,
    enrich_documents,
    initialise_pg_stage,
    start_concept_pg,
    write_concept_batch_pg,
    finish_concept_pg,
    dump_pg_stage_to_sqlite,
)

from tier2.resources import LazyYearIndices



DEFAULT_STORE_BACKEND = "parquet"



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
    stage_pg: bool = False,
    publish_sqlite: bool = True,
    search_backend: str = "faiss",
    exact_store=None,
    exact_workers: int = 1,
    exact_pool: str = "thread",
    exact_shards=None,
):
    """
    Reusable entry point for long-lived processes (UI, FastAPI, etc.).

    Expects already-built lookup. FAISS indexes are required for
    search_backend='faiss'; exact_store (Parquet root) is required for
    search_backend='exact'.

    Year-major processing with resolve-once. Write path:

      stage_pg=False (default) — stream into SQLite (tables only, bulk
        PRAGMAs); create indexes at the end.
      stage_pg=True — stream into Postgres UNLOGGED stage tables.
        publish_sqlite=True (default) also dumps to db_path when done.
        publish_sqlite=False leaves the stage tables alive for Tier 3
        (call dump_pg_stage_to_sqlite later to publish).
    """
    started = time.perf_counter()
    concept_names = [name for name, _ in concepts_to_run]
    logger = setEmit(emit, "[tier2]", {"concepts": concept_names})
    logger.info(
        "[tier2.service] Enter (stage_pg=%s search_backend=%s)",
        stage_pg,
        search_backend,
    )

    if search_backend == "faiss" and indexes is not None and hasattr(lookup, "attach_index"):
        lookup.attach_index(indexes)

    pg = None
    con = None

    if stage_pg:
        pg = get_connection()
        initialise_pg_stage(pg)
        start_fn = lambda name: start_concept_pg(pg, name)
        write_fn = lambda name, events, seed_ids: write_concept_batch_pg(
            pg, name, lookup, events, seed_ids=seed_ids
        )
        finish_fn = lambda name, forms, n_events, aggregate: finish_concept_pg(
            pg, name, forms, n_events, aggregate
        )
        commit_fn = (lambda: pg.commit()) if hasattr(pg, "commit") else (lambda: None)
    else:
        con = initialise_database(db_path, clear=clear)
        start_fn = lambda name: start_concept(con, name)
        write_fn = lambda name, events, seed_ids: write_concept_batch(
            con, name, lookup, events, seed_ids=seed_ids
        )
        finish_fn = lambda name, forms, n_events, aggregate: finish_concept(
            con, name, forms, n_events, aggregate
        )
        commit_fn = lambda: con.commit()

    concept_state = {}
    concepts_by_year = {}

    for concept_name, concept in concepts_to_run:
        if emit:
            emit("concept_start", {"concept": concept_name})

        resolved = resolve_concept_positions(
            concept_name=concept_name,
            concept=concept,
            lookup=lookup,
            false_positives=false_positives,
        )

        start_fn(concept_name)

        concept_state[concept_name] = {
            "concept": concept,
            "token_counts": Counter(),
            "doc_counts": Counter(),
            "window_counts": Counter(),
            "forms": resolved["forms"],
            "false_positives": resolved["false_positives"],
            "n_events": (
                len(resolved["event_ids"])
                if resolved["event_ids"] is not None
                else 0
            ),
            "seed_ids": resolved.get("event_ids_set") or set(),
            "by_year": resolved["by_year"],
            "has_events": False,
        }

        for year in resolved["by_year"]:
            concepts_by_year.setdefault(year, []).append(concept_name)

        del resolved

    for year in sorted(concepts_by_year):
        for concept_name in concepts_by_year[year]:
            state = concept_state[concept_name]
            resolved = {
                "forms": state["forms"],
                "false_positives": state["false_positives"],
                "event_ids_set": state["seed_ids"],
                "by_year": state["by_year"],
            }

            for item in iter_year_concept_batches(
                concept_name=concept_name,
                concept=state["concept"],
                lookup=lookup,
                indexes=indexes,
                year=year,
                top_n=top_n,
                rrf_k=rrf_k,
                oversample=oversample,
                false_positives=state["false_positives"],
                resolved=resolved,
                batch_size=batch_size,
                token_counts=state["token_counts"],
                doc_counts=state["doc_counts"],
                window_counts=state["window_counts"],
                search_backend=search_backend,
                exact_store=exact_store,
                exact_workers=exact_workers,
                exact_pool=exact_pool,
                exact_shards=exact_shards,
            ):
                state["has_events"] = True
                write_fn(concept_name, item["events"], item["seed_ids"])
                if commit_every_batch:
                    commit_fn()

        if indexes is not None and hasattr(indexes, "evict"):
            indexes.evict(year)

    processed = 0
    written = 0
    empty_concepts = []

    for concept_name, _ in concepts_to_run:
        state = concept_state[concept_name]
        processed += 1

        if state["has_events"]:
            finish_fn(
                concept_name,
                state["forms"],
                state["n_events"],
                {
                    "top_tokens": state["token_counts"].most_common(top_n),
                    "top_docs": state["doc_counts"].most_common(top_n),
                    "top_windows": state["window_counts"].most_common(top_n),
                },
            )
            commit_fn()
            written += 1
        else:
            empty_concepts.append(concept_name)

        if emit:
            emit("concept_done", {"concept": concept_name})

    if stage_pg:
        commit_fn()
        if publish_sqlite:
            dump_pg_stage_to_sqlite(pg, db_path, clear=clear)
            logger.info("[tier2.service] Enriching documents")
            try:
                import sqlite3
                sqlite_path = (
                    db_path if isinstance(db_path, (str, bytes)) else str(db_path)
                )
                sqlite_con = sqlite3.connect(sqlite_path)
                try:
                    enrich_documents(sqlite_con, pg)
                    sqlite_con.commit()
                finally:
                    sqlite_con.close()
            except Exception as exc:
                logger.warning(f"[tier2] document enrichment skipped: {exc}")
            if hasattr(pg, "close"):
                pg.close()
        else:
            logger.info(
                "[tier2.service] publish_sqlite=False — "
                "leaving tier2_stage.* alive for Tier 3"
            )
            # Do not close pg: caller / process may keep using the stage.
            # Connection still closed if we own a short-lived CLI run with
            # no further work; CLI passes publish_sqlite explicitly.
            if hasattr(pg, "close"):
                pg.close()
    else:
        logger.info("[tier2.service] Enriching documents")
        try:
            pg_src = get_connection()
            try:
                enrich_documents(con, pg_src)
            finally:
                pg_src.close()
        except Exception as exc:
            logger.warning(f"[tier2] document enrichment skipped: {exc}")

        con.commit()
        create_indexes(con)
        restore_reader_pragmas(con)
        con.commit()
        con.close()

    elapsed = time.perf_counter() - started
    logger.info(f"[tier2.service] Done in {elapsed}")
    return {
        "generated": "tier2_concept_neighbours",
        "summary": {
            "concepts_requested": len(concepts_to_run),
            "concepts_processed": processed,
            "concepts_written": written,
            "concepts_empty": empty_concepts,
            "stage_pg": stage_pg,
            "publish_sqlite": publish_sqlite if stage_pg else False,
        },
        "elapsed_seconds": round(elapsed, 3),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Tier 2: concept neighbourhood analysis over the Tier 1 observation store"
    )
    parser.add_argument("-c", "--concept", default=None)
    parser.add_argument("-m", "--mask", action="store_true")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("-k", "--k", type=int, default=K)
    parser.add_argument("--rrf-k", type=int, default=RRF_K)
    parser.add_argument("--oversample", type=int, default=OVERSAMPLE)
    parser.add_argument("-w", "--max-load-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument(
        "--stage-pg",
        action="store_true",
        help="Write via Postgres UNLOGGED stage",
    )
    parser.add_argument(
        "--no-publish-sqlite",
        action="store_true",
        help="With --stage-pg, leave stage tables alive (skip SQLite dump)",
    )
    parser.add_argument("-fp", "--false-positives", type=str, default=None)
    parser.add_argument(
        "--json",
        default=None,
        help="Path at which to write JSON if required",
    )
    # Observation store backend (aligned with Tier 1 --store-backend).
    parser.add_argument(
        "--store-backend",
        choices=["zarr", "parquet"],
        default=DEFAULT_STORE_BACKEND,
        help=f"Observation store backend (default: {DEFAULT_STORE_BACKEND})",
    )
    parser.add_argument(
        "--store",
        type=str,
        default=None,
        help="Override observation store root path (default depends on --store-backend)",
    )
    # Neighbour search backend: FAISS (in-RAM) vs exact Parquet scan.
    parser.add_argument(
        "--search-backend",
        choices=["faiss", "exact"],
        default="faiss",
        help="Neighbour retrieval: faiss (default) or exact out-of-core Parquet scan",
    )
    parser.add_argument(
        "--exact-workers",
        type=int,
        default=4,
        help="Parallel shard workers for search_backend=exact (default 4)",
    )
    parser.add_argument(
        "--exact-pool",
        choices=["thread", "process"],
        default="thread",
        help="Executor pool for exact shard scoring (default thread)",
    )
    parser.add_argument(
        "--exact-store",
        type=str,
        default=None,
        help=(
            "Parquet root for exact search (default: --store when "
            "store-backend=parquet, else sibling tier1_parquet)"
        ),
    )
    args = parser.parse_args()

    store_backend = args.store_backend
    store_path = (
        Path(args.store)
        if args.store
        else default_store_path(store_backend, masked=args.mask)
    )
    db_path = (
        CORPUS_TIER2_MASKED_DB_PATH if args.mask else CORPUS_TIER2_DB_PATH
    )
    search_backend = args.search_backend

    # Exact search needs a Parquet lake. Default to --store when already
    # parquet; otherwise the conventional sibling tier1_parquet path.
    exact_store = None
    exact_shards = None
    if search_backend == "exact":
        if args.exact_store:
            exact_store = Path(args.exact_store)
        elif store_backend == "parquet":
            exact_store = store_path
        else:
            exact_store = default_store_path("parquet", masked=args.mask)
        from exact_knn_search import discover_shards

        exact_shards = discover_shards(exact_store)
        logger.info(
            "[tier2] exact search store=%s shards=%d rows=%s",
            exact_store,
            len(exact_shards),
            f"{sum(s.n_rows for s in exact_shards):,}",
        )

    # FAISS indices only required for search_backend=faiss.
    indexes = None
    if search_backend == "faiss":
        years = discover_index_years(args.mask)
        if not years:
            raise RuntimeError("No FAISS indices found")
        index_paths = {
            year: faiss_index_paths(masked=args.mask, year=year)
            for year in years
        }
        indexes = LazyYearIndices(index_paths, workers=args.max_load_workers)

    # Restrict forms only when a single concept is requested so backends
    # that support construct-time filtering (Parquet) keep memory proportional
    # to the concept rather than the corpus. Zarr ignores these kwargs today.
    target_forms = None
    target_fps = None
    if args.concept:
        concept_name = args.concept.upper()
        resolved = dict(
            resolve_concepts(
                concept=concept_name,
                false_positives=args.false_positives,
            )
        )
        concept = resolved[concept_name]
        target_forms = set(concept.get("forms", []))
        target_fps = set(concept.get("false_positives", []))

    logger.info(
        "[tier2] store_backend=%s path=%s search_backend=%s forms=%s",
        store_backend,
        store_path,
        search_backend,
        sorted(target_forms) if target_forms else "(all)",
    )

    lookup = open_observation_lookup(
        store_backend,
        store_path,
        forms=target_forms,
        false_positives=target_fps,
    )

    # Resolve which concepts still need work
    concepts = list(
        resolve_concepts(
            concept=args.concept,
            false_positives=args.false_positives,
        )
    )
    logger.info(
        "[tier2] resolved concepts: %d %s",
        len(concepts),
        [c[0] for c in concepts[:20]],
    )

    processed = set() if args.clear else get_processed_concepts(db_path)
    concepts_to_run = [c for c in concepts if c[0] not in processed]

    if not concepts_to_run:
        logger.info(
            "[tier2.main] nothing to write — all concepts already processed"
        )
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
        stage_pg=args.stage_pg,
        publish_sqlite=not args.no_publish_sqlite,
        search_backend=search_backend,
        exact_store=exact_store,
        exact_workers=args.exact_workers,
        exact_pool=args.exact_pool,
        exact_shards=exact_shards,
    )

    if args.json:
        import json

        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(
        "[tier2.main] Complete store_backend=%s search_backend=%s path=%s → %s",
        store_backend,
        search_backend,
        store_path,
        db_path,
    )


if __name__ == "__main__":
    main()
