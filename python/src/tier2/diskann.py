"""
    search_space = SearchSpace( 
        years=(1600, 1700), 
        scale=("local", "medium"), 
    ) 
    
    run_diskann_tier2( 
        concept_name="hair", 
        concept=concept, 
        output_path=output_path, 
        search_space=search_space, 
    )
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from lib.corpus_config import DISKANN_INDEXES_DIR, EVENTSTORE_T1_PATH
from lib.corpus_logging import logger
from retrieval.lazy_year_disk_ann import LazyYearDiskANN
from tier1.observation_store_api import open_observation_lookup
from retrieval.models import SearchSpace

from .analysis import (
    BATCH_SIZE,
    K,
    OVERSAMPLE,
    RRF_K,
    iter_year_concept_batches,
    resolve_concept_positions,
)

DISKANN_DIMENSIONS = 768

_AVAILABLE_SCALES = (
    "local",
    "medium",
    "broad",
)


def run_diskann_tier2(
    *,
    concept_name: str,
    concept: dict,
    output_path: str | Path,
    search_space: SearchSpace | None = None,
    store_path: str | Path = EVENTSTORE_T1_PATH,
    indexes_root: str | Path = DISKANN_INDEXES_DIR,
    top_n: int = K,
    rrf_k: int = RRF_K,
    oversample: int = OVERSAMPLE,
    batch_size: int = BATCH_SIZE,
    false_positives: list[str] | None = None,
) -> Path:
    """
    Run Tier 2 semantic neighbourhood analysis within a SearchSpace.

    Tier 1 Parquet remains the source of truth for observation identity and
    provenance. DiskANN supplies only approximate nearest-neighbour geometry.

    SearchSpace expresses the logical query domain. Physical year and scale
    availability are resolved here, at the backend boundary.

    DiskANN indexes are loaded lazily one publication year at a time. This
    keeps the number of open geometric indexes bounded independently of
    corpus size.
    """
    output_path = Path(output_path)
    lookup = open_observation_lookup( store_path )

    if search_space.years is None:
        candidate_years = lookup.available_years.tolist()
    else:
        start_year, end_year = search_space.years

        candidate_years = [
            int(year)
            for year in lookup.available_years
            if start_year <= int(year) <= end_year
        ]
        
    years = search_space.years

    if years is None:
        candidate_years = lookup.available_years
    else:
        start_year, end_year = years
        candidate_years = [
            year
            for year in lookup.available_years
            if start_year <= int(year) <= end_year
        ]

    year_indexes = LazyYearDiskANN(
        DISKANN_INDEXES_DIR,
        candidate_years,
        dimensions=768,
        num_threads=0,
        search_complexity=100,
        beam_width=2,
        batch_num_threads=0,
        num_nodes_to_cache=0,
    )

    resolved = resolve_concept_positions(
        concept_name=concept_name,
        concept=concept,
        lookup=lookup,
        false_positives=false_positives,
    )

    available_years = {
        int(year)
        for year in lookup.available_years
    }

    available_index_years = {
        int(year)
        for year in year_indexes.available_years()
    }

    searchable_years = (
        available_years
        & available_index_years
    )

    candidate_years = search_space.resolve_years(
        searchable_years,
    )

    scales = search_space.resolve_scales(
        set(_AVAILABLE_SCALES),
    )

    if not scales:
        raise ValueError(
            "SearchSpace resolves to no available scales"
        )

    if not candidate_years:
        logger.warning(
            "[tier2] SearchSpace resolves to no searchable years"
        )

    years_to_process = tuple(
        year
        for year in candidate_years
        if year in resolved["by_year"]
    )

    missing_index_years = sorted(
        set(resolved["by_year"])
        & available_years
        - available_index_years
    )

    for year in missing_index_years:
        logger.warning(
            "[tier2] no DiskANN indexes for year=%s; "
            "seed events will be skipped",
            year,
        )

    logger.info(
        "[tier2] SearchSpace years=%s scales=%s",
        candidate_years,
        scales,
    )

    logger.info(
        "[tier2] query workset: %d years",
        len(years_to_process),
    )

    token_counts: Counter[str] = Counter()
    doc_counts: Counter[str] = Counter()
    window_counts: Counter[tuple[str, int]] = Counter()

    output_events = []

    try:
        for year in years_to_process:
            seed_ids = resolved["by_year"].get(
                year,
                [],
            )

            if not seed_ids:
                continue

            logger.info(
                "[tier2] processing year=%s: %d seed events",
                year,
                len(seed_ids),
            )

            indexes = year_indexes.get(year)

            try:
                for batch in iter_year_concept_batches(
                    concept_name=concept_name,
                    concept=concept,
                    lookup=lookup,
                    indexes=indexes,
                    year=year,
                    scales=scales,
                    top_n=top_n,
                    rrf_k=rrf_k,
                    oversample=oversample,
                    false_positives=false_positives,
                    resolved=resolved,
                    batch_size=batch_size,
                    token_counts=token_counts,
                    doc_counts=doc_counts,
                ):
                    output_events.extend(
                        batch["events"],
                    )
            finally:
                year_indexes.evict(year)

            logger.info(
                "[tier2] completed year=%s",
                year,
            )

    finally:
        year_indexes.close()

    output = {
        "concept": concept_name,
        "search_space": {
            "years": search_space.years,
            "scale": search_space.scale,
        },
        "resolved_years": list(candidate_years),
        "resolved_scales": list(scales),
        "forms": sorted(
            resolved["forms"],
        ),
        "false_positives": sorted(
            resolved["false_positives"],
        ),
        "events": output_events,
        "token_counts": dict(
            token_counts,
        ),
        "doc_counts": dict(
            doc_counts,
        ),
        "window_counts": {
            f"{doc_id}:{window_id}": count
            for (doc_id, window_id), count
            in window_counts.items()
        },
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            output,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(
        "[tier2] wrote %d events to %s",
        len(output_events),
        output_path,
    )

    return output_path
