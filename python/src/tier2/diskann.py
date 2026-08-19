from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from lib.corpus_config import DISKANN_INDEXES_DIR, EVENTSTORE_T1_PATH
from lib.corpus_logging import logger
from retrieval.lazy_year_disk_ann import LazyYearDiskANN
from retrieval.observation_store_api import open_observation_lookup

from .analysis import (
    BATCH_SIZE,
    K,
    OVERSAMPLE,
    RRF_K,
    iter_year_concept_batches,
    resolve_concept_positions,
)

DISKANN_DIMENSIONS = 768


def run_diskann_tier2(
    *,
    concept_name: str,
    concept: dict,
    output_path: str | Path,
    store_path: str | Path = EVENTSTORE_T1_PATH,
    indexes_root: str | Path = DISKANN_INDEXES_DIR,
    years: tuple[int, int] | None = None,
    top_n: int = K,
    rrf_k: int = RRF_K,
    oversample: int = OVERSAMPLE,
    batch_size: int = BATCH_SIZE,
    false_positives: list[str] | None = None,
) -> Path:
    """
    Run Tier 2 semantic neighbourhood analysis using year-local DiskANN.

    Tier 1 Parquet remains the source of truth for observation identity and
    provenance. DiskANN supplies only approximate nearest-neighbour geometry.

    DiskANN indexes are loaded lazily one publication year at a time. This
    keeps the number of open geometric indexes bounded independently of
    corpus size.
    """
    output_path = Path(output_path)

    lookup = open_observation_lookup(
        store_path,
    )

    year_indexes = LazyYearDiskANN(
        indexes_root,
        dimensions=DISKANN_DIMENSIONS,
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

    seed_years = set(
        resolved["by_year"],
    )

    available_years = {
        int(year)
        for year in lookup.available_years
    }

    available_index_years = set(
        year_indexes.available_years(),
    )

    years_to_process = sorted(
        seed_years
        & available_years
        & available_index_years
    )

    if years is not None:
        start, end = years

        years_to_process = [
            year
            for year in years_to_process
            if start <= year <= end
        ]

    missing_index_years = sorted(
        seed_years
        & available_years
        - available_index_years
    )

    for year in missing_index_years:
        logger.warning(
            "[tier2] no DiskANN indexes for year=%s; "
            "seed events will be skipped",
            year,
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
                    top_n=top_n,
                    rrf_k=rrf_k,
                    oversample=oversample,
                    false_positives=false_positives,
                    resolved=resolved,
                    batch_size=batch_size,
                    token_counts=token_counts,
                    doc_counts=doc_counts,
                    window_counts=window_counts,
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
