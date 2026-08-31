"""
tier2/analysis.py

The current within-year restriction is deliberate. Tier 2 is not itself intended to perform the final diachronic analysis.

The immediate objective is to establish **local semantic neighbourhoods** around a concept within a temporally coherent corpus slice. Those neighbourhoods can then be compared across broader chronological buckets—initially perhaps 50-year periods—and subsequently aligned with Modern BERT.

The diachronic process will therefore work backwards from the apparently polysamous or semantically divergent results: identify observations whose neighbourhoods differ substantially across periods, then trace those apparent semantic developments back through progressively narrower historical slices and the underlying corpus evidence.

In that architecture, Tier 2 remains a retrieval layer. Its job is to establish reliable semantic neighbourhoods and preserve the event IDs and provenance needed for subsequent diachronic analysis. The later tiers perform the actual temporal alignment, comparison, and investigation of semantic drift.


"""

from __future__ import annotations

import numpy as np

from lib.corpus_logging import logger
from tier1.observation_store_api import SCALES
from retrieval.models import INVALID_EVENT_ID
from retrieval.lance_search import multiscale_search

K = 60
RRF_K = 60
OVERSAMPLE = 5
BATCH_SIZE = 128

_NO_WPOS = -1


def resolve_concept_positions(
    *,
    concept_name,
    concept,
    lookup,
    false_positives=None,
):
    forms = {
        str(form).lower()
        for form in concept.get("forms", [])
    }

    false_positives = {
        str(value).lower()
        for value in (false_positives or [])
    }

    logger.info( "[tier2] %s forms: %s", concept_name, sorted(forms)[:50], )

    event_ids = lookup.find_matching_event_ids(
        forms,
        false_positives,
    )

    event_ids = [
        int(event_id)
        for event_id in event_ids
    ]

    logger.info( "[tier2] %s: %d seed events", concept_name, len(event_ids), )

    by_year = {}

    for event_id in event_ids:
        metadata = lookup.get_event_metadata(
            event_id
        )
        year = int(metadata["pub_year"])
        by_year.setdefault(
            year,
            [],
        ).append(event_id)

    return {
        "forms": forms,
        "false_positives": false_positives,
        "event_ids": event_ids,
        "event_ids_set": set(event_ids),
        "by_year": by_year,
    }



def _metadata_for_event(
    lookup,
    event_id,
):
    return lookup.get_event_metadata(
        int(event_id)
    )


def _window_metadata(
    metadata,
    scale,
):
    window_id = metadata.get(
        f"{scale}_window_id"
    )
    token_pos = metadata.get(
        f"{scale}_window_token_pos"
    )

    if window_id is not None:
        window_id = int(window_id)

    if token_pos is not None:
        token_pos = int(token_pos)

        if token_pos == _NO_WPOS:
            token_pos = None

    return window_id, token_pos


def _build_batch_events(
    *,
    seed_event_ids,
    neighbours,
    lookup,
    false_positives,
):
    output = []

    for seed_event_id, seed_neighbours in zip(
        seed_event_ids,
        neighbours,
    ):
        seed_event_id = int(seed_event_id)

        seed_metadata = _metadata_for_event(
            lookup,
            seed_event_id,
        )

        neighbours_out = []

        for item in seed_neighbours:
            neighbour_id = int(
                item["event_id"]
            )

            if neighbour_id == seed_event_id:
                continue

            metadata = _metadata_for_event(
                lookup,
                neighbour_id,
            )

            token = str(
                metadata["token"]
            )

            if token.lower() in false_positives:
                continue

            (
                local_window_id,
                local_window_token_pos,
            ) = _window_metadata(
                metadata,
                "local",
            )

            (
                medium_window_id,
                medium_window_token_pos,
            ) = _window_metadata(
                metadata,
                "medium",
            )

            (
                broad_window_id,
                broad_window_token_pos,
            ) = _window_metadata(
                metadata,
                "broad",
            )

            neighbours_out.append(
                {
                    "event_id": neighbour_id,
                    "token": token,
                    "doc_id": str(
                        metadata["doc_id"]
                    ),
                    "pub_year": int(
                        metadata["pub_year"]
                    ),
                    "token_idx": int(
                        metadata["token_idx"]
                    ),
                    "local_window_id": (
                        local_window_id
                    ),
                    "local_window_token_pos": (
                        local_window_token_pos
                    ),
                    "medium_window_id": (
                        medium_window_id
                    ),
                    "medium_window_token_pos": (
                        medium_window_token_pos
                    ),
                    "broad_window_id": (
                        broad_window_id
                    ),
                    "broad_window_token_pos": (
                        broad_window_token_pos
                    ),
                    "score": item["score"],
                    "score_local": (
                        item["score_local"]
                    ),
                    "score_medium": (
                        item["score_medium"]
                    ),
                    "score_broad": (
                        item["score_broad"]
                    ),
                    "depth": 1,
                    "via_event_id": None,
                }
            )

        (
            local_window_id,
            local_window_token_pos,
        ) = _window_metadata(
            seed_metadata,
            "local",
        )

        (
            medium_window_id,
            medium_window_token_pos,
        ) = _window_metadata(
            seed_metadata,
            "medium",
        )

        (
            broad_window_id,
            broad_window_token_pos,
        ) = _window_metadata(
            seed_metadata,
            "broad",
        )

        output.append(
            {
                "event_id": seed_event_id,
                "token": str(
                    seed_metadata["token"]
                ),
                "doc_id": str(
                    seed_metadata["doc_id"]
                ),
                "pub_year": int(
                    seed_metadata["pub_year"]
                ),
                "token_idx": int(
                    seed_metadata["token_idx"]
                ),
                "local_window_id": (
                    local_window_id
                ),
                "local_window_token_pos": (
                    local_window_token_pos
                ),
                "medium_window_id": (
                    medium_window_id
                ),
                "medium_window_token_pos": (
                    medium_window_token_pos
                ),
                "broad_window_id": (
                    broad_window_id
                ),
                "broad_window_token_pos": (
                    broad_window_token_pos
                ),
                "neighbours": neighbours_out,
            }
        )

    return output


def iter_concept_batches(
    *,
    lookup,
    indexes_by_year,
    seed_event_ids,
    scales,
    top_n,
    rrf_k,
    oversample,
    false_positives,
    batch_size,
):
    """
    Yield bounded Tier 2 batches.

    Each seed is searched only against observations from the seed's
    publication year. Multiscale retrieval and RRF therefore operate
    within that temporal population.

    Failure mode:
        A seed year without a corresponding index cannot produce neighbours.
        This should only occur if the caller constructs indexes inconsistently
        with the resolved candidate-year set.
    """
    if not seed_event_ids:
        return

    false_positives = {
        str(value).lower()
        for value in (false_positives or [])
    }

    embeddings_by_scale = {
        scale: lookup.get_scale_embeddings(
            seed_event_ids,
            scale,
        )
        for scale in scales
    }

    seed_years = []

    for event_id in seed_event_ids:
        metadata = lookup.get_event_metadata( int(event_id) )

        seed_years.append( int(metadata["pub_year"]) )

    for start in range(
        0,
        len(seed_event_ids),
        batch_size,
    ):
        seed_batch = seed_event_ids[ start:start + batch_size ]
        batch_years = seed_years[ start:start + len(seed_batch) ]
        queries_by_scale = {
            scale: embeddings_by_scale[scale][ start:start + len(seed_batch) ]
            for scale in scales
        }

        batch_events = [
            None
            for _ in seed_batch
        ]

        year_groups = {}

        for local_index, year in enumerate(batch_years):
            year_groups.setdefault(
                year,
                [],
            ).append(local_index)

        for year, local_indices in year_groups.items():
            indexes = indexes_by_year.get(year)

            if indexes is None:
                raise RuntimeError(
                    f"No temporal indexes available for publication year "
                    f"{year}"
                )

            year_queries_by_scale = {
                scale: queries_by_scale[scale][
                    local_indices
                ]
                for scale in scales
            }

            neighbours = multiscale_search(
                indexes=indexes,
                queries_by_scale=year_queries_by_scale,
                scales=scales,
                top_n=top_n,
                rrf_k=rrf_k,
                oversample=oversample,
            )

            year_seed_ids = [
                seed_batch[index]
                for index in local_indices
            ]

            events = _build_batch_events(
                seed_event_ids=year_seed_ids,
                neighbours=neighbours,
                lookup=lookup,
                false_positives=false_positives,
            )

            for local_index, event in zip(
                local_indices,
                events,
            ):
                batch_events[local_index] = event

        yield {
            "type": "batch",
            "events": batch_events,
        }

