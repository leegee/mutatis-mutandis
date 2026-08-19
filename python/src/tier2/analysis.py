from __future__ import annotations

from collections import Counter

import numpy as np

from lib.corpus_logging import logger
from tier1.observation_store_api import SCALES
from retrieval.lazy_year_disk_ann import LazyYearDiskANN

K = 60
RRF_K = 60
OVERSAMPLE = 5
BATCH_SIZE = 2000

_NO_WPOS = -1


def _chunks(seq, size):
    if size <= 0:
        raise ValueError("chunk size must be positive")

    for start in range(0, len(seq), size):
        yield seq[start:start + size]


def resolve_concept_positions(
    *,
    concept_name,
    concept,
    lookup,
    false_positives=None,
):
    """
    Resolve lexical seed events and group their stable event IDs by year.

    Tier 2 uses event_id as its identity; physical vector positions are not
    part of the retrieval contract.

    Failure mode:
        Metadata lookup is intentionally performed only for seed events.
        The potentially very large observation corpus is never scanned into
        Python merely to construct the year grouping.
    """
    forms = {
        str(form).lower()
        for form in concept.get("forms", [])
    }

    false_positives = {
        str(value).lower()
        for value in (false_positives or [])
    }

    logger.info(
        "[tier2] %s forms: %s",
        concept_name,
        sorted(forms)[:50],
    )

    event_ids = lookup.find_matching_event_ids(
        forms,
        false_positives,
    )

    event_ids = [
        int(event_id)
        for event_id in event_ids
    ]

    logger.info(
        "[tier2] %s: %d seed events",
        concept_name,
        len(event_ids),
    )

    by_year: dict[int, list[int]] = {}

    for event_id in event_ids:
        metadata = lookup.get_event_metadata(event_id)
        year = int(metadata["pub_year"])
        by_year.setdefault(year, []).append(event_id)

    return {
        "forms": forms,
        "false_positives": false_positives,
        "event_ids": event_ids,
        "event_ids_set": set(event_ids),
        "by_year": by_year,
    }


def multiscale_search(
    *,
    indexes,
    queries,
    top_n=K,
    rrf_k=RRF_K,
    oversample=OVERSAMPLE,
):
    """
    Search all three scale-specific DiskANN indexes and fuse their rankings.

    `indexes` is the three-index mapping returned by YearDiskANN.get(year).

    DiskANN distances are retained under the existing Tier 2 score field
    names for downstream compatibility. RRF uses rank rather than distance.

    Failure mode:
        Each scale may return the same event. RRF therefore fuses by stable
        event_id rather than by local DiskANN position.
    """
    queries = np.asarray(
        queries,
        dtype=np.float32,
    )

    if queries.ndim != 2:
        raise ValueError("queries must be two-dimensional")

    if top_n <= 0:
        raise ValueError("top_n must be positive")

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")

    if oversample <= 0:
        raise ValueError("oversample must be positive")

    search_k = top_n * oversample
    results_by_scale = {}

    for scale in SCALES:
        index = indexes.get(scale)

        if index is None:
            raise RuntimeError(
                f"Missing DiskANN index for scale={scale}"
            )

        results_by_scale[scale] = index.batch_search(
            queries,
            k=search_k,
        )

    output = []

    for query_idx in range(len(queries)):
        fused = {}

        for scale in SCALES:
            result = results_by_scale[scale]

            event_ids = result.event_ids[query_idx]
            distances = result.distances[query_idx]

            for rank, (event_id, distance) in enumerate(
                zip(event_ids, distances),
                start=1,
            ):
                event_id = int(event_id)

                item = fused.setdefault(
                    event_id,
                    {
                        "score": 0.0,
                        "score_local": None,
                        "score_medium": None,
                        "score_broad": None,
                    },
                )

                item["score"] += 1.0 / (rrf_k + rank)
                item[f"score_{scale}"] = float(distance)

        ranked = sorted(
            fused.items(),
            key=lambda item: item[1]["score"],
            reverse=True,
        )

        output.append(
            [
                {
                    "event_id": event_id,
                    **payload,
                }
                for event_id, payload in ranked[:top_n]
            ]
        )

    return output


def _metadata_for_event(
    lookup,
    event_id,
):
    """Return public Tier 1 metadata for one stable observation ID."""
    return lookup.get_event_metadata(int(event_id))


def _window_token_pos(metadata):
    value = metadata.get(
        "window_token_pos",
        _NO_WPOS,
    )

    if value is None:
        return None

    value = int(value)

    if value == _NO_WPOS:
        return None

    return value


def _build_batch_events(
    *,
    seed_event_ids,
    neighbours,
    lookup,
    false_positives,
    token_counts,
    doc_counts,
    window_counts,
):
    """
    Materialise one bounded batch in the existing Tier 2 output schema.

    Only metadata for the seeds and their selected neighbours is requested.
    Embeddings are not reconstructed here.
    """
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
            neighbour_id = int(item["event_id"])

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

            doc_id = str(
                metadata["doc_id"]
            )

            window_id = int(
                metadata["window_id"]
            )

            window_token_pos = _window_token_pos(
                metadata,
            )

            token_counts[token] += 1
            doc_counts[doc_id] += 1
            window_counts[(doc_id, window_id)] += 1

            neighbours_out.append(
                {
                    "event_id": neighbour_id,
                    "vector_id": int(
                        metadata["vector_id"]
                    ),
                    "token": token,
                    "doc_id": doc_id,
                    "pub_year": int(
                        metadata["pub_year"]
                    ),
                    "token_idx": int(
                        metadata["token_idx"]
                    ),
                    "window_id": window_id,
                    "window_token_pos": window_token_pos,
                    "score": item["score"],
                    "score_local": item["score_local"],
                    "score_medium": item["score_medium"],
                    "score_broad": item["score_broad"],
                    "depth": 1,
                    "via_event_id": None,
                }
            )

        output.append(
            {
                "event_id": seed_event_id,
                "vector_id": int(
                    seed_metadata["vector_id"]
                ),
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
                "window_id": int(
                    seed_metadata["window_id"]
                ),
                "window_token_pos": _window_token_pos(
                    seed_metadata,
                ),
                "neighbours": neighbours_out,
            }
        )

    return output


def iter_year_concept_batches(
    *,
    concept_name,
    concept,
    lookup,
    indexes: LazyYearDiskANN,
    year,
    top_n=K,
    rrf_k=RRF_K,
    oversample=OVERSAMPLE,
    false_positives=None,
    resolved=None,
    batch_size=BATCH_SIZE,
    token_counts=None,
    doc_counts=None,
    window_counts=None,
):
    """
    Yield bounded Tier 2 event batches for one concept and one year.

    Only seed embeddings for the current batch are materialised. DiskANN
    performs corpus-scale search against the three indexes for this year.

    The year resource is cached by LazyYearDiskANN, so repeated concept
    searches do not repeatedly reopen the physical DiskANN indexes.

    Failure mode:
        A year with no seed events is a no-op and does not open its DiskANN
        indexes.
    """
    year = int(year)

    if resolved is None:
        resolved = resolve_concept_positions(
            concept_name=concept_name,
            concept=concept,
            lookup=lookup,
            false_positives=false_positives,
        )

    false_positives = resolved["false_positives"]

    seed_ids = resolved["by_year"].get(
        year,
        [],
    )

    if not seed_ids:
        return

    if token_counts is None:
        token_counts = Counter()

    if doc_counts is None:
        doc_counts = Counter()

    if window_counts is None:
        window_counts = Counter()

    year_indexes = indexes.get(year)

    for seed_batch in _chunks(
        seed_ids,
        batch_size,
    ):
        queries = lookup.get_embeddings(
            seed_batch,
            scales=SCALES,
        )

        neighbours = multiscale_search(
            indexes=year_indexes,
            queries=queries,
            top_n=top_n,
            rrf_k=rrf_k,
            oversample=oversample,
        )

        events = _build_batch_events(
            seed_event_ids=seed_batch,
            neighbours=neighbours,
            lookup=lookup,
            false_positives=false_positives,
            token_counts=token_counts,
            doc_counts=doc_counts,
            window_counts=window_counts,
        )

        yield {
            "type": "batch",
            "events": events,
            "seed_ids": set(
                resolved["event_ids_set"]
            ),
        }
