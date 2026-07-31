#!/usr/bin/env python

"""
test_concept_probe.py

Probe Tier 1 contextual observations with a natural-language concept.

Retrieval happens over contextual events, but results are aggregated to
transformer windows before display.

FAISS owns geometric retrieval.
Zarr owns event identity and provenance.

The default search spans the corpus. Publication-year restriction is an
experimental filter, not part of the retrieval model.

Cross-query convergence is measured at transformer-window level. Raw cosine
scores are converted to within-query percentile ranks before comparison,
because absolute scores from independently encoded queries are not directly
comparable.


> Contextual MacBERTh embeddings can retrieve historically meaningful lexical/conceptual usages from EEBO through natural-language conceptual probes, with convergence across paraphrased probes providing evidence of semantic validity.

Not according to this experiment which surfaces at the top:

eebo=# select * from pamphlet_corpus where doc_id = 'A74859';
-[ RECORD 1 ]---+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
corpus          | eebo
doc_id          | A74859
filepath        | eebo_all/eebo_phase2/P4_XML_TCP_Ph2/A7/A7/A74859.P4.xml
title           | Paul's Church-yard. Libri theologici, politici, historici, nundinus Paulinis (una╠Ç cum templo) prostant venales. Juxta seriem alphabeti democratici. / Done into English for the Assembly of Divines.
author          | Birkenhead, John, Sir, 1616-1679.
pub_year        | 1651
publisher       | s.n.,
pub_place       | [S.l. :
source_date_raw | 1651-1652]
token_count     | 5367
lang            | eng


"""

from collections import defaultdict, Counter
import argparse

import numpy as np

from lib.macberth import get_macberth_embedder
from lib.zarr_event_lookup import ZarrEventLookup
from lib.eebo_faiss import EeboFaissIndex
from lib.corpus_logging import logger
from lib.corpus_config import ZARR_PATH


TOP_WINDOWS = 5
TOP_ANCHORS = 5
TOP_EVENTS = 100

TOP_CONVERGENCE = 50

MAX_WINDOWS_PER_DOC = 1

PROBE = [
    "the people"
    # "the opinion of the people concerning this matter",
    # "the general opinion and judgement of the nation",
    # "the voice and consent of the people",
    # "what the common people do believe and think",
]


def load_lookup():
    lookup = ZarrEventLookup(ZARR_PATH)
    index = EeboFaissIndex.load_all()
    lookup.attach_index(index)
    return lookup


def build_query_embeddings():
    embedder = get_macberth_embedder()
    return embedder.encode_normalized(PROBE)


def search_all_scales(
    lookup,
    query,
    from_year=None,
    to_year=None,
):
    """
    Retrieve candidates across local/medium/broad spaces.

    Year filtering happens here because publication year is metadata.
    FAISS only knows event geometry.
    """
    results = []

    for year, scales in lookup._index.items():
        if from_year is not None and year < from_year:
            continue

        if to_year is not None and year > to_year:
            continue

        for scale, index in scales.items():
            scores, ids = index.search(
                query,
                TOP_EVENTS,
            )

            for score, event_id in zip(
                scores[0],
                ids[0],
            ):
                if event_id < 0:
                    continue

                results.append(
                    {
                        "event_id": int(event_id),
                        "score": float(score),
                        "scale": scale,
                        "year": year,
                    }
                )

    return sorted(
        results,
        key=lambda x: x["score"],
        reverse=True,
    )


def score_window(scores):
    if not scores:
        return 0.0

    values = sorted(
        scores,
        reverse=True,
    )

    return (
        0.5 * values[0]
        + 0.5 * np.mean(
            values[: min(5, len(values))]
        )
    )


def aggregate_windows(
    results,
    lookup,
):
    """
    Collapse event-level retrieval into transformer windows.

    A window can contain multiple retrieved contextual observations. The
    strongest observations dominate the window score while additional
    observations provide supporting evidence.
    """
    windows = defaultdict(list)

    for result in results:
        pos = lookup.get_pos(
            result["event_id"]
        )

        key = (
            str(lookup.doc_id[pos]),
            int(lookup.window_id[pos]),
        )

        windows[key].append(
            {
                **result,
                "pos": pos,
                "token": lookup.token[pos],
            }
        )

    ranked = []

    for (doc, window), anchors in windows.items():
        ranked.append(
            {
                "doc": doc,
                "window": window,
                "score": score_window(
                    [
                        x["score"]
                        for x in anchors
                    ]
                ),
                "anchors": anchors,
            }
        )

    return sorted(
        ranked,
        key=lambda x: x["score"],
        reverse=True,
    )


def group_anchors(anchors):
    """
    Collapse repeated lexical anchors while retaining scale evidence.

    Multiple scale hits for the same event are combined into one anchor.
    Multiple occurrences of the same token are retained as a hit count.
    """
    grouped = defaultdict(list)

    for anchor in anchors:
        grouped[anchor["token"]].append(anchor)

    output = []

    for token, values in grouped.items():
        values = sorted(
            values,
            key=lambda x: x["score"],
            reverse=True,
        )

        best = values[0]
        scales = {}

        for value in values:
            scale = value["scale"]
            score = value["score"]

            if (
                scale not in scales
                or score > scales[scale]
            ):
                scales[scale] = score

        output.append(
            {
                **best,
                "count": len(values),
                "scales": scales,
            }
        )

    return sorted(
        output,
        key=lambda x: x["score"],
        reverse=True,
    )


def print_scale_agreement(anchors):
    groups = defaultdict(set)

    for anchor in anchors:
        groups[
            anchor["event_id"]
        ].add(anchor["scale"])

    counts = Counter()

    for scales in groups.values():
        if len(scales) == 3:
            counts["local+medium+broad"] += 1
        elif len(scales) == 2:
            counts["two scales"] += 1
        else:
            counts["single scale"] += 1

    print()
    print("SCALE AGREEMENT")

    for key, value in counts.items():
        print(
            f"{key}: {value}"
        )


def extract_window_text(
    lookup,
    result,
):
    anchor = result["anchors"][0]

    pos = anchor["pos"]
    doc = lookup.doc_id[pos]
    token_idx = int(
        lookup.token_idx[pos]
    )

    positions = np.where(
        lookup.doc_id[:] == doc
    )[0]

    before = positions[
        lookup.token_idx[positions] < token_idx
    ]

    after = positions[
        lookup.token_idx[positions] >= token_idx
    ]

    selected = np.concatenate(
        [
            before[-40:],
            after[:40],
        ]
    )

    selected = selected[
        np.argsort(
            lookup.token_idx[selected]
        )
    ]

    return " ".join(
        lookup.token[p]
        for p in selected
    )


def print_result(
    result,
    lookup,
):
    print()
    print("=" * 100)

    print(
        f"score: {result['score']:.4f}"
    )

    anchor = result["anchors"][0]
    pos = anchor["pos"]

    print(
        f"event_id: {anchor['event_id']}"
    )

    print(
        f"doc: {result['doc']}"
    )

    print(
        f"year: {lookup.pub_year[pos]}"
    )

    print(
        f"window: {result['window']}"
    )

    print()
    print("TEXT:")

    print(
        extract_window_text(
            lookup,
            result,
        )
    )

    print()
    print("ANCHORS:")

    grouped = group_anchors(
        result["anchors"]
    )

    for anchor in grouped[:TOP_ANCHORS]:
        pos = anchor["pos"]

        print(
            f"{anchor['score']:.4f}",
            anchor["token"],
            f"@{lookup.token_idx[pos]}",
            f"({anchor['count']} hits)",
        )

        print(
            "   ",
            " ".join(
                f"{scale}={score:.4f}"
                for scale, score
                in anchor["scales"].items()
            ),
        )

    print_scale_agreement(
        result["anchors"]
    )


def diversify_windows(windows):
    seen = Counter()
    output = []

    for window in windows:
        doc = window["doc"]

        if (
            MAX_WINDOWS_PER_DOC is not None
            and seen[doc] >= MAX_WINDOWS_PER_DOC
        ):
            continue

        output.append(window)
        seen[doc] += 1

    return output


def build_convergence(query_results):
    """
    Identify transformer windows supported by every probe query.

    Raw cosine scores are not directly comparable across independently
    encoded queries. Each query is therefore converted to a within-query
    percentile rank.

    A window's convergence score is the weakest query percentile. This
    prevents a window from ranking highly merely because three queries like
    it strongly while one query does not.

    Multiple retrieved events belonging to the same window are collapsed to
    the strongest result for each query.
    """
    by_window = defaultdict(dict)

    for query_idx, results in enumerate(
        query_results
    ):
        if not results:
            continue

        scores = np.asarray(
            [
                result["score"]
                for result in results
                if result["score"] is not None
            ],
            dtype=np.float32,
        )

        if len(scores) == 0:
            continue

        ordered = np.sort(scores)

        for result in results:
            score = result["score"]

            rank = np.searchsorted(
                ordered,
                score,
                side="right",
            )

            percentile = (
                rank / len(ordered)
            )

            key = (
                str(result["doc"]),
                int(result["window"]),
            )

            candidate = {
                "score": score,
                "percentile": float(
                    percentile
                ),
                "result": result,
            }

            current = by_window[key].get(
                query_idx
            )

            if (
                current is None
                or candidate["percentile"]
                > current["percentile"]
            ):
                by_window[key][query_idx] = candidate

    convergence = []

    query_count = len(
        query_results
    )

    for (
        (doc, window),
        queries,
    ) in by_window.items():

        if len(queries) != query_count:
            continue

        percentiles = [
            queries[i]["percentile"]
            for i in range(query_count)
        ]

        scores = [
            queries[i]["score"]
            for i in range(query_count)
        ]

        convergence.append(
            {
                "doc": doc,
                "window": window,
                "convergence": min(
                    percentiles
                ),
                "mean_percentile": float(
                    np.mean(percentiles)
                ),
                "spread": (
                    max(percentiles)
                    - min(percentiles)
                ),
                "scores": scores,
                "queries": queries,
            }
        )

    return sorted(
        convergence,
        key=lambda x: (
            x["convergence"],
            x["mean_percentile"],
            -x["spread"],
        ),
        reverse=True,
    )


def print_convergence(
    convergence,
    lookup,
    limit=TOP_CONVERGENCE,
):
    """
    Print only the strongest cross-query convergent windows.

    The report is intentionally bounded because the full intersection can
    contain thousands of windows and is useful as data, not as terminal
    output.
    """
    print()
    print("=" * 100)
    print("CROSS-QUERY CONVERGENCE")

    for result in convergence[:limit]:
        print()
        print(
            f"convergence: "
            f"{result['convergence']:.4f}  "
            f"mean percentile: "
            f"{result['mean_percentile']:.4f}  "
            f"spread: "
            f"{result['spread']:.4f}"
        )

        print(
            f"doc: {result['doc']}  "
            f"window: {result['window']}"
        )

        for query_idx in range(
            len(result["queries"])
        ):
            query = result["queries"][
                query_idx
            ]

            window = query["result"]
            anchors = group_anchors(
                window["anchors"]
            )

            anchor = anchors[0]

            print(
                f"  Q{query_idx + 1}: "
                f"score={query['score']:.4f}  "
                f"percentile="
                f"{query['percentile']:.4f}  "
                f"token={anchor['token']}"
            )

        first_query = result["queries"][0]
        first_window = first_query["result"]

        print(
            "  text: "
            + extract_window_text(
                lookup,
                first_window,
            )
        )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--from-year",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--to-year",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--top",
        type=int,
        default=TOP_WINDOWS,
    )

    parser.add_argument(
        "--diverse",
        action="store_true",
    )

    parser.add_argument(
        "--convergence-top",
        type=int,
        default=TOP_CONVERGENCE,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(
        "Loading Tier 1 event lookup"
    )

    lookup = load_lookup()

    print()
    print("Concept probe:")

    for line in PROBE:
        print(
            " -",
            line,
        )

    queries = build_query_embeddings()

    query_results = []

    for query_idx, (
        query_text,
        query,
    ) in enumerate(
        zip(PROBE, queries),
        start=1,
    ):
        print()
        print("=" * 100)
        print(
            f"QUERY {query_idx}:"
        )
        print(query_text)

        results = search_all_scales(
            lookup,
            query[None, :],
            args.from_year,
            args.to_year,
        )

        windows = aggregate_windows(
            results,
            lookup,
        )

        if args.diverse:
            windows = diversify_windows(
                windows
            )

        print()
        print(
            f"retrieved events: "
            f"{len(results)}"
        )

        print(
            f"ranked windows: "
            f"{len(windows)}"
        )

        for result in windows[
            :args.top
        ]:
            print_result(
                result,
                lookup,
            )

        query_results.append(
            windows
        )

    convergence = build_convergence(
        query_results
    )

    print_convergence(
        convergence,
        lookup,
        limit=args.convergence_top,
    )


if __name__ == "__main__":
    main()
