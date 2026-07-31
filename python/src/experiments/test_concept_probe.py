#!/usr/bin/env python

"""
test_concept_probe.py

Probe Tier 1 contextual observations with a natural-language concept.

Retrieval happens over contextual events. Results are aggregated to
transformer windows before display. Individual token neighbours are retained
only as evidence for why a context matched.

The search space defaults to the whole corpus. Publication year filtering
is available for diachronic experiments.
"""

from collections import defaultdict
import argparse
import numpy as np

from lib.eebo_logging import logger
from lib.zarr_event_lookup import ZarrEventLookup
from lib.eebo_faiss import EeboFaissIndex
from lib.eebo_config import ZARR_PATH, faiss_index_paths


TOP_EVENTS = 500
TOP_WINDOWS = 20
TOP_ANCHORS = 10


PROBE = [
    "the opinion of the people concerning this matter",
    "the general opinion and judgement of the nation",
    "the voice and consent of the people",
    "what the common people do believe and think",
]


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--from-year", type=int)
    parser.add_argument("--to-year", type=int)

    parser.add_argument(
        "--top",
        type=int,
        default=TOP_WINDOWS,
    )

    return parser.parse_args()



def load_indices(years, masked=False):

    index_paths = {
        year: faiss_index_paths(
            masked=masked,
            year=year,
        )
        for year in years
    }

    index = {}

    for year, paths in index_paths.items():

        index[year] = {}

        for scale, path in paths.items():

            index[year][scale] = EeboFaissIndex.load(path)

    return index



def load_lookup():

    lookup = ZarrEventLookup(
        ZARR_PATH
    )

    years = sorted(
        set(
            int(y)
            for y in lookup.pub_year
        )
    )

    index = load_indices(years)

    lookup.attach_index(index)

    return lookup



def build_query_embedding():

    from lib.macberth import embed_text

    text = " ".join(PROBE)

    return embed_text(text)



def search_all_years(
    lookup,
    query,
    from_year=None,
    to_year=None,
):

    results = []

    for year, scales in lookup._index.items():

        if from_year is not None and year < from_year:
            continue

        if to_year is not None and year > to_year:
            continue


        # Use local FAISS for retrieval.
        # Aggregation later happens over contextual windows.
        index = scales["local"]

        distances, event_ids = index.search(
            query,
            TOP_EVENTS,
        )


        for distance, event_id in zip(
            distances[0],
            event_ids[0],
        ):

            if event_id < 0:
                continue

            results.append(
                {
                    "event_id": int(event_id),
                    "score": float(distance),
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
        +
        0.5 * np.mean(
            values[: min(5, len(values))]
        )
    )



def aggregate_windows(
    results,
    lookup,
):

    windows = defaultdict(list)


    for item in results:

        pos = lookup.get_pos(
            item["event_id"]
        )

        key = (
            str(lookup.doc_id[pos]),
            int(lookup.window_id[pos]),
        )

        windows[key].append(
            {
                **item,
                "pos": pos,
            }
        )


    ranked = []

    for (doc, window), anchors in windows.items():

        positions = [
            a["pos"]
            for a in anchors
        ]

        ranked.append(
            {
                "doc": doc,
                "year": int(
                    lookup.pub_year[
                        positions[0]
                    ]
                ),
                "window": window,
                "score": score_window(
                    [
                        a["score"]
                        for a in anchors
                    ]
                ),
                "anchors": sorted(
                    anchors,
                    key=lambda x: x["score"],
                    reverse=True,
                ),
            }
        )


    return sorted(
        ranked,
        key=lambda x: x["score"],
        reverse=True,
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

    print(
        result["year"],
        result["doc"],
        "window=",
        result["window"],
    )


    print()
    print("ANCHORS:")


    for anchor in result["anchors"][:TOP_ANCHORS]:

        pos = anchor["pos"]

        print(
            f"  {anchor['score']:.4f}",
            str(lookup.token[pos]),
            f"(position={lookup.token_idx[pos]})"
        )



def print_summary(results, lookup):

    docs = defaultdict(int)
    years = defaultdict(int)

    for result in results:

        pos = lookup.get_pos(
            result["event_id"]
        )

        docs[str(lookup.doc_id[pos])] += 1
        years[int(lookup.pub_year[pos])] += 1


    print()
    print("DOCUMENT DISTRIBUTION")

    for doc, count in sorted(
        docs.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:10]:

        print(
            count,
            doc,
        )


    print()
    print("YEAR DISTRIBUTION")

    for year, count in sorted(
        years.items()
    ):

        print(
            year,
            count,
        )



def main():

    args = parse_args()

    lookup = load_lookup()

    print()
    print("Concept probe:")

    for line in PROBE:
        print(
            " -",
            line,
        )


    query = build_query_embedding()


    results = search_all_years(
        lookup,
        query,
        args.from_year,
        args.to_year,
    )


    windows = aggregate_windows(
        results,
        lookup,
    )


    print()
    print(
        f"retrieved events: {len(results)}"
    )

    print(
        f"ranked windows: {len(windows)}"
    )


    print_summary(
        results,
        lookup,
    )


    for result in windows[:args.top]:

        print_result(
            result,
            lookup,
        )



if __name__ == "__main__":
    main()
