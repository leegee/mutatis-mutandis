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
"""

from collections import defaultdict
import argparse
import numpy as np

from lib.eebo_logging import logger
from lib.zarr_event_lookup import ZarrEventLookup
from lib.eebo_faiss import EeboFaissIndex
from lib.eebo_config import ZARR_PATH


TOP_WINDOWS = 20
TOP_ANCHORS = 10
TOP_EVENTS = 500


PROBE = [
    "the opinion of the people concerning this matter",
    "the general opinion and judgement of the nation",
    "the voice and consent of the people",
    "what the common people do believe and think",
]


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

    return parser.parse_args()



def load_lookup():

    lookup = ZarrEventLookup(
        ZARR_PATH
    )

    index = EeboFaissIndex.load_all()

    lookup.attach_index(
        index
    )

    return lookup



def build_query_embedding():

    from lib.macberth import get_macberth_embedder

    embedder = get_macberth_embedder()

    text = " ".join(PROBE)

    return embedder.encode_normalized(
        text
    )



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



def extract_window_text(
    lookup,
    result,
):

    """
    Reconstruct diagnostic context from the same document.

    The Zarr token stream is corpus ordered, so do not blindly slice
    around the global event position.
    """

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


    start_positions = before[-40:]
    end_positions = after[:40]


    selected = np.concatenate(
        [
            start_positions,
            end_positions,
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

    print(
        result["doc"],
        "window=",
        result["window"],
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

    for anchor in result["anchors"][:TOP_ANCHORS]:

        pos = anchor["pos"]

        print(
            f"{anchor['score']:.4f}",
            anchor["scale"],
            lookup.token[pos],
            f"@{lookup.token_idx[pos]}",
        )



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


    query = build_query_embedding()


    results = search_all_scales(
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


    for result in windows[:args.top]:

        print_result(
            result,
            lookup,
        )



if __name__ == "__main__":
    main()