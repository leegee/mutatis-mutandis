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

from collections import defaultdict, Counter
import argparse

import numpy as np

from lib.macberth import get_macberth_embedder
from lib.zarr_event_lookup import ZarrEventLookup
from lib.eebo_faiss import EeboFaissIndex
from lib.eebo_logging import logger
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
    parser.add_argument( "--from-year", type=int, default=None, )
    parser.add_argument( "--to-year", type=int, default=None, )
    parser.add_argument( "--top", type=int, default=TOP_WINDOWS, )
    return parser.parse_args()


def load_lookup():
    lookup = ZarrEventLookup( ZARR_PATH )
    index = EeboFaissIndex.load_all()
    lookup.attach_index( index )
    return lookup


def build_query_embedding():
    embedder = get_macberth_embedder()
    text = " ".join(PROBE)
    return embedder.encode_normalized( text )


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
            scores, ids = index.search( query, TOP_EVENTS, )
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

    values = sorted( scores, reverse=True, )

    return (
        0.5 * values[0]
        + 0.5 * np.mean(
            values[: min(5, len(values))]
        )
    )


def aggregate_windows( results, lookup, ):
    windows = defaultdict(list)

    for result in results:
        pos = lookup.get_pos( result["event_id"] )
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
                "anchors": anchors,
            }
        )

    return sorted(
        ranked,
        key=lambda x: x["score"],
        reverse=True,
    )



def group_anchors( anchors ):
    """
    Collapse duplicate event hits caused by multi-scale retrieval.

    A single event may be returned by local, medium and broad indices.
    Displaying one line avoids visually inflating agreement.
    """
    grouped = {}

    for anchor in anchors:
        key = anchor["event_id"]
        if key not in grouped:
            grouped[key] = {
                **anchor,
                "scales": {},
            }
        grouped[key]["scales"][
            anchor["scale"]
        ] = anchor["score"]

    return sorted(
        grouped.values(),
        key=lambda x: x["score"],
        reverse=True,
    )



def print_scale_agreement( anchors, ):
    groups = defaultdict(set)

    for anchor in anchors:
        groups[ anchor["event_id"] ].add( anchor["scale"] )

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
        print( f"{key}: {value}" )


def extract_window_text( lookup, result, ):
    anchor = result["anchors"][0]
    pos = anchor["pos"]
    doc = lookup.doc_id[pos]
    token_idx = int( lookup.token_idx[pos] )

    positions = np.where( lookup.doc_id[:] == doc )[0]

    before = positions[ lookup.token_idx[positions] < token_idx ]
    after = positions[ lookup.token_idx[positions] >= token_idx ]

    selected = np.concatenate(
        [
            before[-40:],
            after[:40],
        ]
    )

    selected = selected[
        np.argsort( lookup.token_idx[selected] )
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
    print( f"event_id: {anchor['event_id']}" )
    print( f"doc: {result['doc']}" )
    print( f"year: {lookup.pub_year[pos]}" )
    print( f"window: {result['window']}" )
    print()
    print("TEXT:")
    print( extract_window_text( lookup, result ) )
    print()
    print("ANCHORS:")

    grouped = group_anchors( result["anchors"] )

    for anchor in grouped[:TOP_ANCHORS]:
        pos = anchor["pos"]

        print(
            f"{anchor['score']:.4f}",
            lookup.token[pos],
            f"@{lookup.token_idx[pos]}",
        )

        print(
            "   ",
            " ".join(
                f"{scale}={score:.4f}"
                for scale, score
                in anchor["scales"].items()
            )
        )

    print_scale_agreement( result["anchors"] )



def main():
    args = parse_args()

    logger.info( "Loading Tier 1 event lookup" )
    lookup = load_lookup()

    print()
    print("Concept probe:")

    for line in PROBE:
        print( " -", line )

    query = build_query_embedding()

    results = search_all_scales(
        lookup,
        query,
        args.from_year,
        args.to_year,
    )

    windows = aggregate_windows( results, lookup )

    print()
    print( f"retrieved events: {len(results)}" )
    print( f"ranked windows: {len(windows)}" )

    for result in windows[:args.top]:
        print_result( result, lookup )

if __name__ == "__main__":
    main()
