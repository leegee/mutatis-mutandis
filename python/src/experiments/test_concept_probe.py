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

The first experiment failed to demonstrate semantic retrieval because the
queries were too abstract and the resulting top windows were uninterpretable.

The concept-probe experiment now provides positive evidence that contextual
embeddings can retrieve historically related rhetorical contexts, including
the known target document.

The current probe uses four paraphrased descriptions of a known political
argument:

    - the people are encouraged to take up arms
    - by this opinion of cardinal bellarmine
    - the power of the king comes from the people
    - the people make the king

Positive evidence is strongest where independently worded queries converge
on the same transformer window and where the retrieved text expresses the
underlying proposition rather than merely sharing vocabulary.

This is evidence for the retrieval hypothesis, not yet a general validation
of it. A larger evaluation set is required to establish whether the effect
generalises beyond this hand-selected probe.

The experiment therefore reports:

    - rank of known relevant documents/windows
    - lexical overlap with the query
    - scale agreement across local/medium/broad retrieval
    - cross-query convergence
    - document-level convergence

The distinction between lexical overlap and semantic relevance is important:
near-paraphrases provide useful validation, but low-overlap convergent
retrieval is stronger evidence for contextual semantic retrieval.
"""

from collections import Counter, defaultdict
import argparse

import numpy as np

from lib.corpus_config import ZARR_PATH
from lib.corpus_faiss import CorpusFaissIndex
from lib.corpus_logging import logger
from lib.macberth import get_macberth_embedder
from lib.zarr_event_lookup import ZarrEventLookup


TOP_WINDOWS = 3
TOP_EVENTS = 100
TOP_CONVERGENCE = 7
MAX_WINDOWS_PER_DOC = 1

# Known relevant document used to evaluate whether the probe can recover
# the intended conceptual context. This is evaluation metadata, not part
# of retrieval.
TARGET_DOCS = {
    "A49152",
}

PROBE = [
    "the people are encouraged to take up arms",
    "by this opinion of cardinal bellarmine",
    "the power of the king comes from the people",
    "the people make the king",
]


def load_lookup():
    lookup = ZarrEventLookup(ZARR_PATH)
    index = CorpusFaissIndex.load_all()
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


def scale_agreement(anchors):
    """
    Measure how consistently a window is retrieved across contextual scales.

    Agreement is calculated over distinct event IDs so repeated observations
    of the same event do not artificially inflate the result.
    """
    groups = defaultdict(set)

    for anchor in anchors:
        groups[anchor["event_id"]].add(
            anchor["scale"]
        )

    if not groups:
        return 0.0

    total = 0.0

    for scales in groups.values():
        total += len(scales) / 3.0

    return total / len(groups)


def extract_window_text(
    lookup,
    result,
):
    """
    Extract a compact lexical representation around the strongest anchor.

    The output is deliberately short because this script is intended for
    diagnostic evaluation rather than corpus browsing.
    """
    anchor = max(
        result["anchors"],
        key=lambda x: x["score"],
    )

    pos = anchor["pos"]
    doc = lookup.doc_id[pos]
    token_idx = int(lookup.token_idx[pos])

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
            before[-20:],
            after[:20],
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


def lexical_overlap(
    query,
    result,
    lookup,
):
    """
    Report simple token overlap separately from embedding similarity.

    This is deliberately only a diagnostic measure: token identity is not
    treated as evidence of semantic equivalence.
    """
    query_tokens = set(
        query.lower().split()
    )

    text = extract_window_text(
        lookup,
        result,
    )

    window_tokens = set(
        text.lower().split()
    )

    shared = sorted(
        query_tokens & window_tokens
    )

    return {
        "count": len(shared),
        "total": len(query_tokens),
        "ratio": (
            len(shared) / len(query_tokens)
            if query_tokens
            else 0.0
        ),
        "shared": shared,
    }


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

    for query_idx, results in enumerate(query_results):
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

            percentile = rank / len(ordered)

            key = (
                str(result["doc"]),
                int(result["window"]),
            )

            candidate = {
                "score": score,
                "percentile": float(percentile),
                "result": result,
            }

            current = by_window[key].get(query_idx)

            if (
                current is None
                or candidate["percentile"]
                > current["percentile"]
            ):
                by_window[key][query_idx] = candidate

    convergence = []
    query_count = len(query_results)

    for (doc, window), queries in by_window.items():
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
                "convergence": min(percentiles),
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


def log_result(
    query,
    result,
    lookup,
):
    anchor = max(
        result["anchors"],
        key=lambda x: x["score"],
    )

    overlap = lexical_overlap(
        query,
        result,
        lookup,
    )

    agreement = scale_agreement(
        result["anchors"]
    )

    target = (
        " TARGET"
        if result["doc"] in TARGET_DOCS
        else ""
    )

    logger.info(
        f"{result['score']:.4f} "
        f"agreement={agreement:.2f} "
        f"overlap={overlap['count']}/{overlap['total']} "
        f"doc={result['doc']} "
        f"year={lookup.pub_year[anchor['pos']]} "
        f"window={result['window']}"
        f"{target}"
    )

    logger.info(
        "  shared: %s",
        ", ".join(overlap["shared"])
        if overlap["shared"]
        else "-",
    )

    logger.info(
        "  text: %s",
        extract_window_text(
            lookup,
            result,
        ),
    )


def log_convergence(
    convergence,
    lookup,
    limit=TOP_CONVERGENCE,
):
    """
    Log only the strongest cross-query convergent windows.

    The report is intentionally bounded so terminal output remains useful
    for inspection and comparison between probe runs.
    """
    logger.info("")
    logger.info("CROSS-QUERY CONVERGENCE")

    for result in convergence[:limit]:
        target = (
            " TARGET"
            if result["doc"] in TARGET_DOCS
            else ""
        )

        logger.info(
            f"{result['convergence']:.4f} "
            f"mean={result['mean_percentile']:.4f} "
            f"spread={result['spread']:.4f} "
            f"doc={result['doc']} "
            f"window={result['window']}"
            f"{target}"
        )

        parts = []

        for query_idx in range(len(PROBE)):
            query = result["queries"][query_idx]
            window = query["result"]
            anchor = max(
                window["anchors"],
                key=lambda x: x["score"],
            )

            parts.append(
                f"Q{query_idx + 1}="
                f"{query['percentile']:.3f}:"
                f"{anchor['token']}"
            )

        logger.info(
            "  queries: %s",
            " | ".join(parts),
        )

        first_window = result["queries"][0]["result"]

        logger.info(
            "  text: %s",
            extract_window_text(
                lookup,
                first_window,
            ),
        )


def build_document_convergence(
    query_results,
    top_n,
):
    """
    Identify documents appearing in the strongest top-N windows for each
    probe query.

    Document convergence is intentionally discrete. It answers:
    "Which documents recur across the strongest retrieval results for
    multiple formulations of the same concept?"

    The first rank at which a document appears for each query is retained
    so that convergence can be reported without introducing a document-level
    geometric score.
    """
    by_doc = defaultdict(dict)

    for query_idx, windows in enumerate(query_results):
        for rank, window in enumerate(
            windows[:top_n],
            start=1,
        ):
            doc = window["doc"]

            current = by_doc[doc].get(query_idx)

            if current is None or rank < current:
                by_doc[doc][query_idx] = rank

    return sorted(
        by_doc.items(),
        key=lambda item: (
            -len(item[1]),
            min(item[1].values()),
            item[0],
        ),
    )


def log_document_convergence(
    query_results,
    top_n,
):
    """
    Log document-level convergence independently of window-level
    convergence.

    Only documents appearing in the strongest top-N results for at least
    two queries are reported.

    Query ranks are shown because document convergence does not have a
    meaningful geometric score of its own.
    """
    convergence = build_document_convergence(
        query_results,
        top_n,
    )

    logger.info("")
    logger.info("=" * 100)
    logger.info("DOCUMENT CONVERGENCE")

    for doc, query_ranks in convergence:
        if len(query_ranks) < 2:
            continue

        parts = []

        for query_idx in sorted(query_ranks):
            parts.append(
                f"Q{query_idx + 1}@"
                f"{query_ranks[query_idx]}"
            )

        target = (
            " TARGET"
            if doc in TARGET_DOCS
            else ""
        )

        logger.info(
            f"{doc:<12} "
            f"{' '.join(parts)}"
            f"{target}"
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

    logger.info( "Loading Tier 1 event lookup" )

    lookup = load_lookup()

    logger.info("")
    logger.info("Concept probe:")

    for line in PROBE:
        logger.info(
            " - %s",
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
        logger.info("")

        logger.info( "QUERY %d: %s", query_idx, query_text, )

        results = search_all_scales(
            lookup,
            query[None, :],
            args.from_year,
            args.to_year,
        )

        windows = aggregate_windows( results, lookup, )

        if args.diverse:
            windows = diversify_windows( windows )

        logger.info( "events=%d windows=%d", len(results), len(windows), )

        for result in windows[:args.top]:
            log_result(
                query_text,
                result,
                lookup,
            )

        query_results.append(windows)

    convergence = build_convergence( query_results )

    log_convergence( convergence, lookup, limit=args.convergence_top, )

    log_document_convergence( query_results, args.convergence_top, )


if __name__ == "__main__":
    main()
