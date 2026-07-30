from __future__ import annotations

import argparse
import statistics

from lib.corpus_db import get_connection
from lib.macberth import get_macberth_embedder
from lib.segment_boundary import (
    Token,
    SegmentBoundaryExtractor,
    PeriodHeuristic,
    SemicolonHeuristic,
    CommaHeuristic,
)


def load_document(conn, doc_id: str):
    cur = conn.cursor()

    cur.execute(
        """
        SELECT doc_id, token_idx, token
        FROM tokens
        WHERE doc_id = %s
        ORDER BY token_idx
        """,
        (doc_id,),
    )

    return [
        Token(doc_id=r[0], token_idx=r[1], token=r[2])
        for r in cur.fetchall()
    ]


def segment_text(segment, tokens):
    token_map = {t.token_idx: t.token for t in tokens}

    return " ".join(
        token_map[i]
        for i in range(segment.start_token_idx, segment.end_token_idx + 1)
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--doc-id", required=True)
    p.add_argument("--threshold", type=float, default=1.0)
    args = p.parse_args()

    conn = get_connection()
    tokens = load_document(conn, args.doc_id)

    print(f"Loaded tokens: {len(tokens)}")

    extractor = SegmentBoundaryExtractor(
        heuristics=[
            PeriodHeuristic(),
            SemicolonHeuristic(),
            CommaHeuristic(),
        ],
        threshold=args.threshold,
    )

    segments = extractor.extract(tokens)

    print(f"Segments produced: {len(segments)}")

    embedder = get_macberth_embedder()

    texts = [segment_text(s, tokens) for s in segments]
    embeddings = embedder.encode_normalized(texts)

    cohesion = [
        float(embeddings[i] @ embeddings[i + 1])
        for i in range(len(segments) - 1)
    ] + [None]

    valid = [c for c in cohesion if c is not None]

    print("\nCohesion stats")
    print(f"mean: {statistics.mean(valid):.3f}")
    print(f"median: {statistics.median(valid):.3f}")
    print(f"min: {min(valid):.3f}")
    print(f"max: {max(valid):.3f}")

    for i, c in enumerate(cohesion[:-1]):
        if c < 0.60:
            print(f"{i:4d} cohesion={c:.3f}")


if __name__ == "__main__":
    main()
