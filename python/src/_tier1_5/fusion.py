from __future__ import annotations

from typing import Protocol, Sequence

from .ann import RankedResults


class ResultFusion(Protocol):
    """
    Combines independently retrieved ranked candidate lists.

    Fusion is deliberately independent of the ANN implementation.
    """

    def fuse(
        self,
        results: Sequence[RankedResults],
        limit: int,
    ) -> RankedResults:
        ...


class ReciprocalRankFusion:
    def __init__(self, k: int = 60) -> None:
        if k <= 0:
            raise ValueError("RRF k must be positive")
        self.k = k

    def fuse(
        self,
        results: Sequence[RankedResults],
        limit: int,
    ) -> RankedResults:
        if limit <= 0:
            raise ValueError("limit must be positive")

        scores: dict[int, float] = {}

        for result in results:
            for query_ids, _query_scores in zip(
                result.event_ids,
                result.scores,
            ):
                for rank, event_id in enumerate(query_ids, start=1):
                    event_id = int(event_id)

                    # FAISS and some other ANN implementations use -1 to
                    # represent an unfilled result slot.
                    if event_id == -1:
                        continue

                    scores[event_id] = scores.get(event_id, 0.0) + (
                        1.0 / (self.k + rank)
                    )

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:limit]

        ids = np.asarray(
            [event_id for event_id, _ in ranked],
            dtype=np.int64,
        )

        fused_scores = np.asarray(
            [score for _, score in ranked],
            dtype=np.float32,
        )

        return RankedResults(
            event_ids=ids[None, :],
            scores=fused_scores[None, :],
        )
