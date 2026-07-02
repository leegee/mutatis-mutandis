# lib/segment_boundary.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol


@dataclass(frozen=True)
class Token:
    doc_id: str
    token_idx: int
    token: str


@dataclass
class BoundarySignal:
    name: str
    score: float
    weight: float
    contribution: float
    explanation: str


@dataclass
class BoundaryDecision:
    token_idx: int
    total_score: float
    is_boundary: bool
    signals: List[BoundarySignal]


@dataclass
class SegmentComplex:
    doc_id: str
    start_token_idx: int
    end_token_idx: int
    decisions: List[BoundaryDecision]


class Heuristic(Protocol):
    name: str
    weight: float

    def score(self, tokens: List[Token], i: int) -> BoundarySignal:
        ...


class PeriodHeuristic:
    name = "period"
    weight = 0.7

    def score(self, tokens: List[Token], i: int) -> BoundarySignal:
        tok = tokens[i].token
        s = 1.0 if tok == "." else 0.0

        return BoundarySignal(
            name=self.name,
            score=s,
            weight=self.weight,
            contribution=s * self.weight,
            explanation="sentence boundary marker",
        )


class SemicolonHeuristic:
    name = "semicolon"
    weight = 1.6

    def score(self, tokens: List[Token], i: int) -> BoundarySignal:
        tok = tokens[i].token

        if tok != ";":
            return BoundarySignal(
                name=self.name,
                score=0.0,
                weight=self.weight,
                contribution=0.0,
                explanation="no semicolon",
            )

        return BoundarySignal(
            name=self.name,
            score=1.0,
            weight=self.weight,
            contribution=self.weight,
            explanation="strong segment boundary",
        )


class CommaHeuristic:
    name = "comma"
    weight = 0.25

    def score(self, tokens: List[Token], i: int) -> BoundarySignal:
        tok = tokens[i].token
        s = 1.0 if tok == "," else 0.0

        return BoundarySignal(
            name=self.name,
            score=s,
            weight=self.weight,
            contribution=s * self.weight,
            explanation="weak boundary signal",
        )


class SegmentBoundaryExtractor:
    def __init__(
        self,
        heuristics: List[Heuristic],
        threshold: float = 1.0,
        min_segment_len: int = 5,
        max_segment_len: int = 60,
    ):
        self.heuristics = heuristics
        self.threshold = threshold
        self.min_segment_len = min_segment_len
        self.max_segment_len = max_segment_len

    def extract(self, tokens: List[Token]) -> List[SegmentComplex]:
        if not tokens:
            return []

        segments: List[SegmentComplex] = []
        decisions: List[BoundaryDecision] = []
        start = 0

        for i in range(len(tokens)):
            signals = [h.score(tokens, i) for h in self.heuristics]
            total = sum(s.contribution for s in signals)

            span_len = i - start

            is_boundary = (
                total >= self.threshold
                or span_len >= self.max_segment_len
            )

            decisions.append(
                BoundaryDecision(
                    token_idx=i,
                    total_score=total,
                    is_boundary=is_boundary,
                    signals=signals,
                )
            )

            if is_boundary and span_len >= self.min_segment_len:
                segments.append(
                    SegmentComplex(
                        doc_id=tokens[start].doc_id,
                        start_token_idx=tokens[start].token_idx,
                        end_token_idx=tokens[i].token_idx,
                        decisions=decisions[start : i + 1],
                    )
                )
                start = i + 1

        if start < len(tokens):
            segments.append(
                SegmentComplex(
                    doc_id=tokens[start].doc_id,
                    start_token_idx=tokens[start].token_idx,
                    end_token_idx=tokens[-1].token_idx,
                    decisions=decisions[start:],
                )
            )

        return segments
