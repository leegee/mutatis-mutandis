from dataclasses import dataclass
import numpy as np

@dataclass(slots=True)
class TokenRow:
    corpus: str
    doc_id: str
    token_idx: int
    token: str
    pub_year: int | None

@dataclass(slots=True)
class EmbeddedVector:
    vector: np.ndarray
    window_id: int
    window_token_pos: int


@dataclass(slots=True)
class Observation:
    event_id: int | None
    corpus: str
    doc_id: str
    token: str
    token_idx: int
    pub_year: int | None

    local_window_id: int | None
    local_window_token_pos: int | None

    medium_window_id: int | None
    medium_window_token_pos: int | None

    broad_window_id: int | None
    broad_window_token_pos: int | None


@dataclass(slots=True)
class EmbeddedObservation:
    observation: Observation
    vectors: dict[str, np.ndarray]


@dataclass(slots=True)
class SpanObservation:
    event_id: int | None
    corpus: str
    doc_id: str
    phrase: str
    token_idx: int
    span_end_idx: int
    pub_year: int | None

    local_window_id: int | None
    local_window_token_pos: int | None

    medium_window_id: int | None
    medium_window_token_pos: int | None

    broad_window_id: int | None
    broad_window_token_pos: int | None


@dataclass(slots=True)
class EmbeddedSpanObservation:
    observation: SpanObservation
    vectors: dict[str, np.ndarray]


AnyEmbedded = EmbeddedObservation | EmbeddedSpanObservation


@dataclass(slots=True)
class DocumentOutcome:
    """
    Result of processing one document.

    status:
        processed         work was done (or, in dry-run, embedded)
        already_complete  events of this kind already exist (queue only)
        no_targets        document has no seeds / phrase spans
        missing           document not found in pamphlet_tokens/corpus
        would_backfill    dry-run only: a backfill job that was not run
    """

    status: str
    targets: int = 0
    observations: int = 0
    written: int = 0


@dataclass(slots=True)
class DocBuffer:
    corpus: str
    doc_id: str
    pub_year: int | None
    rows: list[TokenRow]

    @property
    def tokens(self) -> list[str]:
        return [row.token for row in self.rows]

    def __bool__(self) -> bool:
        return bool(self.rows)

