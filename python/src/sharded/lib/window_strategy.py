"""
lib/window_strategy.py

Defines all embedding window strategies used across the pipeline.
A WindowStrategy is immutable and hashable — safe to use as a dict key or
set member.  WINDOW_STRATEGIES is the single source of truth; add new
strategies here and the rest of the pipeline picks them up automatically.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class WindowStrategy:
    """
    Describes how a document is segmented before embedding.

    name:   "sliding" | "doc" | "paragraph" | "sentence"
    size:   token window size  (sliding only; None otherwise)
    stride: token stride       (sliding only; None otherwise)
    """
    name:   str
    size:   int | None = None
    stride: int | None = None

    def __post_init__(self):
        if self.name == "sliding":
            if self.size is None or self.stride is None:
                raise ValueError("Sliding strategy requires size and stride")
            if self.stride >= self.size:
                raise ValueError("stride must be less than size")
        else:
            if self.name not in {"doc", "paragraph", "sentence"}:
                raise ValueError(f"Unknown strategy name: {self.name!r}")
            if self.size is not None or self.stride is not None:
                raise ValueError(f"Non-sliding strategy '{self.name}' must have size=None, stride=None")

    @property
    def tag(self) -> str:
        """
        Canonical filesystem-safe identifier for this strategy.
        Used as the leaf directory name in the Zarr hierarchy.

            sliding_512_256
            sliding_256_128
            doc
            paragraph
            sentence
        """
        if self.name == "sliding":
            return f"sliding_{self.size}_{self.stride}"
        return self.name

    def __str__(self) -> str:
        return self.tag


# ---------------------------------------------------------------------------
# Registry — the single source of truth for all active strategies.
# Comment out a strategy to disable it pipeline-wide.
# ---------------------------------------------------------------------------

WINDOW_STRATEGIES: list[WindowStrategy] = [
    WindowStrategy("doc"),
    WindowStrategy("paragraph"),
    WindowStrategy("sentence"),
    WindowStrategy("sliding", size=512, stride=256),
    # WindowStrategy("sliding", size=256, stride=128),  # add when ready
]
