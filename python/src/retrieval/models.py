from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Float32Array = NDArray[np.float32]
Int64Array = NDArray[np.int64]
UInt64Array = NDArray[np.uint64]


@dataclass(frozen=True, slots=True)
class SearchSpace:
    """
    Logical constraints on an observation search.

    years:
        None for all years, or an inclusive ``(start, end)`` range.

    scale:
        None for all scales, or a tuple containing one or more scales.
    """

    years: tuple[int, int] | None
    scale: tuple[str, ...] | None

    _VALID_SCALES = frozenset({
        "local",
        "medium",
        "broad",
    })

    def __post_init__(self) -> None:
        if self.years is not None:
            if isinstance(self.years, int):
                object.__setattr__(
                    self,
                    "years",
                    (self.years, self.years),
                )
            elif isinstance(self.years, tuple):
                if len(self.years) != 2:
                    raise ValueError(
                        "year range must contain exactly two years"
                    )

                if not all(
                    isinstance(year, int)
                    for year in self.years
                ):
                    raise TypeError(
                        "year range must contain integers"
                    )

                start, end = self.years

                if start > end:
                    raise ValueError(
                        "year range must be in ascending order"
                    )
            else:
                raise TypeError(
                    "years must be an int or a two-year tuple"
                )

        if self.scale is not None:
            if isinstance(self.scale, str):
                if self.scale not in self._VALID_SCALES:
                    raise ValueError(
                        f"invalid scales: {[self.scale]}"
                    )

                object.__setattr__(
                    self,
                    "scale",
                    (self.scale,),
                )
            elif isinstance(self.scale, tuple):
                if not self.scale:
                    raise ValueError(
                        "scale selection must contain at least one scale"
                    )

                if not all(
                    isinstance(scale, str)
                    for scale in self.scale
                ):
                    raise TypeError(
                        "scale selection must contain strings"
                    )

                invalid = set(self.scale) - self._VALID_SCALES

                if invalid:
                    raise ValueError(
                        f"invalid scales: {sorted(invalid)}"
                    )
            else:
                raise TypeError(
                    "scale must be a string or tuple of strings"
                )

@dataclass(slots=True)
class SearchResult:
    """ANN results expressed in stable observation IDs."""

    event_ids: UInt64Array
    distances: Float32Array

    def __post_init__(self) -> None:
        if self.event_ids.shape != self.distances.shape:
            raise ValueError(
                "event_ids and distances must have identical shapes"
            )


@dataclass(slots=True)
class BatchSearchResult:
    """ANN results for multiple queries."""

    event_ids: UInt64Array
    distances: Float32Array

    def __post_init__(self) -> None:
        if self.event_ids.ndim != 2:
            raise ValueError(
                "batch event_ids must be two-dimensional"
            )

        if self.distances.ndim != 2:
            raise ValueError(
                "batch distances must be two-dimensional"
            )

        if self.event_ids.shape != self.distances.shape:
            raise ValueError(
                "batch event_ids and distances must have identical shapes"
            )

    def row(self, index: int) -> SearchResult:
        return SearchResult(
            event_ids=self.event_ids[index],
            distances=self.distances[index],
        )
