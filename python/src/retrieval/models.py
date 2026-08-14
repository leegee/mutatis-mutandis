from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Float32Array = NDArray[np.float32]
Int64Array = NDArray[np.int64]
UInt64Array = NDArray[np.uint64]


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
