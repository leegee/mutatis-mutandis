#retrieval/mapping.py

from pathlib import Path
import numpy as np
from .models import UInt64Array


class ObservationIdMapping:
    """Maps DiskANN local integer IDs to stable observation IDs."""

    def __init__(self, path: str | Path) -> None:
        self._event_ids: UInt64Array = np.load(path, mmap_mode="r")

        if self._event_ids.ndim != 1:
            raise ValueError("event ID mapping must be one-dimensional")

        if self._event_ids.dtype != np.uint64:
            raise ValueError( f"event ID mapping must use uint64, got {self._event_ids.dtype}" )

    def __len__(self) -> int:
        return self._event_ids.shape[0]

    def event_ids(self, local_ids: np.ndarray) -> UInt64Array:
        if np.any(local_ids < 0) or np.any(local_ids >= len(self)):
            raise IndexError("DiskANN local ID is outside the mapping")

        return self._event_ids[local_ids]
