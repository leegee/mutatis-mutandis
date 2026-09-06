# observation_index_store

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import SearchSpace
from .observation_index import ObservationIndex


class ObservationIndexStore(ABC):

    @abstractmethod
    def get(
        self,
        space: SearchSpace,
    ) -> list[ObservationIndex]:
        """Return the indexes required to search the requested space."""
        raise NotImplementedError

    @property
    def available_scales(self) -> tuple[str, ...]:
        ...
