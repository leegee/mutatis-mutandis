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
    ) -> ObservationIndex:
        """Return the index corresponding to a search space."""
        raise NotImplementedError
