# diskann_observation_index_store.py

from __future__ import annotations

from pathlib import Path

from lib.corpus_config import DISKANN_INDEXES_DIR

from .diskann_observation_index import DiskANNObservationIndex
from .observation_index import ObservationIndex
from .observation_index_store import ObservationIndexStore
from .models import SearchSpace


class DiskANNObservationIndexStore(ObservationIndexStore):
    """Resolve a logical SearchSpace to its physical DiskANN indexes."""

    _SCALE_ORDER = (
        "local",
        "medium",
        "broad",
    )

    def __init__(
        self,
        indexes_root: str | Path = DISKANN_INDEXES_DIR,
    ) -> None:
        self._indexes_root = Path(indexes_root)

    def get(
        self,
        space: SearchSpace,
    ) -> list[ObservationIndex]:
        """Return all physical indexes covered by the search space."""

        years = self._available_years()

        if space.years is not None:
            start, end = space.years
            years = tuple(
                year
                for year in years
                if start <= year <= end
            )

        if space.scale is None:
            scales = self._SCALE_ORDER
        else:
            scales = tuple(
                scale
                for scale in self._SCALE_ORDER
                if scale in space.scale
            )

        return [
            self._build_index(
                year=year,
                scale=scale,
            )
            for year in years
            for scale in scales
            if self._index_exists(
                year=year,
                scale=scale,
            )
        ]

    def _available_years(self) -> tuple[int, ...]:
        """Discover years from the physical index directory layout."""

        years: list[int] = []

        if not self._indexes_root.exists():
            return ()

        for path in self._indexes_root.glob("year=*"):
            if not path.is_dir():
                continue

            try:
                year = int(path.name.removeprefix("year="))
            except ValueError:
                continue

            years.append(year)

        return tuple(sorted(set(years)))

    def _index_exists(
        self,
        *,
        year: int,
        scale: str,
    ) -> bool:
        index_directory = (
            self._indexes_root
            / f"year={year}"
            / scale
        )

        return (
            index_directory.is_dir()
            and (
                index_directory
                / f"{scale}_event_ids.npy"
            ).is_file()
        )

    def _build_index(
        self,
        *,
        year: int,
        scale: str,
    ) -> ObservationIndex:
        index_directory = (
            self._indexes_root
            / f"year={year}"
            / scale
        )

        return DiskANNObservationIndex(
            index_directory=index_directory,
            event_ids_path=(
                index_directory
                / f"{scale}_event_ids.npy"
            ),
            # These parameters describe the current on-disk index format.
            # Expose them through configuration when index construction/loading
            # needs to vary independently of the store.
            dimensions=768,
            num_threads=0,
            search_complexity=100,
            beam_width=2,
            num_nodes_to_cache=0,
            index_prefix=scale,
        )
