# diskann_observation_index_store.py

from __future__ import annotations

from pathlib import Path

from lib.corpus_config import DISKANN_INDEXES_DIR

from .diskann_observation_index import DiskANNObservationIndex
from .observation_index import ObservationIndex
from .observation_index_store import ObservationIndexStore
from .models import SearchSpace


class DiskANNObservationIndexStore(ObservationIndexStore):
    """Resolve a single-year, single-scale SearchSpace to a DiskANN index."""

    def __init__(
        self,
        indexes_root: str | Path = DISKANN_INDEXES_DIR,
    ) -> None:
        self._indexes_root = Path(indexes_root)

    def get(
        self,
        space: SearchSpace,
    ) -> list[ObservationIndex]:
        index_directory = (
            self._indexes_root
            / f"year={space.years[0]}"
            / space.scale[0]
        )

        scale = space.scale[0]

        return [
            DiskANNObservationIndex(
                index_directory=index_directory,
                event_ids_path=(
                    index_directory
                    / f"{scale}_event_ids.npy"
                ),
                # TODO Expose
                dimensions=768,
                num_threads=0,
                search_complexity=100,
                beam_width=2,
                num_nodes_to_cache=0,
                index_prefix=scale,
            )
        ]
