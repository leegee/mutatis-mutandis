from __future__ import annotations

from pathlib import Path

from lib.corpus_config import DISKANN_INDEXES_DIR

from .diskann_observation_index import DiskANNObservationIndex
from .observation_index import ObservationIndex
from .observation_index_store import ObservationIndexStore
from .models import SearchSpace


class DiskANNObservationIndexStore(ObservationIndexStore):
    """Resolve logical search spaces to temporal DiskANN indexes."""

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
        """Return physical indexes intersecting the requested year range."""

        buckets = self._available_buckets()

        if space.years is not None:
            start, end = space.years

            buckets = tuple(
                bucket
                for bucket in buckets
                if bucket[1] >= start and bucket[0] <= end
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
                bucket_start=bucket_start,
                bucket_end=bucket_end,
                scale=scale,
            )
            for bucket_start, bucket_end in buckets
            for scale in scales
            if self._index_exists(
                bucket_start=bucket_start,
                bucket_end=bucket_end,
                scale=scale,
            )
        ]

    def _available_buckets(
        self,
    ) -> tuple[tuple[int, int], ...]:
        """Discover temporal buckets from the physical index layout."""

        buckets: set[tuple[int, int]] = set()

        if not self._indexes_root.exists():
            return ()

        for path in self._indexes_root.glob("year=*"):
            if not path.is_dir():
                continue

            value = path.name.removeprefix("year=")

            try:
                start_text, end_text = value.split("-", 1)
                start = int(start_text)
                end = int(end_text)
            except ValueError:
                continue

            if start > end:
                continue

            buckets.add((start, end))

        return tuple(sorted(buckets))

    def _index_exists(
        self,
        *,
        bucket_start: int,
        bucket_end: int,
        scale: str,
    ) -> bool:
        index_directory = (
            self._indexes_root
            / f"year={bucket_start}-{bucket_end}"
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
        bucket_start: int,
        bucket_end: int,
        scale: str,
    ) -> ObservationIndex:
        index_directory = (
            self._indexes_root
            / f"year={bucket_start}-{bucket_end}"
            / scale
        )

        return DiskANNObservationIndex(
            index_directory=index_directory,
            event_ids_path=(
                index_directory
                / f"{scale}_event_ids.npy"
            ),
            dimensions=768,
            num_threads=0,
            search_complexity=100,
            beam_width=2,
            num_nodes_to_cache=0,
            index_prefix=scale,
        )
