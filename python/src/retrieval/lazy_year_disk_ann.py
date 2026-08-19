from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from lib.corpus_logging import logger
from retrieval.diskann_observation_index import DiskANNObservationIndex

SCALES = ("local", "medium", "broad")


class LazyYearDiskANN:
    """
    Lazily opens the three DiskANN indexes belonging to a publication year.

    At most the years explicitly requested by the caller are eligible for
    loading. A loaded year remains resident until evicted.

    DiskANN indexes are disposable geometric resources. Stable observation
    identity and provenance remain in the Tier 1 observation store.
    """

    def __init__(
        self,
        indexes_root: str | Path,
        years,
        *,
        dimensions: int = 768,
        num_threads: int = 0,
        search_complexity: int = 100,
        beam_width: int = 2,
        batch_num_threads: int = 0,
        num_nodes_to_cache: int = 0,
    ) -> None:
        self._root = Path(indexes_root)
        self._years = tuple(sorted(set(int(year) for year in years)))

        self._dimensions = dimensions
        self._num_threads = num_threads
        self._search_complexity = search_complexity
        self._beam_width = beam_width
        self._batch_num_threads = batch_num_threads
        self._num_nodes_to_cache = num_nodes_to_cache

        self._loaded: dict[
            int,
            dict[str, DiskANNObservationIndex],
        ] = {}

    def get(
        self,
        year: int,
    ) -> dict[str, DiskANNObservationIndex]:
        """
        Return the three DiskANN indexes for one year, loading them lazily.
        """
        year = int(year)

        if year not in self._years:
            raise KeyError(
                f"Year {year} was not included when the resource was opened"
            )

        if year not in self._loaded:
            self._loaded[year] = self._load_year(year)

        return self._loaded[year]

    def _load_year(
        self,
        year: int,
    ) -> dict[str, DiskANNObservationIndex]:
        logger.info(
            "[tier2] opening DiskANN indexes for year=%s",
            year,
        )

        jobs: dict[
            str,
            tuple[Path, Path],
        ] = {}

        for scale in SCALES:
            directory = (
                self._root
                / f"year={year}"
                / scale
            )

            event_ids_path = (
                directory
                / f"{scale}_event_ids.npy"
            )

            if not directory.is_dir():
                raise RuntimeError(
                    f"Missing DiskANN directory: {directory}"
                )

            if not event_ids_path.is_file():
                raise RuntimeError(
                    f"Missing DiskANN event-ID mapping: "
                    f"{event_ids_path}"
                )

            jobs[scale] = (
                directory,
                event_ids_path,
            )

        loaded: dict[str, DiskANNObservationIndex] = {}

        with ThreadPoolExecutor(
            max_workers=len(SCALES),
        ) as pool:
            futures = {
                pool.submit(
                    DiskANNObservationIndex,
                    index_directory=directory,
                    event_ids_path=event_ids_path,
                    dimensions=self._dimensions,
                    num_threads=self._num_threads,
                    search_complexity=self._search_complexity,
                    beam_width=self._beam_width,
                    batch_num_threads=self._batch_num_threads,
                    num_nodes_to_cache=self._num_nodes_to_cache,
                    index_prefix=scale,
                ): scale
                for scale, (directory, event_ids_path)
                in jobs.items()
            }

            for future in as_completed(futures):
                scale = futures[future]
                loaded[scale] = future.result()

        return loaded

    def evict(
        self,
        year: int,
    ) -> None:
        """
        Release the indexes associated with one year.

        DiskANN owns the underlying index resources. Dropping the references
        is the cache boundary used by the year-major Tier 2 loop.
        """
        self._loaded.pop(int(year), None)

    def close(self) -> None:
        """Release all currently loaded years."""
        self._loaded.clear()

    def loaded_years(self) -> tuple[int, ...]:
        """Return the years currently resident in memory."""
        return tuple(sorted(self._loaded))

    def available_years(self) -> tuple[int, ...]:
        """
        Return years that have physical DiskANN directories.

        This is independent of the years selected when this resource was
        constructed.
        """
        if not self._root.exists():
            return ()

        years: list[int] = []

        for path in self._root.glob("year=*"):
            if not path.is_dir():
                continue

            try:
                years.append(
                    int(path.name.removeprefix("year="))
                )
            except ValueError:
                continue

        return tuple(sorted(set(years)))
