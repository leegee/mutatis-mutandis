from __future__ import annotations

from pathlib import Path

import lancedb

from lib.corpus_logging import logger
from tier1.observation_store_api import SCALES
from .lance_observation_index import LanceObservationIndex
from .models import SearchSpace
from .observation_index import ObservationIndex
from .observation_index_store import ObservationIndexStore


class LanceObservationIndexStore(ObservationIndexStore):
    """
    ObservationIndexStore backed by one LanceDB table per embedding scale.

    Tier 1 supplies the authoritative searchable year domain. Lance supplies
    the ANN geometry and query-time filtering.

    The physical Lance tables cover the whole corpus, so SearchSpace does not
    select physical index directories; it determines the filters attached to
    the returned ObservationIndex instances.
    """

    def __init__(
        self,
        lance_root: str | Path,
        *,
        available_years,
        available_scales: tuple[str, ...] = SCALES,
        dimensions: int = 768,
        nprobes: int = 20,
        model: str | None = None,
    ) -> None:
        self._lance_root = Path(lance_root)
        self._available_years_set = {
            int(year)
            for year in available_years
        }
        self._available_scales = tuple(available_scales)
        self._dimensions = dimensions
        self._nprobes = nprobes
        self._model = model

        if not self._available_scales:
            raise ValueError(
                "available_scales must not be empty"
            )

        if not self._available_years_set:
            raise ValueError(
                "available_years must not be empty"
            )

        self._db = lancedb.connect(
            str(self._lance_root)
        )

        self._tables = {}

        for scale in self._available_scales:
            try:
                self._tables[scale] = self._db.open_table(scale)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not open Lance table for scale={scale}: "
                    f"{self._lance_root}"
                ) from exc

        logger.info(
            "[retrieval] opened %d Lance observation indexes",
            len(self._tables),
        )

    def get(
        self,
        space: SearchSpace,
    ) -> list[ObservationIndex]:
        scales = tuple(
            space.resolve_scales(
                set(self._available_scales)
            )
        )

        if not scales:
            raise ValueError(
                "SearchSpace resolves to no available scales"
            )

        years = tuple(
            space.resolve_years(
                self._available_years_set
            )
        )

        year_start = min(years) if years else None
        year_end = max(years) if years else None

        return [
            LanceObservationIndex(
                self._tables[scale],
                dimensions=self._dimensions,
                year_start=year_start,
                year_end=year_end,
                model=self._model,
                nprobes=self._nprobes,
            )
            for scale in scales
        ]
