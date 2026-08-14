
# retrieval/parquet_observations.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.dataset as ds

from .models import UInt64Array


class ParquetObservationStore:
    """Resolve stable observation IDs to their non-embedding Parquet data.

    The Parquet corpus is the source of truth for observation provenance.
    ANN indexes contain only positional references back to event IDs, so
    this class deliberately knows nothing about DiskANN, FAISS, or any
    other index implementation.

    Failure mode:
        event_id is not assumed to be unique across the entire corpus
        until the corpus schema/provenance guarantees that invariant.
        A lookup therefore returns all matching rows rather than silently
        selecting one.
    """

    def __init__(
        self,
        corpus_root: str | Path,
    ) -> None:
        self.corpus_root = Path(corpus_root)

        if not self.corpus_root.exists():
            raise FileNotFoundError(
                f"Parquet corpus does not exist: {self.corpus_root}"
            )

        self._dataset = ds.dataset(
            str(self.corpus_root),
            format="parquet",
            partitioning="hive",
        )

        self._columns = tuple(
            field.name
            for field in self._dataset.schema
        )

        self._observation_columns = tuple(
            column
            for column in self._columns
            if not column.startswith("emb_")
        )

        if "event_id" not in self._columns:
            raise ValueError(
                "Parquet corpus does not contain an event_id column"
            )

    @property
    def columns(self) -> tuple[str, ...]:
        """Return the available non-embedding observation columns."""
        return self._observation_columns

    def get(
        self,
        event_id: int,
    ) -> dict[str, Any]:
        """Return one observation by stable event ID.

        Raises:
            KeyError: if no observation exists.
            ValueError: if event_id resolves to multiple observations.
        """

        observations = self.get_many(
            [event_id],
        )

        if not observations:
            raise KeyError(
                f"Observation not found: event_id={event_id}"
            )

        if len(observations) != 1:
            raise ValueError(
                f"event_id={event_id} resolved to "
                f"{len(observations)} observations"
            )

        return observations[0]

    def get_many(
        self,
        event_ids: UInt64Array | list[int] | tuple[int, ...],
    ) -> list[dict[str, Any]]:
        """Return non-embedding observations for the supplied event IDs.

        Results are returned in Parquet scan order rather than input order.
        Use ``get_many_ordered`` when result order matters.
        """

        ids = [
            int(event_id)
            for event_id in event_ids
        ]

        if not ids:
            return []

        unique_ids = sorted(set(ids))

        table = self._dataset.to_table(
            columns=list(self._observation_columns),
            filter=ds.field("event_id").isin(unique_ids),
        )

        return table.to_pylist()

    def get_many_ordered(
        self,
        event_ids: UInt64Array | list[int] | tuple[int, ...],
    ) -> list[dict[str, Any] | None]:
        """Resolve event IDs while preserving the requested order.

        Missing observations are represented by ``None``. Duplicate input
        IDs therefore produce duplicate output entries.
        """

        observations = self.get_many(event_ids)

        by_id: dict[int, dict[str, Any]] = {}

        for observation in observations:
            observation_id = int(observation["event_id"])

            if observation_id in by_id:
                raise ValueError(
                    f"event_id={observation_id} occurs more than once "
                    "in the Parquet observation layer"
                )

            by_id[observation_id] = observation

        return [
            by_id.get(int(event_id))
            for event_id in event_ids
        ]
