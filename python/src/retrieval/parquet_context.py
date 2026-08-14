from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.dataset as ds
import numpy as np

from .models import SearchResult
from .parquet_observations import ParquetObservationStore


@dataclass(frozen=True)
class ContextToken:
    corpus: str
    doc_id: str
    token_idx: int
    token: str


@dataclass(frozen=True)
class ObservationContext:
    event_id: int
    distance: float
    observation: dict[str, Any]
    before: tuple[ContextToken, ...]
    after: tuple[ContextToken, ...]

    @property
    def text(self) -> str:
        centre = ContextToken(
            event_id=self.event_id,
            token=str(self.observation["token"]),
            token_idx=int(self.observation["token_idx"]),
        )

        return " ".join(
            token.token
            for token in (
                *self.before,
                centre,
                *self.after,
            )
        )


class ParquetContext:
    """Resolve observations into human-readable token context."""

    def __init__(
        self,
        corpus_root: str | Path,
        *,
        context_before: int = 20,
        context_after: int = 20,
    ) -> None:
        if context_before < 0:
            raise ValueError(
                "context_before must be non-negative"
            )

        if context_after < 0:
            raise ValueError(
                "context_after must be non-negative"
            )

        self._observations = ParquetObservationStore(
            corpus_root,
        )

        self._dataset = ds.dataset(
            str(corpus_root),
            format="parquet",
            partitioning="hive",
        )

        self._context_before = context_before
        self._context_after = context_after

    def get(
        self,
        event_id: int,
        *,
        distance: float = 0.0,
    ) -> ObservationContext:
        """Resolve one observation and its surrounding token context."""

        observation = self._observations.get(
            event_id,
        )

        doc_id = observation["doc_id"]
        token_idx = int(observation["token_idx"])

        start_idx = max(
            0,
            token_idx - self._context_before,
        )

        end_idx = (
            token_idx
            + self._context_after
        )

        table = self._dataset.to_table(
            columns=[
                "event_id",
                "doc_id",
                "token",
                "token_idx",
            ],
            filter=(
                (ds.field("doc_id") == doc_id)
                & (ds.field("token_idx") >= start_idx)
                & (ds.field("token_idx") <= end_idx)
            ),
        )

        rows = sorted(
            table.to_pylist(),
            key=lambda row: int(row["token_idx"]),
        )

        before: list[ContextToken] = []
        after: list[ContextToken] = []

        for row in rows:
            row_token_idx = int(row["token_idx"])

            context_token = ContextToken(
                event_id=int(row["event_id"]),
                token=str(row["token"]),
                token_idx=row_token_idx,
            )

            if row_token_idx < token_idx:
                before.append(context_token)
            elif row_token_idx > token_idx:
                after.append(context_token)

        return ObservationContext(
            event_id=int(event_id),
            distance=float(distance),
            observation=observation,
            before=tuple(before),
            after=tuple(after),
        )

    def get_many(
        self,
        result: SearchResult,
    ) -> list[ObservationContext]:
        """Resolve a SearchResult while preserving ANN result order."""

        event_ids = np.asarray(
            result.event_ids,
            dtype=np.uint64,
        )

        distances = np.asarray(
            result.distances,
            dtype=np.float32,
        )

        if event_ids.ndim != 1:
            raise ValueError(
                "SearchResult event_ids must be one-dimensional"
            )

        if distances.ndim != 1:
            raise ValueError(
                "SearchResult distances must be one-dimensional"
            )

        if len(event_ids) != len(distances):
            raise ValueError(
                "SearchResult event_ids and distances have "
                "different lengths"
            )

        observations = self._observations.get_many_ordered(
            event_ids,
        )

        contexts: list[ObservationContext] = []

        for event_id, distance, observation in zip(
            event_ids,
            distances,
            observations,
        ):
            if observation is None:
                raise KeyError(
                    f"Search result event_id={int(event_id)} "
                    "does not exist in the Parquet observation layer"
                )

            contexts.append(
                self.get(
                    int(event_id),
                    distance=float(distance),
                )
            )

        return contexts

