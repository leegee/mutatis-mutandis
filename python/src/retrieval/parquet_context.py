from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lib.corpus_db import get_connection

from .context_models import ContextToken
from .models import SearchResult
from .parquet_observations import ParquetObservationStore
from .postgres_token_store import PostgresTokenStore


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
            corpus=str(self.observation["corpus"]),
            doc_id=str(self.observation["doc_id"]),
            token_idx=int(self.observation["token_idx"]),
            token=str(self.observation["token"]),
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

        self._tokens = PostgresTokenStore(
            get_connection(),
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

        corpus = str(observation["corpus"])
        doc_id = str(observation["doc_id"])
        token_idx = int(observation["token_idx"])

        rows = self._tokens.get_context(
            corpus=corpus,
            doc_id=doc_id,
            token_idx=token_idx,
            before=self._context_before,
            after=self._context_after,
        )

        before: list[ContextToken] = []
        after: list[ContextToken] = []

        for context_token in rows:
            if context_token.token_idx < token_idx:
                before.append(context_token)
            elif context_token.token_idx > token_idx:
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
