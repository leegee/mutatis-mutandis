# embedding/import_batch.py

from __future__ import annotations

from pathlib import Path

from embedding.batch import EmbeddingBatch, read_batch
from embedding.work_queue import complete_work, record_inventory


class BatchImporter:
    """Import completed embedding batches into the local vector store."""

    def __init__(self, lance_writer):
        self._lance_writer = lance_writer

    def import_batch(
        self,
        batch_directory: Path,
        *,
        worker_id: str,
    ) -> None:
        batch = read_batch(batch_directory)

        self._validate_inventory(
            batch,
            worker_id=worker_id,
        )

        self.write_lance(batch)

        self._record_inventory(
            batch,
            worker_id=worker_id,
        )

        complete_work(
            work_id=batch.work_id,
            worker_id=worker_id,
        )

    def _validate_inventory(
        self,
        batch: EmbeddingBatch,
        *,
        worker_id: str,
    ) -> None:
        if not worker_id:
            raise ValueError("worker_id must not be empty")

        if not batch.event_ids:
            raise ValueError("cannot import an empty batch")

        if batch.vectors.ndim != 2:
            raise ValueError(
                f"batch vectors must be two-dimensional, "
                f"got {batch.vectors.ndim} dimensions"
            )

        if batch.vectors.shape[0] != len(batch.event_ids):
            raise ValueError(
                f"batch contains {len(batch.event_ids)} event IDs but "
                f"{batch.vectors.shape[0]} vectors"
            )

        expected_dimension = self._lance_writer.dimensions

        if batch.vectors.shape[1] != expected_dimension:
            raise ValueError(
                f"batch dimension "
                f"{batch.vectors.shape[1]} != "
                f"Lance dimension {expected_dimension}"
            )

        if len(set(batch.event_ids)) != len(batch.event_ids):
            raise ValueError(
                "batch contains duplicate event IDs"
            )

        # Work ownership/model/event membership is validated by
        # record_inventory() immediately before inventory insertion.
        #
        # We deliberately do not duplicate that authoritative check here,
        # because the work may change state between validation and import.

    def write_lance(
        self,
        batch: EmbeddingBatch,
    ) -> None:
        self._lance_writer.write(
            event_ids=batch.event_ids,
            vectors=batch.vectors,
        )

    def _record_inventory(
        self,
        batch: EmbeddingBatch,
        *,
        worker_id: str,
    ) -> None:
        for event_id in batch.event_ids:
            record_inventory(
                work_id=batch.work_id,
                worker_id=worker_id,
                event_id=event_id,
                embedding_key=(
                    f"{batch.model_id}:{event_id}"
                ),
            )
