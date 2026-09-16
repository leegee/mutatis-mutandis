# import_batch.py
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

        self._record_inventory(batch)

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
        if not batch.event_ids:
            raise ValueError("cannot import an empty batch")

        expected_dimension = (
            self._lance_writer.dimensions
        )

        if batch.vectors.shape[1] != expected_dimension:
            raise ValueError(
                f"batch dimension "
                f"{batch.vectors.shape[1]} != "
                f"Lance dimension {expected_dimension}"
            )

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
    ) -> None:
        for event_id in batch.event_ids:
            record_inventory(
                event_id=event_id,
                model_id=batch.model_id,
                embedding_key=(
                    f"{batch.model_id}:{event_id}"
                ),
            )

