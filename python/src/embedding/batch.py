from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np


BATCH_FORMAT_VERSION = 2

VALID_SCALES = frozenset({"local", "medium", "broad"})


@dataclass(frozen=True)
class EmbeddingBatch:
    work_id: int
    model_id: int
    model_key: str
    model_revision: str
    scale: str
    event_ids: tuple[int, ...]
    vectors: np.ndarray


def embedding_key(
    model_id: int,
    scale: str,
    event_id: int,
) -> str:
    """Produce a stable identity shared by retries and transport copies."""
    if scale not in VALID_SCALES:
        raise ValueError(f"invalid scale: {scale!r}")

    return f"{model_id}:{scale}:{event_id}"


def _vectors_sha256(vectors: np.ndarray) -> str:
    array = np.ascontiguousarray(vectors, dtype=np.float32)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def write_batch(
    batch: EmbeddingBatch,
    destination: Path,
) -> None:
    if batch.scale not in VALID_SCALES:
        raise ValueError(f"invalid scale: {batch.scale!r}")

    vectors = np.asarray(batch.vectors, dtype=np.float32)

    if vectors.ndim != 2:
        raise ValueError("vectors must be a two-dimensional array")

    if len(batch.event_ids) != vectors.shape[0]:
        raise ValueError(
            "event_ids count must match vector count"
        )

    if vectors.shape[1] <= 0:
        raise ValueError("vectors must have a positive dimension")

    if len(set(batch.event_ids)) != len(batch.event_ids):
        raise ValueError("event_ids must be unique")

    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format_version": BATCH_FORMAT_VERSION,
        "work_id": batch.work_id,
        "model_id": batch.model_id,
        "model_key": batch.model_key,
        "model_revision": batch.model_revision,
        "scale": batch.scale,
        "event_ids": list(batch.event_ids),
        "event_count": len(batch.event_ids),
        "embedding_dimension": vectors.shape[1],
        "dtype": "float32",
        "vectors_sha256": _vectors_sha256(vectors),
    }

    with tempfile.TemporaryDirectory(
        dir=destination.parent
    ) as temporary_directory:
        temporary_directory = Path(temporary_directory)

        vectors_path = temporary_directory / "vectors.npy"
        manifest_path = temporary_directory / "manifest.json"

        np.save(vectors_path, vectors, allow_pickle=False)

        manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        destination.mkdir(parents=True, exist_ok=True)

        vectors_path.replace(destination / "vectors.npy")
        manifest_path.replace(destination / "manifest.json")


def read_batch(source: Path) -> EmbeddingBatch:
    source = source.resolve()

    manifest_path = source / "manifest.json"
    vectors_path = source / "vectors.npy"

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    if not vectors_path.exists():
        raise FileNotFoundError(vectors_path)

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    if manifest.get("format_version") != BATCH_FORMAT_VERSION:
        raise ValueError(
            f"unsupported batch format: "
            f"{manifest.get('format_version')}"
        )

    scale = manifest.get("scale")
    if scale not in VALID_SCALES:
        raise ValueError(
            f"batch manifest contains invalid scale: {scale!r}"
        )

    event_ids_value = manifest.get("event_ids")
    if not isinstance(event_ids_value, list):
        raise ValueError(
            "batch manifest must contain an event_ids list"
        )

    event_ids = tuple(
        int(event_id)
        for event_id in event_ids_value
    )

    if len(set(event_ids)) != len(event_ids):
        raise ValueError("event_ids must be unique")

    if manifest.get("event_count") != len(event_ids):
        raise ValueError(
            "manifest event_count does not match event_ids"
        )

    vectors = np.load(
        vectors_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    if vectors.dtype != np.float32:
        raise ValueError(
            f"unexpected vector dtype: {vectors.dtype}"
        )

    if vectors.ndim != 2:
        raise ValueError("vectors must be two-dimensional")

    if vectors.shape[0] != len(event_ids):
        raise ValueError(
            "vector count does not match event_ids"
        )

    embedding_dimension = manifest.get("embedding_dimension")
    if not isinstance(embedding_dimension, int):
        raise ValueError(
            "batch manifest must contain an integer embedding_dimension"
        )

    if vectors.shape[1] != embedding_dimension:
        raise ValueError(
            "vector dimension does not match manifest"
        )

    if manifest.get("dtype") != "float32":
        raise ValueError(
            "batch manifest dtype does not match float32"
        )

    expected_hash = manifest.get("vectors_sha256")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise ValueError(
            "batch manifest must contain vectors_sha256"
        )

    actual_hash = _vectors_sha256(vectors)

    if actual_hash != expected_hash:
        raise ValueError(
            "vector checksum does not match manifest"
        )

    return EmbeddingBatch(
        work_id=int(manifest["work_id"]),
        model_id=int(manifest["model_id"]),
        model_key=str(manifest["model_key"]),
        model_revision=str(manifest["model_revision"]),
        scale=scale,
        event_ids=event_ids,
        vectors=vectors,
    )
