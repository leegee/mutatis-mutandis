from __future__ import annotations

from pathlib import Path

import numpy as np
import diskannpy

from lib.corpus_config import DISKANN_INDEXES_DIR


def validate_mapping(
    event_ids_path: Path,
    expected_count: int,
) -> list[str]:
    errors: list[str] = []

    if not event_ids_path.exists():
        return [f"missing event-ID mapping: {event_ids_path}"]

    try:
        event_ids = np.load(event_ids_path, mmap_mode="r")
    except Exception as exc:
        return [f"cannot load event-ID mapping: {exc}"]

    if event_ids.dtype != np.uint64:
        errors.append(
            f"event-ID mapping dtype is {event_ids.dtype}, "
            "expected uint64"
        )

    if event_ids.ndim != 1:
        errors.append(
            f"event-ID mapping has shape {event_ids.shape}, "
            "expected one-dimensional"
        )

    if len(event_ids) != expected_count:
        errors.append(
            f"event-ID mapping contains {len(event_ids)} IDs, "
            f"index contains {expected_count} points"
        )

    if len(event_ids) != len(np.unique(event_ids)):
        errors.append("event-ID mapping contains duplicate IDs")

    return errors


def validate_index(index_directory: Path) -> list[str]:
    errors: list[str] = []

    prefix = index_directory.name

    if not index_directory.is_dir():
        return [f"not a directory: {index_directory}"]

    index_file = index_directory / f"{prefix}_disk.index"
    if not index_file.exists():
        errors.append(f"missing DiskANN index: {index_file}")

    event_ids_path = index_directory / f"{prefix}_event_ids.npy"

    if not index_file.exists():
        return errors

    try:
        diskannpy.StaticDiskIndex(
            index_directory=str(index_directory),
            num_threads=0,
            num_nodes_to_cache=0,
            distance_metric="l2",
            vector_dtype=np.float32,
            dimensions=768,
            index_prefix=prefix,
        )
    except Exception as exc:
        errors.append(f"DiskANN failed to open: {exc}")
        return errors

    try:
        num_points = index.get_num_points()
    except Exception as exc:
        errors.append(f"could not determine point count: {exc}")
        return errors

    errors.extend(
        validate_mapping(
            event_ids_path,
            num_points,
        )
    )

    return errors


def main() -> None:
    print("DISKANN INDEX VALIDATION")
    print(f"root: {DISKANN_INDEXES_DIR}")
    print()

    if not DISKANN_INDEXES_DIR.exists():
        raise SystemExit(f"Index root does not exist: {DISKANN_INDEXES_DIR}")

    index_directories = sorted(
        path
        for path in DISKANN_INDEXES_DIR.rglob("*")
        if path.is_dir()
        and (path / f"{path.name}_disk.index").exists()
    )

    if not index_directories:
        raise SystemExit("No DiskANN indexes found")

    total = len(index_directories)
    passed = 0
    failed = 0

    for number, index_directory in enumerate(
        index_directories,
        start=1,
    ):
        relative = index_directory.relative_to(DISKANN_INDEXES_DIR)

        print(
            f"[{number}/{total}] {relative}",
            flush=True,
        )

        errors = validate_index(index_directory)

        if errors:
            failed += 1

            for error in errors:
                print(f"  FAIL: {error}")

        else:
            passed += 1
            print("  OK")

    print()
    print("SUMMARY")
    print(f"  indexes: {total}")
    print(f"  passed:  {passed}")
    print(f"  failed:  {failed}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
