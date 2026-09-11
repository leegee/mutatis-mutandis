from pathlib import Path
import shutil

from lib.corpus_logging import logger


INDEX_HEADROOM_FACTOR = 1.5
INDEX_MIN_FREE_GB = 20


def _check_index_disk_space(
    lance_root: Path,
    table_name: str,
    row_count: int,
    dimensions: int = 768,
) -> None:
    """Refuse an index build without conservative Lance-volume headroom.

    IVF index construction can require substantial temporary working space,
    and the exact peak is implementation-dependent. The check therefore
    reserves a multiple of the raw vector payload rather than assuming the
    final index size is the peak requirement.
    """
    table_path = lance_root / table_name

    vector_bytes = row_count * dimensions * 4
    required_bytes = max(
        int(vector_bytes * INDEX_HEADROOM_FACTOR),
        INDEX_MIN_FREE_GB * 1024**3,
    )

    usage = shutil.disk_usage(table_path)

    free_gb = usage.free / 1024**3
    required_gb = required_bytes / 1024**3

    logger.info(
        "[tier1] index disk check: table=%s path=%s "
        "rows=%d vector_payload=%.1f GB free=%.1f GB required=%.1f GB",
        table_name,
        table_path,
        row_count,
        vector_bytes / 1024**3,
        free_gb,
        required_gb,
    )

    if usage.free < required_bytes:
        raise RuntimeError(
            f"Refusing to build index for {table_name}: "
            f"{free_gb:.1f} GB free on {table_path.anchor or table_path}, "
            f"but {required_gb:.1f} GB of headroom is required."
        )
