# lib/zarr_store_dirs.py
from pathlib import Path

EVENTS_GROUP = "events"

def store_dirs(root: Path) -> list[Path]:
    """Return store directories to traverse, supporting both single-store
    and legacy slice-directory layouts."""
    if (root / EVENTS_GROUP).exists():
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir())
