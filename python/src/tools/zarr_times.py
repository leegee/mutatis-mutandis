#!/usr/bin/env python
"""check_zarr_mtimes.py — compare per-field write timestamps to spot stale (un-cleared) fields."""

from pathlib import Path
from datetime import datetime

from lib.corpus_config import EVENTSTORE_T1_PATH


def field_mtime(field_dir: Path):
    """Latest mtime across all chunk files in a field's directory."""
    files = list(field_dir.rglob("*"))
    files = [f for f in files if f.is_file()]
    if not files:
        return None
    return max(f.stat().st_mtime for f in files)


def main():
    events_dir = Path(EVENTSTORE_T1_PATH) / "events"
    if not events_dir.exists():
        print(f"No 'events' group at {events_dir}")
        return

    rows = []
    for field_dir in sorted(p for p in events_dir.iterdir() if p.is_dir()):
        mtime = field_mtime(field_dir)
        rows.append((field_dir.name, mtime))

    for name, mtime in rows:
        ts = datetime.fromtimestamp(mtime).isoformat(timespec="seconds") if mtime else "EMPTY / no chunk files"
        print(f"{name:<20} {ts}")


if __name__ == "__main__":
    main()
