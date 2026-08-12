#!/usr/bin/env python
"""check_locks.py — see if anything currently holds handles into the stale fields."""

from pathlib import Path
from lib.corpus_config import EVENTSTORE_T1_PATH

for name in ("corpus", "emb_raw"):
    d = Path(EVENTSTORE_T1_PATH) / "events" / name
    print(f"{name}: exists={d.exists()}")
    if d.exists():
        try:
            # crude lock probe: try renaming in place, which Windows refuses if a handle is open
            d.rename(d)
            print("  -> no lock detected via rename probe")
        except PermissionError as e:
            print(f"  -> LOCKED: {e}")
