#!/usr/bin/env python
"""find_duplicate_event.py — check whether a given event_id appears more
than once in the Tier1 Zarr store, and with what pub_year values."""

import zarr
import numpy as np

from lib.corpus_config import ZARR_PATH
from lib.zarr_store_dirs import store_dirs

TARGET_EVENT_ID = 6172490477035448692

for store_dir in store_dirs(ZARR_PATH):
    g = zarr.open_group(str(store_dir), mode="r")
    if "events" not in g:
        continue
    e = g["events"]
    eids = np.asarray(e["event_id"][:], dtype=np.int64)
    hits = np.where(eids == TARGET_EVENT_ID)[0]
    if len(hits):
        years = np.asarray(e["pub_year"][:], dtype=np.int16)
        docs = e["doc_id"][:].astype(str)
        for pos in hits:
            print(f"{store_dir}: pos={pos} pub_year={years[pos]} doc_id={docs[pos]}")
