#!/usr/bin/env python3

import zarr
import numpy as np

from lib.corpus_config import EVENTSTORE_T1_PATH
from lib.zarr_store_dirs import store_dirs

min_year = None
max_year = None
events = 0

for store_dir in store_dirs(EVENTSTORE_T1_PATH):
    g = zarr.open_group(str(store_dir), mode="r")

    if "events" not in g:
        continue

    years = g["events"]["pub_year"]

    events += years.shape[0]

    local_min = int(np.min(years))
    local_max = int(np.max(years))

    min_year = local_min if min_year is None else min(min_year, local_min)
    max_year = local_max if max_year is None else max(max_year, local_max)

print(f"Events   : {events:,}")
print(f"Min year : {min_year}")
print(f"Max year : {max_year}")