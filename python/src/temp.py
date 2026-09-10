from lib.corpus_config import (
    LANCE_INDEXES_DIR,
)

import time
import lancedb
import numpy as np

db = lancedb.connect(LANCE_INDEXES_DIR)
table = db.open_table(db.table_names()[0])

q = np.random.random(768).astype('float32')
q /= np.linalg.norm(q)

# Warm up (first call often pays one-time setup cost)
table.search(q, vector_column_name='vector').nprobes(20).limit(60).select(['event_id', '_distance']).to_list()

# Now time several clean, sequential single-query calls
times = []
for _ in range(10):
    t0 = time.perf_counter()
    rows = table.search(q, vector_column_name='vector').nprobes(20).limit(60).select(['event_id', '_distance']).to_list()
    times.append(time.perf_counter() - t0)

print('per-call seconds:', [round(t, 4) for t in times])
print('rows returned:', len(rows))
