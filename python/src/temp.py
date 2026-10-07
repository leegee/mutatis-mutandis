from pathlib import Path
from tier1.tier1_seeds2events import *

conn = get_connection(application_name="purge")
vw = VectorWriter(Path(LANCE_INDEXES_DIR), scale="medium",
                  model_name=LANCE_MODEL_NAME, bucket_size=LANCE_BUCKET_SIZE)

print(vw.purge_orphans(conn, year_range=(1818, 1818), apply=True))
print(vw.purge_orphans(conn, year_range=(1818, 1818), apply=False))  # expect orphans=0