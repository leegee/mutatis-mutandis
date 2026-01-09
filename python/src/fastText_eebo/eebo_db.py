import sqlite3
import sys
import eebo_config as config

try:
    dbh = sqlite3.connect(config.DB_PATH)
except Exception as exc:
    print(f"[ERROR] Cannot open database: {exc}")
    sys.exit(1)

