import sqlite3
from pathlib import Path

def sqlite_connection(path: Path):
    """
    SQLite settings chosen for concurrent readers during later visualisation.
    """
    con = sqlite3.connect(path)
    con.execute( "PRAGMA journal_mode=WAL" )
    con.execute( "PRAGMA synchronous=NORMAL" )
    con.execute( "PRAGMA busy_timeout=5000" )
    return con


def get_processed_concepts(path):
    if not path.exists():
        return set()

    con = sqlite_connection(path)

    try:
        rows = con.execute("SELECT concept FROM concepts")
        return {
            r[0]
            for r in rows
        }

    except sqlite3.OperationalError:
        return set()
    finally:
        con.close()
