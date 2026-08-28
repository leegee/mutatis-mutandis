import sqlite3
from pathlib import Path
from lib.corpus_db import analysis_db_connection

def get_processed_concepts(path):
    if not path.exists():
        return set()

    con = analysis_db_connection(path)
    con.execute( "PRAGMA journal_mode=WAL" )
    con.execute( "PRAGMA synchronous=NORMAL" )
    con.execute( "PRAGMA busy_timeout=5000" )


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
