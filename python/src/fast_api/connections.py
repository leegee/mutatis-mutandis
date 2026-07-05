# fast_api/connections.py
# persistent per worker process (NOT cross-process)

import sqlite3

from lib.eebo_faiss import EeboFaissIndex
from lib.eebo_config import FAISS_TIER1_INDEX, ZARR_PATH, JOBS_DB_PATH, CORPUS_TIER2_DB_PATH
from tier2_0_concept_events import ZarrEventLookup # NB Still contains embeddings that might better be reconstructed


_conn_corpus_tier2      = None
_index                  = None
_lookup                 = None
_jobs_conn              = None


# FAISS Index (singleton)
def get_index():
    global _index
    if _index is None:
        _index = EeboFaissIndex.load(FAISS_TIER1_INDEX)
    return _index


# Zarr lookup (singleton per worker)
def get_tier1_zarr_lookup():
    global _lookup
    if _lookup is None:
        _lookup = ZarrEventLookup(ZARR_PATH)
    return _lookup


def get_corpus_tier2_conn():
    global _conn_corpus_tier2
    if _conn_corpus_tier2 is None:
        _conn_corpus_tier2 = sqlite3.connect(CORPUS_TIER2_DB_PATH)
    return _conn_corpus_tier2


def get_corpus_tier2_conn_fresh():
    global _conn_corpus_tier2

    if _conn_corpus_tier2 is not None:
        _conn_corpus_tier2.close()

    _conn_corpus_tier2 = sqlite3.connect(
        # CORPUS_TIER2_DB_PATH
        f"file:{CORPUS_TIER2_DB_PATH}?cache=shared",
        uri=True
    )
    return _conn_corpus_tier2


# JOBS_CONN Connection to the database of jobs.
def get_jobs_conn():
    global _jobs_conn
    if _jobs_conn is None:
        _jobs_conn = sqlite3.connect(
            JOBS_DB_PATH,
            timeout=30,
            isolation_level=None,   # disables implicit transactions
            check_same_thread=False,
        )
        _jobs_conn.execute("PRAGMA read_uncommitted;")
        _jobs_conn.execute("PRAGMA journal_mode=WAL;")
        _jobs_conn.execute("PRAGMA synchronous=NORMAL;")
        _jobs_conn.execute("PRAGMA busy_timeout=5000;")
    return _jobs_conn


def reset_conn_corpus_tier2ections():
    global _conn_corpus_tier2, _index, _lookup
    _conn_corpus_tier2   = None
    _index  = None
    _jobs_conn = None
    _lookup = None


