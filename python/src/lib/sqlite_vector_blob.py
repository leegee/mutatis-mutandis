# sqlite_vector_blob.py

import sqlite3
import numpy as np


def vector_to_blob(vector: np.ndarray) -> sqlite3.Binary:
    """
    Serialize a float32 embedding vector for SQLite storage.
    """
    return sqlite3.Binary(
        np.asarray(vector, dtype=np.float32).tobytes()
    )


def blob_to_vector(blob: bytes) -> np.ndarray:
    """
    Deserialize a float32 embedding vector from SQLite.
    """
    return np.frombuffer(blob, dtype=np.float32)