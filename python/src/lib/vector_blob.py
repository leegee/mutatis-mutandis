from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def vector_to_bytes(
    vector: NDArray[np.float32],
) -> bytes:
    """
    Serialize a float32 vector for PostgreSQL bytea storage.
    """
    return np.asarray(
        vector,
        dtype=np.float32,
    ).tobytes()


def bytes_to_vector(
    data: bytes,
) -> NDArray[np.float32]:
    """
    Deserialize a float32 vector stored as PostgreSQL bytea.
    """
    return np.frombuffer(
        data,
        dtype=np.float32,
    )
