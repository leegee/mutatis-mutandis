from psycopg import Connection

_vector_cache: dict[tuple[str, int], int] = {}

def lookup_vector_id(conn: Connection, doc_id: str, token_idx: int) -> int:
    """
    Canonical identity resolver for token occurrences.

    Failure mode:
        - If DB is inconsistent or missing row → raises KeyError
    """
    key = (doc_id, token_idx)

    if key in _vector_cache:
        return _vector_cache[key]

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT vector_id
            FROM tokens
            WHERE doc_id = %s AND token_idx = %s
            """,
            (doc_id, token_idx)
        )
        row = cur.fetchone()

    if not row:
        raise KeyError(f"Missing vector_id for {key}")

    vid = row[0]
    _vector_cache[key] = vid
    return vid

def save_vectors(slice_id: str, vecs: list, ids: list):
    import numpy as np
    from lib.mb_paths import vectors_path

    if not vecs:
        return

    X = np.vstack(vecs).astype(np.float32)
    I = np.array(ids, dtype=np.int64)

    path = vectors_path(slice_id)
    np.savez_compressed(path, X=X, I=I)
