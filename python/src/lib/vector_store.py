import numpy as np
from lib.mb_paths import vectors_path


def load_id_vectors(slice_id: str):
    path = vectors_path(slice_id)

    data = np.load(path, allow_pickle=False)

    vecs = data["vecs"].astype(np.float32)
    ids = data["ids"].astype(np.int64)

    # Invariant: ids must be unique
    if len(ids) != len(set(ids)):
        raise ValueError("NPZ contains duplicate token_occurrence_id values")

    id_to_pos = {int(i): idx for idx, i in enumerate(ids)}

    return vecs, id_to_pos, ids
