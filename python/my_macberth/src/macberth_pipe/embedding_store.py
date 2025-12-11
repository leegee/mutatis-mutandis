# embedding_store.py
import json
import numpy as np
from pathlib import Path
from typing import List
from .types import Embeddings, ChunkMeta

def save_embeddings(emb: Embeddings, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)

    np.save(directory / "vectors.npy", emb.vectors.astype(np.float32))

    with open(directory / "ids.json", "w") as f:
        json.dump(emb.ids, f)

    with open(directory / "metas.jsonl", "w") as f:
        for m in emb.metas:
            f.write(json.dumps(m.__dict__) + "\n")


def load_embeddings(directory: Path) -> Embeddings:
    vectors = np.load(directory / "vectors.npy")

    with open(directory / "ids.json") as f:
        ids = json.load(f)

    metas: List[ChunkMeta] = []
    with open(directory / "metas.jsonl") as f:
        for line in f:
            metas.append(ChunkMeta(**json.loads(line)))

    return Embeddings(ids=ids, vectors=vectors, metas=metas)


def append_embeddings(directory: Path, new_emb: Embeddings):
    vec_path = directory / "vectors.npy"
    old_vecs = np.load(vec_path)
    combined = np.vstack([old_vecs, new_emb.vectors])
    np.save(vec_path, combined.astype(np.float32))

    id_path = directory / "ids.json"
    with open(id_path) as f:
        ids = json.load(f)
    ids.extend(new_emb.ids)
    with open(id_path, "w") as f:
        json.dump(ids, f)

    meta_path = directory / "metas.jsonl"
    with open(meta_path, "a") as f:
        for m in new_emb.metas:
            f.write(json.dumps(m.__dict__) + "\n")


class EmbeddingsStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, emb: Embeddings):
        save_embeddings(emb, self.directory)

    def load(self) -> Embeddings:
        return load_embeddings(self.directory)

    def append(self, emb: Embeddings):
        append_embeddings(self.directory, emb)
