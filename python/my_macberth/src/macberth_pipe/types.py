from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class ChunkMeta:
    # TODO Finalise this list
    doc_id: str
    chunk_idx: int
    text: str
    start_char: int
    end_char: int
    title: str = ""
    author: str = ""
    year: str = ""
    permalink: str = ""

@dataclass(frozen=True)
class Embeddings:
    ids: List[str]
    vectors: np.ndarray
    metas: List[ChunkMeta]
