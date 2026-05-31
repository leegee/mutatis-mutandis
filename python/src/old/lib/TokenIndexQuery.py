from pathlib import Path
import numpy as np
from typing import List, Tuple

from lib.TokenFaissIndex import TokenFaissIndex


class TokenIndexQuery:
    """
    Query wrapper for token-level FAISS index.

    Invariant:
        index positions correspond exactly to token_list lines.
    """

    def __init__(self, index_path: Path, token_list_path: Path):
        self.index = TokenFaissIndex.load(index_path)
        self.tokens = self._load_tokens(token_list_path)

        # critical invariant: 1:1 alignment
        if len(self.tokens) != self.index.ntotal:
            raise ValueError(
                f"Token list ({len(self.tokens)}) does not match index size ({self.index.ntotal})"
            )

    @staticmethod
    def _load_tokens(path: Path) -> List[str]:
        with open(path, "r", encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f]

    def search(self, query_vec: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """
        Returns top-k similar tokens with similarity scores.
        """
        query_vec = np.ascontiguousarray(
            query_vec.reshape(1, -1),
            dtype=np.float32,
        )

        distances, indices = self.index.search(query_vec, k)

        results: List[Tuple[str, float]] = []

        for idx, score in zip(indices[0], distances[0], strict=True):
            if idx == -1:
                continue

            token = self.tokens[idx]
            results.append((token, float(score)))

        return results
