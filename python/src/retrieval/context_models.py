# retrieval/context_models.py

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextToken:
    corpus: str
    doc_id: str
    token_idx: int
    token: str
