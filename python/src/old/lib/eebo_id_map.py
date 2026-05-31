# lib/eebo_id_map.py
from typing import Dict, List
import json

from lib.eebo_config import FAISS_ID_TO_EEBO_DOC_ID

class EEBOIDMap:
    def __init__(self):
        self.doc_id_to_index: Dict[str, int] = {}
        self.index_to_doc_id: List[str] = []

    def get_numeric_id(self, doc_id: str) -> int:
        """Assign a stable numeric ID to an EEBO doc_id."""
        if doc_id not in self.doc_id_to_index:
            idx = len(self.index_to_doc_id)
            self.doc_id_to_index[doc_id] = idx
            self.index_to_doc_id.append(doc_id)
        return self.doc_id_to_index[doc_id]

    def save(self) -> None:
        with open(FAISS_ID_TO_EEBO_DOC_ID, "w", encoding="utf-8") as f:
            json.dump(self.index_to_doc_id, f, ensure_ascii=False)

    def load(self) -> None:
        if not FAISS_ID_TO_EEBO_DOC_ID.exists():
            return
        with open(FAISS_ID_TO_EEBO_DOC_ID, "r", encoding="utf-8") as f:
            self.index_to_doc_id = json.load(f)
        self.doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(self.index_to_doc_id)}
