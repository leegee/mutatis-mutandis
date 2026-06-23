from umap import UMAP
from pydantic import BaseModel
from bertopic.representation import KeyBERTInspired

from lib.macberth import get_macberth_embedder

class RunJobRequest(BaseModel):
    concept: str

class CreateConceptRequest(BaseModel):
    concept: str
    forms: list[str]
    false_positives: list[str] = []

class CreateConceptAndRunRequest(BaseModel):
    concept: str
    forms: list[str]
    false_positives: list[str] = []


embedder = get_macberth_embedder(pooling="mean")

representation_model = KeyBERTInspired(top_n_words=12)

def make_umap(n_docs: int):
    return UMAP(
        n_neighbors=max(2, min(5, n_docs - 1)),
        n_components=min(5, n_docs - 1),
        metric="cosine"
    )
