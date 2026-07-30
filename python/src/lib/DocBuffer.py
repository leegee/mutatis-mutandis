from dataclasses import dataclass

@dataclass
class DocBuffer:
    doc_id: str
    corpus: str
    pub_year: int | None = None

    def __post_init__(self):
        self.tokens = []
        self.vector_ids = []
        self.corpus_token_idxs = []

    @property
    def key(self):
        return (self.corpus, self.doc_id)

    def append(self, token, vector_id, token_idx):
        self.tokens.append(token)
        self.vector_ids.append(vector_id)
        self.corpus_token_idxs.append(token_idx)

    def __bool__(self):
        return bool(self.tokens)

