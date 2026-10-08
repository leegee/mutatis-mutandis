from dataclasses import dataclass


@dataclass
class DocBuffer:
    doc_id: str
    corpus: str
    pub_year: int | None = None

    def __post_init__(self):
        self.tokens = []
        self.corpus_token_idxs = []

    @property
    def key(self):
        return (self.corpus, self.doc_id)

    def append(self, token: str, corpus_token_idx: int):
        self.tokens.append(token)
        self.corpus_token_idxs.append(corpus_token_idx)

    def __bool__(self):
        return bool(self.tokens)
