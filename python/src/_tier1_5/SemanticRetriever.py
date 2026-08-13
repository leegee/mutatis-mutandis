class SemanticRetriever:
    def __init__(
        self,
        indexes: Mapping[str, Sequence[ANNIndex]],
        embeddings: EmbeddingStore,
        fusion: ResultFusion,
    ) -> None:
        self.indexes = indexes
        self.embeddings = embeddings
        self.fusion = fusion
