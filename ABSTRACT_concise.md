# Mutatis Mutandis: Semantic and Ideological Transformation in Early Modern English Pamphlets

## Aims

This project aims to develop a computational framework for modelling semantic change in Early Modern English pamphlet literature (EEBO corpus), with a particular focus on distinguishing genuine lexical semantic drift from variation induced by genre, register, and discourse context. It seeks to move beyond static or slice-based representations of meaning by modelling lexical semantics as diachronic trajectories at the token level, enabling fine-grained analysis of how meanings evolve across time in highly heterogeneous textual environments.

---

## Method

The project employs contextual language models (MacBERTh) to generate token-level embeddings across a temporally segmented EEBO corpus, stored and managed using Zarr-based infrastructure. Approximate nearest neighbour search (FAISS) is used to recover dynamic contextual neighbourhoods for each token instance across time slices. From these retrieval structures, the project derives “semantic behaviour” profiles for lexical items, capturing properties such as neighbourhood stability, dispersion, and cross-slice recontextualisation. These behavioural signatures are then clustered post-hoc to identify emergent regimes of semantic change, rather than relying on pre-defined genre or metadata categories.

---

## Contribution

The project contributes a new empirical framework for studying semantic change in early modern texts that is sensitive to both temporal dynamics and discourse heterogeneity. It provides a method for separating semantic drift from genre- and context-driven variation in historical corpora, offering a more robust basis for interpreting lexical change in pamphlet literature. The approach also produces a reusable analytical pipeline for large-scale diachronic semantic analysis in digital humanities research.

---

## Innovation

The core innovation lies in shifting from static representations of meaning or metadata-dependent models of textual classification to a behavioural model of semantics grounded in token-level diachronic dynamics. By introducing post-hoc clustering of semantic behaviour—rather than embedding space alone—the project reframes genre and discourse structure as emergent properties of semantic change processes. This enables a novel, unsupervised perspective on lexical evolution that integrates distributional semantics with interpretable models of change in historical corpora.
