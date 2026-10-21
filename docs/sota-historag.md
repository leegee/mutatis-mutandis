# HistoRAG and the research opportunity for diachronic search

## 1. What HistoRAG does

Kim-Baumann and Hiltmann (2026), *HistoRAG: Embedding Historical Methodology in Retrieval-Augmented Generation Through Critical Technical Practice*, investigates how historical research methodology can be incorporated into retrieval-augmented generation (RAG). Its central concern is not simply retrieving passages that are semantically similar to a query, but structuring retrieval in ways that better accommodate historians' needs.

The work is relevant in several respects:

- **Temporal coverage:** retrieval can be organised around historical periods rather than allowing the highest-scoring passages to dominate regardless of chronological distribution.
- **Semantic retrieval and relevance:** vector similarity is not equivalent to historical relevance. The paper reports a Spearman correlation of 0.275 between vector similarity and LLM-assessed relevance in its evaluation.
- **Discovery versus interpretation:** retrieved documents should be treated as evidence for historical investigation, not as conclusions. LLM-generated interpretations or intermediate proposals should remain distinguishable from source material.
- **Critical technical practice:** system design and evaluation are themselves methodological decisions that affect what historians can discover.

HistoRAG therefore provides a useful precedent for treating historical retrieval as a methodological problem, rather than merely an engineering task.

**Reference:** Kim-Baumann, N. J., & Hiltmann, T. (2026). *HistoRAG: Embedding Historical Methodology in Retrieval-Augmented Generation Through Critical Technical Practice*. arXiv:2606.18103. https://arxiv.org/abs/2606.18103

## 2. The further problem: diachronic discovery

Temporal coverage and diachronic retrieval are related, but they are not the same problem.

A system can ensure that its results cover several periods without being able to discover historical evidence whose vocabulary, conceptual associations or mode of expression differs substantially from those of the original query.

Suppose a researcher begins with a modern concept or a passage from a later historical period. Earlier writers may have expressed related ideas through different terminology, biblical allusion, metaphor, inherited literary conventions or explanatory frameworks that do not map neatly onto modern categories. A fixed query, even when expanded with lexical neighbours, may fail to retrieve such material. A general-purpose embedding model may also rank passages by superficial resemblance rather than historically meaningful relationships.

This creates a more specific research question:

> Can iterative, passage-based retrieval across chronological periods recover historically relevant evidence that is missed by fixed-query semantic search, lexical expansion and conventional hybrid retrieval?

The proposed distinction is that retrieved passages can become candidate queries for the next chronological period. Instead of repeatedly asking each period the same modern-language question, the system carries forward contextual representations of material already judged relevant, searching for earlier passages that may express related ideas in different terms.

This is a hypothesis to investigate, not an established advantage. Passage-to-passage retrieval could also propagate irrelevant associations, amplify an initial retrieval error or mistake semantic resemblance for historical continuity.

## 3. What could be investigated beyond HistoRAG?

### 3.1 Fixed-query retrieval versus iterative diachronic retrieval

Compare at least four approaches:

1. **Lexical retrieval:** exact terms, historically appropriate spelling variants and curated phrase lists.
2. **Single-step semantic retrieval:** a fixed modern query or source passage searched against each historical period.
3. **Hybrid retrieval:** lexical and semantic retrieval combined, with appropriate ranking and deduplication.
4. **Iterative diachronic retrieval:** relevant passages retrieved in one period are used as additional retrieval seeds in the next earlier period.

This comparison would test whether iterative retrieval contributes anything beyond stronger query formulation, semantic expansion or hybrid ranking.

### 3.2 Temporal organisation versus historical connection

Temporal quotas can improve the chronological distribution of results, but chronological distribution alone does not establish a meaningful connection between passages.

Investigate whether retrieval edges between passages across periods can be used to represent *candidate relationships for historical investigation*. Each edge should retain provenance: source passage, target passage, dates, retrieval method, similarity score and the query or seed that produced it.

These edges must not be treated as evidence that a concept was transmitted, that meanings were equivalent, or that one passage historically influenced another. Those conclusions require independent contextual and historical analysis.

### 3.3 Retrieval under lexical and evidential sparsity

A particularly important test case is a concept with relatively abundant evidence in some periods but sparse, indirect or differently framed evidence in others.

In a dense corpus, clusters and distributions may reveal recurring usages. In a sparse corpus, the central problem is often discovering relevant evidence in the first place. The system may need to retrieve passages about an image, description, analogy or textual tradition without assuming that historical authors shared the researcher's modern category.

The investigation should therefore distinguish:

- direct lexical evidence;
- contextual or descriptive evidence;
- possible intertextual or traditional relationships;
- modern interpretations of historical material.

These categories should be established and assessed by historically informed researchers, rather than inferred from embedding similarity alone.

### 3.4 Human evaluation and error propagation

Evaluate not only whether the system retrieves relevant passages, but also how relevance changes as retrieval moves further backwards in time.

Useful measures include recall at a fixed review budget, precision among retrieved passages, the proportion of newly discovered relevant evidence, chronological coverage, and the number of steps through which useful retrieval can be sustained. Human relevance judgements should distinguish direct evidence from plausible but uncertain associations.

Compare iterative retrieval against strong fixed-query and hybrid baselines under comparable retrieval budgets. Analyse failure cases, including semantic drift, recurring generic imagery, irrelevant chains and the loss of relevant material through ranking or deduplication.

An LLM could help propose search terms, summarise candidate relationships or prioritise passages for review. Its proposals should remain explicitly provisional and traceable to the underlying evidence.

## 4. A possible research contribution

The defensible contribution would not be simply to combine embeddings, lexical search, temporal filtering and an LLM. Those components already exist in various forms.

Instead, the project could investigate whether **provenance-preserving, passage-to-passage retrieval across chronological periods provides a measurable advantage for discovering historically relevant evidence expressed through changing vocabularies and modes of representation**.

HistoRAG provides a useful methodological precedent, while the proposed work would focus more narrowly on the retrieval process itself: how candidate evidence is carried across periods, when that process helps, when it fails, and how its results can be evaluated without equating semantic similarity with historical equivalence.

The humanities case would remain essential. A substantive historical investigation—for example, how extraordinary bodily whiteness was represented and interpreted across different textual traditions—could provide the evidence through which the retrieval method is tested. The research would need to establish historical relevance through contextual reading, textual relationships and relevant scholarship, rather than presupposing that all retrieved passages describe the same phenomenon.

## 5. Questions for the literature review

Before claiming novelty, establish:

- Whether existing diachronic semantic-search systems already use retrieved passages as seeds for searches in earlier periods.
- Whether query expansion, pseudo-relevance feedback, iterative retrieval or temporal knowledge-retrieval methods already implement equivalent procedures.
- What HistoRAG's published framework and implementation actually do regarding temporal retrieval, query expansion, embedding selection and evaluation.
- Whether prior work evaluates retrieval quality specifically for sparse, indirect or historically non-equivalent evidence.
- How to construct a credible human-judged benchmark that distinguishes historical relevance from surface-level semantic similarity.

**Working hypothesis:** Iterative passage-based retrieval across chronological periods will recover some historically relevant evidence missed by fixed-query lexical, semantic and hybrid retrieval. Its benefit will depend on the evidence type, chronological distance and quality of intermediate results, and will need to be demonstrated through historically informed evaluation.

The key question is not whether iterative retrieval produces more passages, but whether it helps historians find additional, defensible evidence that they would otherwise have missed.