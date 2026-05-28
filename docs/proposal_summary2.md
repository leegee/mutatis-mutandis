**Event-Led Semantic Analysis of Early Modern English Pamphlet Discourse**

Lee Goddard MSc
May 2026

### Research Aim
This project develops an **event-based framework** for historical semantics in which meaning is modelled as a distribution over contextual semantic events, rather than as drift between aggregated word representations.

Working with the EEBO-TCP corpus (focusing on 1625–1665) and the MacBERTh early modern English model, the project examines how key political and religious concepts functioned in highly polemical pamphlet discourse during the Civil War, Interregnum, and Restoration.

The central claim is that **semantic change is better captured through the dynamics of instance-level neighbourhood fields** than through conventional time-sliced aggregation of word vectors. The project aims to bridge distributional semantics with the contextualist traditions of Quentin Skinner, J.G.A. Pocock, and Reinhart Koselleck.

### Background and Motivation
Whilst working outside academia for many years, I returned to research through finding a practical limitation to visualising etymological change, which lead to hands-on experimentation with historical texts. This revealed a limitation in existing computational approaches to lexical semantic change: the routine aggregation of thousands of individual usages into single representations per time slice. As discussed in Tahmasebi et al and SemEval-2020 Task 1, this step often reduces interpretability and obscures the fine-grained variation that is most historically significant during periods of ideological conflict.

In response I developed a system that treats each token occurrence as a discrete **semantic event** with full provenance. Meaning is modelled not as a point in vector space, but as the evolving relational neighbourhood surrounding each event.

### Research Questions
- How do instance-level neighbourhood fields around contested concepts shift across temporal slices?
- Does geometric fragmentation in embedding space tend to precede, coincide with, or follow documented periods of political or religious controversy?
- What distinctions between stable and contested concepts become visible only at event granularity?
- How do genre, rhetorical intensity, and political context shape local semantic neighbourhoods?

### Key Concepts
The project explores politically salient terms with rich historiographical traditions:
- **Liberty** and **Prerogative**
- **Divine** and **Temporal**
- **Revolution**
- **Parliament**

These were chosen to enable direct comparison with existing scholarship by Skinner, Pocock, Williams, and Koselleck.

### Methodology
The system is organised in two layers:

**Tier 1 – Contextual Observation Layer**
A semantic event is defined as a single token occurrence represented by its contextual embedding within a fixed window, together with full corpus provenance (document, date, and position). Events are the atomic unit of analysis.

**Tier 2 – Neighbourhood Analysis Layer**
Local semantic neighbourhoods are computed for each event using FAISS.

Event overlap graphs are defined over semantic events as nodes, with edges weighted by the intersection size of their respective k-nearest neighbour sets in contextual embedding space, computed via FAISS retrieval.

In initial experiments, direct projection to token-level graphs often collapses into dense attractor structures.

To preserve instance-level geometry, this approach enables analysis of similarity manifolds, clustering, and temporal reconfiguration through the resulting graph structure.

Exploratory visualisation tools support both aggregated and raw event-level views, with full traceability back to original corpus documents.

### Current Progress
A functioning end-to-end pipeline has been implemented, including:
- Contextual event extraction and Zarr-based storage (Tier 1)
- ANN indexing (FAISS) and neighbourhood retrieval (Tier 2)
- Interactive visualisation tools supporting dual-mode exploration

Early experiments on temporally restricted subsets have revealed strong structural effects, particularly the emergence of dense semantic attractors during periods of heightened political tension.

### Evaluation Strategy
Evaluation will combine quantitative and qualitative approaches:
- **Historical alignment**: Testing whether the method recovers well-documented semantic shifts (e.g., *liberty*, *interest*, *revolution*) discussed by Skinner, Pocock, and others.
- **Philological validation**: Systematic close reading of documents surfaced by the model to assess whether geometric patterns correspond to interpretable rhetorical or ideological changes.
- **Comparative baselines**: Comparison against standard aggregated embedding methods, plus stability testing under corpus perturbation.

The framework will be considered successful if event-level neighbourhood analysis yields stable, interpretable signals not easily recoverable through conventional aggregation.

### Fit with Your Research
Your work on unsupervised lexical semantic change detection (including SemEval-2020 Task 1), sense aggregation challenges, and interpretability in embedding models aligns closely with this project. I am particularly interested in your guidance on rigorous evaluation design and the integration of quantitative outputs with philological interpretation.

### Practical Considerations
- **Corpus**: EEBO-TCP (access secured)
- **Model**: MacBERTh
- **Duration**: 3–4 years full-time
- **Funding**: Preparing an AHRC/UKRI studentship application
- **Co-supervision**: Would you be open to a co-supervisor from intellectual history or early modern studies
