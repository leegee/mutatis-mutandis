# Proposal Summary
## Event-Led Semantic Analysis of Early Modern English Pamphlet Discourse

> Lee Goddard MSc
> May 2026

### Research Aim
This PhD project develops a new computational framework for historical semantics that models meaning as a **distribution of contextual semantic events** rather than as aggregated word vectors or drifting centroids. Using the EEBO-TCP corpus (c. 1475–1700) and the MacBERTh early modern English language model, the project investigates how key political and religious concepts behaved in highly polemical pamphlet discourse during the Reformation, Civil War, Interregnum, and Restoration periods.

The central claim is that **instance-level neighbourhood structure** in embedding space can recover patterns of semantic contestation and relational fields that traditional diachronic embedding methods (vector averaging and drift measurement) obscure. This approach aims to bridge distributional semantics with the contextualist traditions of Quentin Skinner, J.G.A. Pocock, and Reinhart Koselleck.

### Background and Motivation
Computational approaches to lexical semantic change have made significant progress, yet many current methods collapse thousands of individual usages into single representations per time slice. As noted in work by Tahmasebi et al. and in SemEval-2020 Task 1 (on which you have published), this aggregation step often sacrifices interpretability and fails to capture fine-grained variation — precisely the kind of variation that is historically significant in periods of intense ideological conflict.

This project addresses that limitation by treating every token occurrence as a discrete **semantic event**, preserving full provenance (document, date, transformer window) and local neighbourhood structure. Meaning is operationalised not as a point in vector space, but as the evolving **relational field** surrounding each event.

### Research Questions
- How do neighbourhood fields around key concepts (e.g. *liberty*, *revolution*, *interest*, *enthusiasm*, *fanatic*) shift across temporal slices of the EEBO corpus?
- To what extent does semantic fragmentation in embedding space precede, coincide with, or follow documented periods of historical controversy?
- Can instance-level analysis recover distinctions between stable and contested concepts that aggregated methods miss?
- How do discourse conditions (genre, polemic intensity, political crisis) shape local semantic environments?

### Key Concepts
The project focuses on politically salient terms whose semantic histories are well documented in the scholarly literature:
- **Revolution** (perhaps from circular motion to political rupture)
- **Interest** (perhaps from financial stake to political advantage)
- **Liberty** and **Prerogative**
- **King** and **Parliament**

These choices allow direct validation against the work of Skinner, Pocock, Raymond Williams, and Koselleck.

### Methodology
The pipeline is structured in two tiers:

**Tier 1 – Contextual Observation Layer**
Token-in-window events are extracted with full metadata and stored in a queryable “semantic event ledger.”

**Tier 2 – Neighbourhood Analysis Layer**
Using FAISS on MacBERTh embeddings, the system computes local semantic neighbourhoods for each event. Rather than projecting directly to token-token graphs (which recent diagnostics showed leads to structural collapse due to dense semantic attractors), the project is moving toward **event–event overlap graphs**. This preserves the geometry produced by contextual embeddings and enables analysis of similarity manifolds, clustering, and temporal reconfiguration of relational fields.

All higher-level outputs (heatmaps, scatter plots, temporal comparisons) are treated as reversible projections over the event ledger. Computational findings are systematically validated through close reading of the specific EEBO documents surfaced by the method.

### Current Progress
I have implemented a working end-to-end pipeline and conducted initial experiments on year-filtered and concept-restricted subsets. Recent diagnostics have revealed important structural properties of contextual embeddings — specifically the tendency of FAISS neighbourhoods to form dense semantic attractor sets — which directly informs the shift toward event-level graph construction. This reflective development process strengthens the methodological contribution of the project.

### Fit with Your Research and Potential Supervision
Your expertise in computational modelling of word meaning in historical corpora, diachronic semantic change, and the challenges of sense aggregation and interpretability makes this project a strong potential fit. The work builds constructively on approaches you and your collaborators have advanced while addressing some of their known limitations through an explicitly event-led and relational-field framework.

I am particularly keen to benefit from your guidance on rigorous evaluation of semantic change detection methods and the integration of quantitative outputs with humanistic interpretation.

### Practical Considerations
- **Corpus**: EEBO-TCP (access already secured via institutional subscription)
- **Model**: MacBERTh (domain-appropriate contextual embeddings)
- **Duration**: 3–4 years
- **Funding**: Preparing an application for AHRC/UKRI or equivalent studentship
- **Co-supervision**: Open to a second supervisor from intellectual history or digital humanities to support the philological validation strand

I would be delighted to send the full (10–12 page) research proposal or to discuss this project in more detail via email or a short call.

Thank you for considering this enquiry.

Lee Goddard MSc

