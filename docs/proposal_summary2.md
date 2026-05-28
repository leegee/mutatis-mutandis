## Event-led Semantic Analysis of Early Modern English Pamphlet Discourse

Lee Goddard MSc
May 2026

### Research Aim
This project develops an event-based framework for historical semantics in which meaning is modelled as a distribution over contextual semantic events, rather than as drift between aggregated word representations.

Working with the EEBO-TCP corpus (focusing on 1625–1665) and the MacBERTh early modern English model, the project examines how key political and religious concepts functioned in highly polemical pamphlet discourse during the Civil War, Interregnum, and Restoration.

The central claim is that semantic change is better captured through the dynamics of instance-level neighbourhood fields than through conventional time-sliced aggregation of word vectors. The project aims to bridge distributional semantics with the contextualist traditions of Quentin Skinner, JGA Pocock and Reinhart Koselleck.

### Background and Motivation
The project originates in a practical limitation I encountered whilst working with historical texts: the routine aggregation of individual token occurrences into single representations per time slice, a step that, as Tahmasebi et al and SemEval-2020 Task 1 document, reduces interpretability and obscures fine-grained variation. This limitation prompted the development of a framework that treats each occurrence as a discrete semantic event with full provenance.

### Research Questions
- How do instance-level neighbourhood fields around contested concepts shift across temporal slices?
- Does geometric fragmentation in embedding space tend to precede, coincide with, or follow documented periods of political or religious controversy?
- What distinctions between stable and contested concepts become visible only at event granularity?
- How do genre, rhetorical intensity, and political context shape local semantic neighbourhoods?

### Key Concepts
The project explores politically salient terms with rich historiographical traditions:
- Liberty / Prerogative
- Divine / Temporal
- Revolution
- Parliament

These were chosen to enable direct comparison with existing scholarship by Skinner, Pocock, Williams, and Koselleck.

<div style="page-break-after: always"></div>

### Methodology
The system is organised in two layers:

**Tier 1 – Contextual Observation Layer**
A semantic event is defined as a single token occurrence represented by its contextual embedding within a fixed window, together with full corpus provenance (document, date, and position). Events are the atomic unit of analysis.

**Tier 2 – Neighbourhood Analysis Layer**
Local semantic neighbourhoods are computed for each event using FAISS.

Event overlap graphs are defined over semantic events as nodes, with edges weighted by the intersection size of their respective k-nearest neighbour sets in contextual embedding space, computed via FAISS retrieval.

Exploratory visualisation tools support both aggregated and raw event-level views, with full traceability back to original corpus documents.

### Current Progress
A functioning end-to-end pipeline has been implemented, including:
- Contextual event extraction and Zarr-based storage (Tier 1)
- ANN indexing (FAISS) and neighbourhood retrieval (Tier 2)
- Interactive visualisation tools supporting dual-mode exploration

### Fit with Your Research
Your work on unsupervised lexical semantic change detection and the sense aggregation challenges documented in SemEval-2020 Task 1 bears directly on the core methodological problem this project addresses. I would particularly value your guidance on evaluation design - specifically, how to construct baselines that fairly test whether event-level neighbourhood analysis recovers signals that aggregated methods genuinely miss, rather than simply producing different ones. I am also keen to discuss how quantitative outputs might be anchored to philological interpretation in a way that is defensible to both communities.

