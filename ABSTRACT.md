# Mutatis Mutandis: Semantic and Ideological Transformation in Early Modern English Pamphlets

PhD Proposal, Digital Humanities / History of Ideas

## 1. Abstract

This project traces the evolution of moral, political, and legal vocabulary in early modern England by combining computational modelling with interpretive close reading. Building on the frameworks of Quentin Skinner, Reinhart Koselleck, and Ryan Heuser, it investigates the semantic drift of key conceptual poles—pairs of historically contingent opposites such as liberty / authority and conscience / obedience—across the pamphlet literature of the seventeenth century.

Using diachronic distributional semantics and orthological mapping, the project renders shifts in meaning empirically visible, while close reading interprets their rhetorical and ideological significance. By situating these shifts within Alan Sinfield’s notion of ideological faultlines, the project aims to illuminate how concepts were repurposed, contested, and secularised under conditions of social and institutional strain, contributing to the emergence of modern English secular law and political discourse.

The methodology combines:

* TEI-encoded corpora (EEBO-TCP Phase I) with rich metadata

* Dynamic fastText embeddings aligned across temporal slices

* FAISS-based retrieval and graph-based visualisation

* Interactive web-based querying and visualisation

* Skinnerian close reading at points of maximal semantic tension

This dual approach demonstrates the explanatory power of computational methods while situating the analysis in a deep historiographical context, providing insights into the historically contingent formation of political concepts and their enduring ideological significance.

## 2. Research Questions

The project addresses the following questions:

* How did key moral and political concepts evolve in early modern English pamphlets under conditions of institutional and ideological strain?

* How do Koselleckian conceptual poles (Begriffspole) reveal asymmetries, tensions, and faultlines in political discourse over time?

* How can distributional semantics and diachronic embeddings empirically trace these shifts without collapsing historically salient distinctions?

* How did secularisation of moral vocabulary unfold as a gradual, uneven process prior to formal consolidation in law?

* What do patterns of semantic drift reveal about the emergence of modern notions of legal and civic authority?

## 3. Context and Rationale
### 3.1 Historiographical Background

The formation of early modern English political vocabulary has long been studied through close reading (Skinner, 1969; Raymond, 2003) and lexicographical work (Williams, 1976). However, much of the field has focused on isolated keywords, leaving the dynamics of conceptual opposition and ideological tension underexplored. Koselleck’s framework of Begriffspole provides a model for understanding the relational and asymmetrical nature of historical concepts, showing how opposing poles define the conceptual space in which debate occurs.

Sinfield’s notion of ideological faultlines further frames the study by identifying moments where competing systems of meaning collide, producing sites of negotiation rather than resolution. Early modern pamphlets, as Joad Raymond demonstrates, were uniquely responsive media where such debates unfolded in real time, before any stabilisation into law or doctrine.

### 3.2 Computational Motivation

Building on Ryan Heuser’s work in diachronic distributional semantics, this project leverages dynamic embeddings to trace semantic drift across temporal slices, while preserving distinctions that smoothing methods risk obscuring. By anchoring canonical terms and analysing the trajectories of surrounding lexical fields, it becomes possible to visualise semantic tension along conceptual poles, providing a computationally rigorous complement to close reading.

## 4. Methodology
### 4.1 Corpus

Primary source: TEI-encoded EEBO-TCP Phase I/II

Secondary sources: scans of pamphlets at the Lincoln College Sr Library, Bodleian.

Metadata: date, genre, printer/place, political alignment (inferred where necessary)

Scope: Pamphlets spanning 1600–1700, covering civil, religious, and legal debates

### 4.2 Conceptual Pole Framework

Define canonical poles based on historical scholarship and corpus evidence

Example poles:

    liberty / authority

    conscience / obedience

    divine / secular

    law / custom

Track semantic trajectories

Dynamic fastText embeddings per temporal slice

Orthogonal Procrustes alignment across slices for global vector consistency

Cosine-distance trajectories to measure drift and pole divergence

Neighbourhood analysis

Identify asymmetrical co-occurrence and embedding overlaps

Operationalise Sinfieldian faultlines as lexical neighbourhood tension

### 4.3 Visualisation

Time vs Pole Tension Heatmaps: x = temporal slices, y = poles, color = semantic distance

Embedding Trajectories: UMAP or t-SNE visualisation of term evolution

Co-occurrence Networks: nodes = words, edges = co-occurrence strength, showing conceptual clusters

Faultline Heatmaps: slices where opposing poles show maximal semantic tension

Geospatial Mapping: printer networks, orthographical markers

### 4.4 Close Reading

Applied selectively at slices with maximal divergence or high pole tension

Focus on illocutionary force, rhetorical strategy, and negotiation of ideological positions

### 4.5 Software Pipeline

Core tools: fastText, FAISS, MacBERTh embeddings, Docker deployment

Web interface: PWA for interactive search and visualisation

Deliverables: orthological maps, canonical dictionaries, embeddings, corpus dump, reproducible analysis pipeline

### 4.6 Note on Methodology

Recent work in digital humanities highlights the interpretive nature of computational keyword extraction. Blanke and Papadopoulou (2025) demonstrate that the choice of algorithm—whether TF‑IDF, BERT‑based, or attention‑driven—significantly shapes which terms emerge as salient, particularly for historically and conceptually contested notions such as “security” and “freedom.” This underscores a key principle guiding our approach: semantic drift and keyword analysis are not purely objective processes but interpretive acts, dependent on model selection, parameterization, and normalization of orthographic variants. By carefully canonicalizing word forms and integrating computational embeddings with historical and conceptual context, our methodology mirrors these insights, producing more robust and transparent mappings of conceptual change over time.

## 5. Significance

Introduces a conceptual-pole approach to Digital Humanities, bridging close reading and computational modelling

Demonstrates how semantic drift can reveal distributed secularisation of moral vocabulary prior to formal legal codification

Provides a reproducible, open-source digital resource for the study of early modern political language

Offers insights relevant to modern constitutional debates, highlighting patterns of ideological negotiation under institutional strain

## 6. Preliminary Work

Pilot embedding analysis of selected keywords (liberty, conscience, law, authority)

Prototype visualization of semantic trajectories and co-occurrence networks

Orthological mapping of early modern spelling variants for accurate cross-slice alignment

## 7. Proposed Timeline
| Year |Activities |
|---|---|
| 1	| Corpus curation, metadata extraction, canonical pole definition, pilot embedding analysis |
| 2	| Full embedding computation, pole trajectory visualisation, network analysis, faultline mapping |
| 3	| Close reading integration, interpretive writing, publication of pilot results, dissertation drafting |
| 4	| Dissertation completion, software and corpus release, conference dissemination, final submission |

## 8. Bibliography (Select)

Heuser, R. (2020). Historical Semantics and Digital Methods.

Koselleck, R. (2002). The Practice of Conceptual History.

Raymond, J. (2003). Pamphlets and Political Culture in Early Modern England.

Sinfield, A. (1992). Faultlines: Cultural Materialism and the Politics of Interpretation.

Williams, R. (1976). Keywords: A Vocabulary of Culture and Society.

Skinner, Q. (1969). Meaning and Understanding in the History of Ideas.

## 9. Expected Outputs

PhD Thesis: integrating computational analysis with close reading of conceptual poles

Digital Corpus Resource: TEI-encoded, aligned embeddings, orthological maps, diachronic geospatial maps

Interactive Visualisations: web-based exploratory tools for scholars

Peer-Reviewed Publications: on semantic drift, ideological faultlines, and DH methodology

## 10. Resources Required

Access to EEBO-TCP Phase II and related corpora (Bodleian, Lincoln College Senior Library's large collection of unscanned pamphlets)

High-performance CPU/GPU resources for embedding computation

PostgreSQL RDBMS for corpus storage and retrieval

Open-source libraries: fastText, FAISS, MacBERTh, UMAP/t-SNE

## 11. Conclusion

This project combines historiography, Digital Humanities methods, and computational linguistics to trace the historical formation and secularisation of moral and political concepts in early modern England. By focusing on Koselleckian poles and Sinfieldian faultlines, it provides a nuanced account of ideological transformation, offering both theoretical innovation and practical digital resources for future scholarship.
