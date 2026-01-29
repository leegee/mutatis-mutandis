Mutatis Mutandis: Semantic and Ideological Transformation in Early Modern English Pamphlets

PhD Proposal – Digital Humanities
Discipline: Digital Humanities / History of Ideas
Funding Body: AHRC / UKRI

# 1. Summary / Abstract

This project investigates the evolution of key moral, legal, and political concepts in early modern English pamphlets (c.1600–1700) by combining computational modelling with interpretive close reading. Drawing on the frameworks of Quentin Skinner, Reinhart Koselleck, and Ryan Heuser, the research traces Koselleckian conceptual poles—historically contingent opposites such as liberty ↔ authority and conscience ↔ obedience—to reveal asymmetry, tension, and ideological faultlines.

By operationalising dynamic distributional semantics across diachronic slices of TEI-encoded EEBO-TCP pamphlets, the project visualises semantic drift while close reading interprets rhetorical and historical significance. It explores how moral vocabulary was secularised and contested under institutional strain, offering insights into the emergence of modern English political and legal discourse.

# 2. Research Questions

How did key moral and political concepts evolve across early modern pamphlets under social and institutional pressure?

How do Koselleckian poles reveal conceptual asymmetry and ideological faultlines in vernacular discourse?

How can dynamic embeddings and semantic neighbourhood analysis empirically trace these shifts without collapsing historically significant distinctions?

What patterns of secularisation and negotiation in moral vocabulary emerge prior to formal legal codification?

# 3. Originality and Rationale

Moves beyond single-keyword analysis to conceptual poles, capturing tension and asymmetry rather than consensus.

Combines Skinnerian performative theory with Koselleckian conceptual history and Heuser-style computational methods.

Operationalises Sinfield’s ideological faultlines, quantifying and visualising moments of discursive conflict.

Provides a reproducible digital resource for the study of early modern political language, integrating corpus, embeddings, and visualisation.

This approach is innovative in its integration of theory, digital methods, and historiography, offering a bridge between traditional humanities research and computational analysis.

# 4. Methodology

Corpus: TEI-encoded EEBO-TCP Phase I, enriched with metadata (date, genre, printer/place, political alignment where possible).

Computational Methods:

Dynamic embeddings (fastText) per temporal slice, aligned via Orthogonal Procrustes to preserve global semantic structure.

Pole analysis: define canonical pairs (liberty ↔ authority, divine ↔ secular, etc.) and track relative semantic distance, neighbourhood overlap, and trajectory shifts.

Faultline mapping: identify slices of maximal conceptual tension using semantic and co-occurrence metrics.

Visualization:

Time vs Pole Tension Heatmaps

Trajectory plots via UMAP/t-SNE

Co-occurrence networks for ideological clusters

Interactive web interface (PWA) for exploration

Interpretive Methods:

Close reading at points of maximal divergence to interpret rhetorical strategies, illocutionary force, and ideological negotiation.

Software Infrastructure:

FAISS-based retrieval for embedding queries

Docker-deployed reproducible environment

Orthological mapping of spelling variants to maintain historical accuracy

# 5. Feasibility

Existing corpora (EEBO-TCP) and open-source embedding models (fastText, MacBERTh) are compatible with available CPU/GPU resources.

Preliminary pilot embedding analyses demonstrate semantic drift in selected keywords (liberty, conscience, law, authority).

Pipeline design supports reproducible research, including corpus, embeddings, orthological maps, and visualization tools.

# 6. Expected Outputs

PhD thesis integrating computational and interpretive analysis of conceptual poles.

Open-source digital resource: corpus, aligned embeddings, orthological maps, visualisation tools.

Publications in Digital Humanities and History of Ideas journals.

Interactive visualisations for scholarly use, illustrating semantic drift and ideological faultlines.

#  7. Timeline (4-year plan)
| Year | Activities |
|--|--|
| 1 | Corpus curation, metadata extraction, canonical pole definition, pilot embeddings |
| 2	| Full embedding computation, semantic trajectory visualisation, pole/neighbourhood analysis |
| 3	| Close reading integration, faultline mapping, initial publications |
| 4	| Thesis completion, corpus and software release, final dissemination |

## 8. Impact

Academic: advances methodology in Digital Humanities; bridges theory, computational linguistics, and conceptual history.

Public Engagement: interactive visualisations allow wider access to historical debates and the evolution of political concepts.

Future Research: provides a reproducible digital infrastructure for further study of semantic change and ideological negotiation in early modern texts.

## 9. Select Bibliography

Heuser, R. (2020). Historical Semantics and Digital Methods.

Koselleck, R. (2002). The Practice of Conceptual History.

Raymond, J. (2003). Pamphlets and Political Culture in Early Modern England.

Sinfield, A. (1992). Faultlines: Cultural Materialism and the Politics of Interpretation.

Williams, R. (1976). Keywords: A Vocabulary of Culture and Society.

Skinner, Q. (1969). Meaning and Understanding in the History of Ideas.