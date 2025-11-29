Building a digital corpus, a TEI pipeline, a WYSIWYG editor, computational analysis, and corpus-level novelty detection.


Methodology
1. Material Foundations

This project treats each pamphlet as a material manifestation in both time and space, acknowledging its production, dissemination, and reception within early modern England and the broader British Isles. For every pamphlet, I will document:

* Imprint data: printer, publisher, bookseller, city, and date
* Format and physical characteristics (quarto, octavo, signatures, length)
* Provenance and known surviving copies
* Associations with political, religious, or intellectual networks

This grounding ensures that computational analysis is anchored in the historical realities of print culture and aligns with the bibliographical approaches advocated by scholars of early modern pamphlets. By foregrounding materiality, the project situates computational and textual analysis within the broader epistemology of historical and literary scholarship.

2. Digitisation and Text Ingestion

The corpus will comprise pamphlets digitised through existing repositories (e.g., Thomason Tracts, Early English Books Online) supplemented where necessary by high-resolution scans from the Bodleian Library. Recognising the limitations of OCR for seventeenth-century typefaces, a post-processing stage will correct errors through:

* Contextual spell correction leveraging transformer-based models fine-tuned on early modern English corpora
* Pattern-based restoration of early modern orthography (e.g., long-s to s; u/v, i/j distinctions)
* Preservation of original orthography in a reversible mapping to maintain fidelity for humanistic analysis

The digitised texts and associated metadata will be ingested into a structured database (PostgreSQL/JSONB or ElasticSearch) to support scalable querying, retrieval, and integration with computational pipelines.

3. Normalisation and Tokenisation

Early modern English requires bespoke preprocessing. The project will implement:

* Normalisation: consistent handling of spelling variants, contractions, and hyphenation; reversible mappings ensure original forms remain accessible
* Tokenisation: early modern-aware tokeniser with custom rules for punctuation, line breaks, and abbreviations
* Stopword management: augmented with a 17th-century stoplist and domain-specific exclusions
* Ideally: POS tagging using adapted models trained on historical English corpora

These steps ensure the corpus is both machine-readable for computational analysis and scholarly-valid for close reading and textual criticism.

4. Entity and Concept Tagging

The project will annotate pamphlets with automated and semi-automated (and crowd-sourced) tagging, including:

* Named entities: persons (historical, biblical, classical), places, institutions
* Political and religious concepts: liberty, tyranny, covenant, popery, conscience
* Rhetorical and genre markers: petitions, remonstrances, narratives, polemical arguments
* Sentiment and epistemic markers: truth, lies, report, rumour

Tagging will be achieved using:

* Domain-adapted transformer models for NER and conceptual detection
* Pattern-based and dictionary-assisted tagging for rhetorical forms
* Weak supervision (after eg Snorkel) to scale annotation, validated with human (ideally crowd-sourced) review

This combination ensures rich semantic enrichment of the corpus while remaining transparent and reproducible.

5. Retrieval-Augmented Analysis (RAG)

To assist in corpus navigation and discovery of ideological patterns, the project will employ retrieval-augmented generation (RAG) methods. Applications include:

* Identifying clusters of pamphlets sharing key phrases or arguments
* Detecting reuse of rhetoric and thematic influence across the corpus
* Assisting in novelty detection of ideas and terms in political discourse

RAG will function as a research tool rather than a generative endpoint, providing historians and literary scholars with explainable, queryable insights into ideological networks.

6. Geospatial and Temporal Tagging

Every pamphlet will be geospatially situated using:

* Place of printing, publication, and/or sale
* Locations mentioned in the text
* Known distribution routes (eg London to counties to Scotland/Ireland)
* Associated/concurrent/temporarily proximate events, such as battles, riots, or parliamentary actions

Analyses will visualise the spread of ideas across time and space, enabling dynamic mapping of political, religious, and ideological influence. Temporal tagging will allow tracing conceptual evolution across key historical milestones (eg Irish Rebellion 1641, Root and Branch Petition, Pride’s Purge).

7. Inverse Frequency Indexing

To quantify the novelty of each pamphlet, the project will implement an inverse frequency index (IFI):

Terms or phrases rare in prior publications are weighted higher, highlighting innovative rhetoric.

Early appearances of key political concepts will be tracked, providing insight into idea genesis and dissemination.

IFI scores combined with network and geospatial analysis will provide a quantitative measure of ideological influence and propagation

8. Semantic Drift and Conceptual Evolution

Dynamic word embeddings and cosine-distance timelines will track semantic drift of key terms (eg liberty, conscience, covenant, popery). Analyses will:

* Detect shifts in rhetorical and conceptual meaning over the period
* Identify polarisation, adaptation, or radicalisation of language in response to historical events
* Combine quantitative embeddings with qualitative interpretation for historically-grounded insights

8.5. The case for MacBERTh:

MacBERTh covers 1450-1950 ([Hugging Face: emanjavacas/MacBERTh](https://huggingface.co/emanjavacas/MacBERTh?utm_source=chatgpt.com)) ([PDF](https://jdmdh.episciences.org/9690/pdf))

MacBERTh (1450–1950) is the leading transformer-based model for Early Modern and Late Middle English, pretrained from scratch on large historical corpora. Out-of-the-box, it handles OCR correction, tokenisation, lemmatisation, POS tagging, semantic drift analysis, RAG-style clustering (by ideology, author, or phrase), TEI-adjacent tagging, argument-structure extraction, and geo-spatial semantics, while also producing stable phrase- and sentence-level embeddings for semantic neighbour extraction. With fine-tuning or additional pretraining on a target corpus, MacBERTh can perform historically accurate sentiment classification (e.g., godly vs ungodly, liberty vs bondage), political alignment detection (Royalist vs Parliamentarian vs Radical sectary), genre classification (sermon, polemic, petition, newsbook, “answer,” “remonstrance,” “tract,” “diurnall”), and sentence-level nearest-neighbour search to identify recurring rhetorical tropes.

9. Network Analysis

The corpus will be represented as interconnected networks, including:

* Citation and phrase-sharing networks
* Authorial and printer/publisher/sales networks
* Geo-temporal diffusion networks

Community detection and clustering will identify ideological ecosystems, revealing how pamphlets functioned within social, political, and geographic networks.

10. Integration of Materiality and Computation

At all stages, the project maintains a dual framing:

* Material cultural artefacts: pamphlet as historical object situated in time, space, and social network
* Computational objects: text as structured, tokenised, enriched data amenable to advanced analysis

This duality ensures that the project remains both methodologically rigorous and historically meaningful, combining digital humanities innovation with deep engagement in early modern print culture.

11. Historical Interpretation and Theoretical Framing

Finally, computational outputs will inform:

* The spread, mutation, and reception of political and religious ideas
* Ideological innovation and dissemination within fractured print cultures
* Connections between material production, textual rhetoric, and political crisis

By integrating computational and historical methods, the project will contribute new insights into the dynamics of seventeenth-century political communication while demonstrating the potential of DH approaches for early modern studies.

12. Output Summary

Summary: The project delivers a richly annotated digital corpus, tools for editing and computational analysis, and quantitative and visual methods for understanding the dynamics of early modern print culture, ideological innovation, and political communication during a time of national crisis.

This project aims to produce a digitally enhanced corpus of early modern English pamphlets alongside a robust suite of computational and analytical tools that support historical, bibliographical, and literary research. The key outputs include:

Digital Corpus of Pamphlets

A structured, searchable corpus of seventeenth-century pamphlets, sourced from repositories (e.g., Thomason Tracts, EEBO) and high-resolution Bodleian scans.

Each pamphlet annotated with material metadata (printer, publisher, format, provenance) and contextual metadata (political, religious, or intellectual associations).

A TEI-based encoding pipeline supporting digitisation, normalisation, and tokenisation of texts while preserving historical orthography.

Potentially a WYSIWYG editor enabling scholars to annotate, correct, and enrich texts without requiring TEI expertise, integrating both automated and human-in-the-loop tagging.

Entity and concept tagging (persons, places, institutions, political/religious concepts, rhetorical forms, sentiment) using transformer models and weak supervision, and potentially software for manual  tagging.

Retrieval-augmented analysis (RAG) for exploration of ideological clusters, phrase reuse, and thematic influence.

Inverse frequency indexing (IFI) to quantify novelty and trace early occurrences of political concepts.

Semantic drift analysis via dynamic word embeddings and cosine-distance timelines to track conceptual evolution over time.

Network analysis mapping citation, phrase-sharing, authorial, and geo-temporal diffusion networks.

Pamphlets, ideas and drift mapped across time and space to visualise the spread of political and religious thought, distribution networks, and influence patterns.

Computational outputs remain grounded in the material realities of early modern print culture, connecting bibliographical data with textual and semantic analysis.

Enables historically-informed interpretation of ideological innovation, political discourse, and pamphlet circulation.

Traces the origins, spread, and mutation of key political and religious concepts across seventeenth-century England and the British Isles.

Provides a framework for digital humanities research combining rigorous computational methods with close engagement in historical and literary scholarship.

