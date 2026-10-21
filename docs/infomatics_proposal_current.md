# Working PhD Proposal Summary
## Finding People Across Time: Provenance-Preserving Diachronic Search for Historical Descriptions of Appearance

### 1. Research problem

How can researchers find historical individuals whose recorded appearance resembles that of a person with albinism, when the descriptions of those individuals use vocabularies, categories and conventions that differ from those used today?

Historical records describe people through a wide range of physical characteristics: exceptionally pale skin, white or unusually light hair, distinctive eyes and combinations of features that may be recognisable to a modern reader but were not necessarily understood in modern medical terms. Some individuals may have been identified as albinos; others may have been described through the language of natural history, travel writing, religion, literature, portraiture or popular accounts.

A search for the modern term *albinism* will miss people whose appearance was recorded without that label. Conversely, searching for *white hair* or *pale skin* can produce large numbers of irrelevant results, including descriptions of ageing, metaphorical whiteness and conventional literary imagery.

The research problem is therefore one of historical discovery: finding people through descriptions of what they looked like, without presupposing the terminology through which they should be found or the explanation for their appearance.

The motivation is also personal. As a person with albinism, I want to discover historical individuals who may have looked like me, whether or not their contemporaries recognised or named the condition. I also want to investigate how people with white rather than merely grey or hoary hair, and potentially distinctive eyes, have been described and recorded across history.

### 2. Proposed humanities investigation

The historical investigation would focus on the descriptions, depictions and records of individuals whose appearance included some combination of:

- exceptionally pale skin or complexion;
- white or very light hair, including hair described as white from childhood rather than as a consequence of ageing;
- distinctive or unusually described eye colouring or appearance;
- other recorded characteristics that help establish a more complete description of an individual's appearance.

The project would search in both directions.

**First, from a modern point of reference to historical individuals.** Starting with the visible characteristics associated with my own appearance, the research would seek individuals described in historical sources in comparable terms. The sources might include chronicles, biographies, travel narratives, medical and natural-history texts, newspapers, correspondence, portraits and engravings.

**Second, from historical descriptions towards individuals of potential relevance.** Starting with records of people explicitly described as having white hair, exceptional paleness or distinctive eyes, the research would investigate who those people were, how their appearance was documented, and how it was interpreted in their own contexts and subsequently.

The study would examine how individuals were described, depicted, classified and remembered. It would consider how accounts of similar appearances might reflect different historical understandings, including social status, religious interpretation, ideas about human variation, natural history and medical classification.

The project would not presume that every person with white hair or pale skin had albinism. Nor would it retrospectively diagnose individuals on the basis of incomplete descriptions or portraits. Its central object of investigation is the historical person and the surviving evidence of their appearance, not the history of a medical condition considered in isolation.

### 3. Proposed methodological investigation

The computational component would investigate **iterative, provenance-preserving retrieval across chronological periods**.

The central challenge is that the vocabulary used to describe a person's appearance can change substantially across time. A modern query may not retrieve an earlier account because the author used different terminology, described individual features separately, relied on an analogy or recorded the person's appearance within a narrative whose principal subject was something else.

The proposed method would allow passages judged relevant in one period to become candidate retrieval seeds for an earlier period. Contextual representations of these passages would be used to identify potentially related descriptions, which could then support further searches backwards through time.

The approach would combine contextual passage embeddings, lexical and phrase-based search, hybrid retrieval and chronological filtering. It would preserve the provenance of each retrieval step, including the source passage, date, retrieval method, similarity or ranking score, and the query or passage that led to the discovery.

This would allow researchers to trace how a particular individual or description was found, inspect the evidence directly and distinguish a promising lead from an established historical connection.

An LLM could assist with query formulation, candidate interpretation and prioritisation for human review. Its suggestions would remain provisional and traceable to source evidence. The research would test whether such assistance improves discovery rather than assuming that it does.

### 4. Research questions

**Primary humanities question**

How have historical individuals characterised by extraordinary paleness, white hair and potentially distinctive eye characteristics been described, depicted and interpreted, and what can surviving records establish about their appearance and the ways it was understood?

**Primary methodological question**

Can iterative passage-based retrieval across chronological periods discover historical individuals whose recorded appearance resembles a modern researcher's point of reference, when their descriptions are missed by modern terminology, keyword searches and single-step semantic retrieval?

**Supporting questions**

1. Which historical descriptions, depictions and records can be discovered through combinations of physical characteristics rather than searches for a modern medical label?
2. Under what conditions does iterative retrieval improve the discovery of relevant individuals, particularly where evidence is sparse, indirect or expressed in historically different vocabularies?
3. How do retrieval performance and relevance vary with chronological distance, genre, descriptive specificity and the quality of intermediate results?
4. How can the method distinguish a genuine description of white hair from ordinary greying, metaphorical whiteness, conventional imagery and other false positives?
5. How can retrieval provenance and contextual historical analysis distinguish resemblance from historical identification, and both from modern interpretation?

### 5. Research design and evaluation

The project would compare four retrieval approaches:

- **Lexical retrieval:** searches using modern terminology, historical spelling variants, curated terms and phrases for relevant physical characteristics.
- **Single-step semantic retrieval:** a fixed query or passage searched against historical material using contextual embeddings.
- **Hybrid retrieval:** lexical and semantic retrieval combined, with appropriate ranking and deduplication.
- **Iterative diachronic retrieval:** relevant passages from one period used as candidate seeds for searches in earlier periods.

The comparisons would use comparable retrieval budgets and historically informed relevance judgements. Evaluation would consider recall, precision among reviewed results, the number of newly discovered relevant individuals, chronological coverage, retrieval stability and error propagation across successive steps.

The relevance assessment would distinguish different levels of evidence:

- **Direct description:** the source explicitly records one or more relevant physical characteristics.
- **Corroborated identification:** multiple sources or forms of evidence support the identification of an individual and their recorded appearance.
- **Potential resemblance:** the description or depiction appears relevant but is incomplete, ambiguous or open to alternative interpretations.
- **Contextual association:** a passage concerns whiteness, pale appearance or related imagery but does not establish that an actual person's appearance is being described.

These categories would be refined during the research design. They are intended to prevent the evaluation from counting every semantically similar passage as a successful discovery.

Qualitative analysis would examine both successful discoveries and failures. Examples of failure include generic descriptions, mistaken identification of grey hair as white, metaphorical language, irrelevant associations propagated between periods, and the loss of useful evidence through ranking or deduplication.

The research would need a clearly defined corpus, a documented sampling strategy and a transparent procedure for assessing relevance. Historical conclusions about individuals would be established through contextual reading and corroboration, independently of embedding scores.

### 6. Positioning in existing research

The proposal would engage with three overlapping research areas.

**Historical RAG and digital history.** Kim-Baumann and Hiltmann's *HistoRAG* (2026) provides a relevant methodological precedent for incorporating historical research practice into retrieval-augmented generation. Its concerns with temporal coverage, the limitations of vector similarity and the distinction between source discovery and interpretation are directly relevant. It should be treated as a key comparator, rather than presumed to represent the entire state of the art.

**Diachronic semantic search and historical NLP.** Research on semantic change, historical language processing, contextual embeddings and semantic search provides foundations for retrieving material expressed in vocabularies different from those of a modern query. The literature review must establish which existing approaches already implement iterative retrieval, cross-period query transfer or passage-to-passage search, and what remains unresolved.

**Information retrieval and evidence traceability.** Commercial information-retrieval systems, including LexisNexis, offer relevant points of comparison for hybrid retrieval, ranking, graph-based relationships and source provenance. Their methods and evaluations should be examined where publicly documented, without assuming that proprietary systems establish academic evidence for the proposed approach.

The proposed contribution is not simply to combine embeddings, keywords, temporal filters and LLMs. It is to investigate whether contextual evidence transferred between periods improves the discovery of historical individuals through descriptions of their appearance, and to develop an evaluation approach that distinguishes useful discoveries from superficial resemblance.

### 7. Working hypothesis

Iterative passage-based retrieval across chronological periods will recover some historically relevant descriptions and records of individuals that fixed-query lexical, semantic and hybrid retrieval misses. Its benefits will vary with descriptive specificity, evidence type, chronological distance and the quality of intermediate results, and must be demonstrated through historically informed evaluation.

The hypothesis is falsifiable. If iterative retrieval does not improve discovery under fair comparison, or if its apparent gains largely reflect irrelevant associations, that is an important research finding.

### 8. Intended contribution

The project aims to make two connected contributions.

**To historical scholarship:** an investigation of individuals whose recorded appearance includes extraordinary paleness, white hair and potentially distinctive eyes, examining how their appearance was described, represented and interpreted. The study would recover and contextualise historical evidence without assuming that the individuals shared a medical condition or that similar appearances carried the same meaning.

**To historical information retrieval:** an evaluated method for discovering individuals and descriptions across changing vocabularies and periods, preserving the path by which each discovery was made and making the method's successes and failures inspectable.

The wider ambition is to help researchers find people in historical records by starting from what those people looked like, rather than requiring the researcher to know in advance the historical terms or classifications under which they were recorded.

### 9. Outstanding decisions

Before this becomes a formal proposal, the following need to be settled:

1. **Historical scope:** which periods, genres and primary sources offer sufficient evidence for a coherent humanities investigation within a PhD.
2. **Historical framing:** how to formulate a substantive question about the description, representation and recording of individuals, rather than merely assembling a catalogue of people with similar features.
3. **Research contribution:** what the literature already establishes about iterative and cross-period retrieval, and precisely what remains untested.
4. **Corpus feasibility:** whether sufficient reliable, accessible and contextually interpretable evidence exists, including images and records where relevant.
5. **Evaluation design:** how to judge relevance, identify individuals reliably, handle uncertain descriptions and avoid favouring the proposed method.
6. **Disciplinary fit:** whether the application is primarily a humanities PhD with a substantial computational method, or an Informatics PhD in which historical discovery provides a demanding evaluation setting.

The central principle is that the research begins with people and their recorded appearance. Computational retrieval is valuable insofar as it helps uncover historical individuals and evidence that would otherwise remain difficult to find, while leaving identification, interpretation and historical judgement open to scrutiny.
