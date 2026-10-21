# Marked Whiteness: Hair, Skin and Eyes in English Writing from the Old English Period to the Nineteenth Century

*Preliminary research notes/outline, second revised draft. Items in \[square brackets\] need your attention. The title is provisional.*

## 1. Summary

This project asks how a recurring cluster of bodily features, namely unusually white hair, very pale skin and unusual, pale or fiery eyes, was described and interpreted in English writing from the Old English period to the nineteenth century, and how those descriptions relate to the later emergence of "albinism" as a medical and racial category. The principal object is the English textual tradition. Biblical and classical material is included where it reached English writers. Welsh, Irish and Norse material is used only where the sources justify it (section 4.3).

The project combines close literary-historical reading with a bounded computational method: diachronic semantic search that works from both ends of the timeline, with every machine-retrieved passage checked by hand. The computational work serves the humanities question of locating a rare image, and is not an end in itself.

The project does not assume that earlier religious or literary representations lead to the later medical category. Whether, where and how the two are connected is one of the things it tests.

## 2. Central question and hypotheses

**Question.** How was extreme bodily whiteness described and interpreted in English writing between the Old English period and the nineteenth century, and what relationships, if any, can be demonstrated between those interpretations and the later medical category of albinism?

**Working hypotheses (to be tested, not assumed).**

1. Extreme bodily whiteness carried several interpretations from early in its textual history, among them the numinous, the wondrous, the ominous, the monstrous, and later the natural-philosophical and medical. These coexisted and competed. The project asks which were available to a writer in a given period and genre, and how their relative prominence changed.
2. Whether the later medical category drew on earlier textual material is an open question. It may have done so in some respects and not in others. The project treats it as a question about specific channels (quotation, allusion, shared authorities, genre conventions) and does not assume a general descent.
3. The meanings of bodily whiteness are historically constructed, and the project does not assume that the cluster of features is a trans-historical symbol or a natural response to the phenotype.

A finding that the earlier and later material are largely unrelated, or that matter-of-fact treatments are rare or absent, is a valid result.

## 3. Object of study

"Extreme bodily whiteness" is the project's own term and is not a single feature. White hair, pale skin, pale or fiery eyes, luminous bodies and generic whiteness are separate categories, and part of the project's work is to say how they relate. This section sets out provisional definitions, to be refined in the pilot annotation (year 1).

### 3.1 Core object

A described human or human-like body (including angelic, divine, supernatural and monstrous bodies) in which:

- white or very pale hair co-occurs with at least one further bodily feature from the cluster (very pale skin, pale, red or flame-like eyes); **or**
- white or very pale hair, or very pale skin, is presented as a peculiarity of that body in its own right.

### 3.2 Associated features, tracked but not defining

Luminosity or radiance of the body; fiery or piercing eyes without accompanying pallor; whiteness of clothing or surroundings; ominous or uncanny reactions to the body. These are annotated as separate tags so that their relationship to the core object can be measured, but their presence alone does not make a passage part of the object.

### 3.3 Excluded

Generic whiteness-as-purity or whiteness-as-holiness symbolism not attached to a described body; white-hair-and-snow/wool similes applied to non-bodies; fair complexion within conventional beauty description.

### 3.4 White hair and age

White hair is conventionally a sign of age, and Daniel 7's Ancient of Days is explicitly aged. The project must therefore separate "white because old" from "white because marked". The approach, to be settled before full retrieval begins:

- Every core-object passage is tagged for age cues (explicit age, implicit age, no age cue, explicit youth).
- The object is defined primarily by co-occurrence with the other features and by the explicit or implicit absence of an age reading. Passages with a strong age cue are retained but analysed separately.
- Revelation 1:14 applies Daniel 7's imagery to a figure who is not presented as aged, and exegetical readings of this transfer (eternity, divinity, wisdom) are part of the evidence for how English writers handled the question. \[Check this against specific commentators and English translations.\]
- The Ancient of Days and the Son of Man are treated as anchors and as reception data (section 7.5), and their interpretations are not assumed to be the object itself.

### 3.5 Controls

Results for the core object are compared with controls chosen to show that the pattern is not simply a generic association of whiteness with holiness or of bodily extremes with wonder: other colour terms applied to hair and skin (red, black, yellow), other bodily extremes (giants, dwarfs, the very old), and white-but-ordinary descriptions (age, snow-covered beards). \[Finalise the control list after the pilot.\]

## 4. Scope and corpora

### 4.1 Primary object: the English textual tradition

English literary, religious, natural-historical and medical writing from the Old English period to the nineteenth century, together with the biblical and classical material that demonstrably reached English writers. The material falls into three tiers, with the weight of the project on the second:

1. **Old English (c. 700–1150).** A bounded component, treated by lexical and collocational search of the Dictionary of Old English Corpus and close reading, since no historical language model is available for this period \[confirm\]. The researcher reads Old English with a dictionary, and where searches are run through modern-English translation or other finding aids, those serve only to locate candidate passages. Every passage is checked against the Old English text, and translation is never used as evidence. This is the same principle that applies to the Welsh results (4.3).
2. **c. 1450–1900.** The principal computational and interpretive corpus: EEBO, ECCO and CLMET, using MacBERTh for the earlier and middle periods and a modern model for the later period (7.4).
3. **Middle English (c. 1150–1450).** This falls between the Old English corpus work and MacBERTh's coverage. It is covered by targeted lexical search and selected close readings, and not by systematic retrieval. \[Decide which Middle English resources to use and whether this tier is realistic within three years.\]

### 4.2 Biblical and apocryphal sources

Revelation 1 and Daniel 7 supply the inherited vocabulary. 1 Enoch 106, which presents Noah's luminous, white-haired birth as a cause of paternal fear, offers an earlier and more ambivalent treatment, but was not available in English before Richard Laurence's 1821 translation of the Ethiopic text. Book of the Watchers material reached English readers earlier through the Greek fragments in Syncellus. \[Verify and cite the specific editions.\] Transmission facts of this kind constrain what can be claimed about what English writers could have known at a given date, and the project treats them as constraints.

Classical ethnography and paradoxography (for example Pliny's and Solinus's accounts of the white-haired, pale-eyed people of Caucasian Albania) are included where they reached English readers and are otherwise treated as comparative.

### 4.3 Non-English material: reception and comparison only

Non-English material is not part of the primary corpus. It enters under one of two explicitly separate relations to the English tradition, and the default is exclusion:

1. **Reception evidence.** Material with a documented channel into English culture within the period, such as nineteenth-century Eddic translation, antiquarianism and the medievalism of figures like William Morris, is primary evidence for the English tradition. This includes English translations of Welsh, Irish and Norse texts, which show translators' vocabulary and the images they gave English readers, and say nothing independent about the source traditions. The prototype's retrieval of Welsh mythological material from the query "hair as white as wool" falls under this heading. \[Add any results you want to cite, and note which translations they come from.\]
2. **Comparative material.** Material without a demonstrable channel is used only where it resolves a specific interpretive problem in the English evidence, by contrast or structural parallel. No claim of descent or equivalence is made. Old Norse and Welsh/Irish material enters on this basis only when the English evidence raises a question it can clarify. The Welsh *gwyn* and the Irish *finn*, whose senses include whiteness, fairness, blessedness and holiness, are an example of the kind of contrast that may be useful, since their breadth contrasts with the later English association of extreme pallor with the uncanny.

On Baldr: Snorri's pale Baldr is often argued to owe something to Christian imagery, and the earlier poetic Edda presents a less strongly pale figure under the same name, with its own connotations of brightness. Establishing independence from Christian sources is outside the project's scope. The image is nonetheless historically real and influential, and it is traced as part of the reception history. Baldr and the Ancient of Days are not presented as two independent witnesses to one complex.

### 4.4 Twentieth-century appropriation

Twentieth-century reception lies outside the period but bears on how the project is framed. Norse whiteness-symbolism was appropriated by racial-nationalist and occult movements (see Goodrick-Clarke on the occult roots of Nazism and Van Deusen on Old Norse reception history) \[confirm the specific works\]. This is one reason the project treats the meanings of bodily whiteness as historically constructed and not inherent. Its aim is to historicise such symbolism, and it does not endorse any use of it.

## 5. Position within existing scholarship

Thomas White's 2012 Manchester thesis, *"Their Whiteness Is Not Like Ours": A Social and Cultural History of Albinism and Albino Identities, 1650–1914*, establishes how albinism emerged as a medico-scientific category through travel writing, natural history, medical classification and racial discourse. The present project asks a different primary question: what is the longer history of the cultural semantics surrounding extreme bodily whiteness in English, and was the later category connected to that history in any specific way? White's account is the history of a category, and this project does not assume that the category inherits earlier material.

Charlotte Baker's work on the representation of albinism in Francophone and African contexts informs the project in two ways. It shows that the meaning of extreme bodily whiteness is produced locally within specific discursive and social conditions, which disciplines the project's own historical claims. And her grounding in critical disability studies offers a model for keeping embodiment, stigma and the lived consequences of visible difference in view even where the evidence is textual.

For the early material, the project draws on Alaric Hall's work on Old English supernatural beings and their radiance and whiteness (his PhD thesis and, I believe, its published form, *Elves in Anglo-Saxon England*, 2007) \[confirm the relevant chapters and that the book is the same work\] and on Jackson Crawford's work on Old Norse colour terminology \[confirm the specific work\].

## 6. Research questions

1. **Construction and change.** How is extreme bodily whiteness described and interpreted in English writing from the Old English period to the nineteenth century, and how do the interpretations available to writers, including their valence (blessed or holy versus cursed or unholy), vary across period and genre?
2. **Discourses.** How do religious, literary, natural-historical and medical discourses treat visibly marked bodies, and where, if anywhere, do symbolic descriptions come to be applied to actual bodily conditions such as albinism?
3. **Selection, salience and the mundane phenotype.** Textual survival favours encounters that provoked reaction. Does the corpus preserve evidence of matter-of-fact encounters with the phenotype, and how do such depictions compare with symbolic ones? (See 7.8.)
4. **Connection and its limits.** What specific channels, such as quotation, allusion, shared authorities or genre conventions, link earlier representations to later ones, and where is no such link demonstrable? Where non-English material is relevant, what does it clarify about what is specific to the English case?
5. **Method.** Can a two-ended diachronic semantic search, anchored on physical description at one end and the modern term at the other, retrieve passages that describe similar bodies without presupposing that they are equivalent or connected?

## 7. Methodology

### 7.1 Overall design

Computational retrieval identifies candidate passages across the corpus at scale. Close reading then interprets them. The aim is to find large-scale patterns in how the object is described and interpreted that conventional reading cannot easily survey, and to subject those patterns to detailed literary and historical analysis.

### 7.2 Lexicon-chain search from both ends, and the convergence principle

A seed lexicon is built from two directions:

- **Forward from the earliest material.** Descriptive anchors drawn from Revelation 1, Daniel 7 and comparable early passages, namely white hair ("like wool", "like snow"), pale skin and pale, flame-like or red eyes. The lexicon is expanded slice by slice through collocates and nearest neighbours.
- **Backward from the modern term.** "Albino" and its historical variants and neighbours (to be compiled from White's thesis and from corpus evidence), expanded backward in the same way.

**Convergence is not connection.** That the two chains overlap, and where, shows only that their vocabularies overlap in the corpus. It is a prompt for close reading and is not evidence of historical connection. Shared stock similes, translation artefacts, generic whiteness vocabulary and the drift of embedding neighbourhoods can all produce overlap without any historical relationship. The project treats the following as evidence for a connection: a documented channel (citation, quotation, allusion, a shared authority), shared genre conventions, or demonstrable reception. Overlap without any of these is reported as coincidence or left undetermined. This principle applies to every stage of the analysis. It also governs the comparison between models (7.4) and any use of non-English material (4.3).

Anchoring on description reduces anachronism, but the category of "albino-like" is the project's own construction and is treated as such.

### 7.3 Passage-level retrieval

Because the features are usually spread across clauses, retrieval works at sentence or short-passage level, with each feature (hair, skin, eyes) annotated separately.

### 7.4 Models, time slices and what exists

- A historical language model (MacBERTh, whose coverage is roughly 1450 to 1950 \[confirm\]) is used for the earlier and middle periods, and a modern model for the late period.
- **No cross-model alignment.** The same queries are run in each model separately, and the project compares the passages retrieved and their neighbours, not the vectors. This is more transparent and avoids the noise of aligning contextual embeddings.
- Time slices are sized by corpus volume, with overlapping windows and genre balancing, since the phenomenon is rare and religious texts would otherwise dominate the early slices.
- Material before the models' coverage (Old English and Middle English) is analysed by lexical and collocational methods on annotated editions. Welsh, Irish and Old Norse material, for which comparable models are not available \[confirm\], enters only under 4.3.

**Status of components.** The project distinguishes what already exists from what must still be built and evaluated:

| Component | Status |
| --- | --- |
| MacBERTh passage retrieval over EEBO/ECCO | Built and tested; to be tuned as the work progresses \[add test results\] |
| Prototype descriptive query ("hair as white as wool") | Run; produced Welsh-translation hits (4.3) |
| Seed lexicons, both ends | To be built (year 1) |
| Chain expansion and convergence analysis | \[State what exists\]; evaluation still needed, under the principle in 7.2 |
| Old English lexical/collocational pipeline | To be built (year 1) |
| Annotation scheme (features, age cues, reaction, quotation status) | To be built and piloted (year 1) |
| Genre balancing and time-slice design | To be designed and tested |
| Precision per period; inter-annotator agreement | To be evaluated (year 2) |

### 7.5 Quotation and reception

Biblical quotation is not filtered out. Each occurrence is tagged as quotation, allusion or independent use, and quotation is treated as reception data in its own right. Statistical summaries report results with and without quotations.

### 7.6 Controls

See 3.5. Results for each control are reported alongside the core object, and differences are interpreted case by case.

### 7.7 Validation

A hand-annotated sample of retrieved passages, double-annotated where possible, checks the retrieval against close reading, and gives precision estimates for each period. Disagreements are reported. The annotated sample is a research output in its own right.

### 7.8 Methodology note: the recording of reaction

The surviving record may preserve reactions to the phenotype rather than ordinary encounters with it. Each occurrence is therefore annotated on two dimensions: whether the text records an explicit reaction at all, and, where it does, its character and intensity (astonishment, veneration, fear, disgust, theological or medical interpretation). Special attention goes to matter-of-fact descriptions, which are the best test of the selection effect. The frequency of reaction in the surviving record can be measured directly. The probability that an unremarkable encounter was recorded at all can only be approached indirectly, and the project keeps these two questions apart when interpreting counts.

## 8. Researcher position and reflexivity

The project is undertaken by a researcher who himself has albinism. This informs the choice of questions and brings an awareness of the consequences of historical representation. It does not determine the meanings the research identifies. The project does not assume that albinism or extreme bodily whiteness has inherent symbolic, moral or cultural value. Its object is historical: the meanings that particular writers, communities and traditions have attached to visibly marked bodies. The methodology is explicitly reflexive, with interpretative claims grounded in textual, linguistic and historical evidence, and with evidence, scholarly interpretation and the researcher's own judgement kept distinct.

## 9. Outputs and a possible three-year plan

| Period | Work |
| --- | --- |
| Year 1 | Settle the definition of the object, including the age question (section 3); corpus assembly and licensing; compile seed lexicons; build the annotation scheme and pilot annotation; Old English lexical work and close reading; literature review |
| Year 2 | Full two-ended retrieval across slices; validation sample; close-reading case studies (biblical reception; Old and Middle English material; early modern and Gothic writing; natural-historical and medical writing) |
| Year 3 | Interpretive synthesis, including the question of connection between earlier and later material; use of reception and comparative material where the English evidence calls for it; thesis writing; release of annotated dataset and code |

Expected outputs: a PhD thesis; an annotated, openly released dataset of passages; a reusable retrieval pipeline; and journal articles in digital humanities and in literary or disability studies.

### 9.1 Beneficiaries and impact

This project examines how extreme bodily whiteness, including the features now associated with albinism, has been represented in English writing. Existing scholarship points mainly to representations of monstrosity, wonder and medical curiosity. The project asks a different question: what range of representations survives in the record, including veneration and matter-of-fact treatment, and what can the record tell us about ordinary lives?

The main beneficiaries beyond academia are people with albinism and their families. Albinism affects roughly 1 in 17,000 people in Europe and North America \[confirm and cite\], yet the cultural history available to them is largely one of stigma. A fuller historical picture, including any positive images the record preserves, could support education and public understanding.

The project will release an openly accessible, annotated collection of the passages it identifies, together with a plain-language summary of its findings. It will seek to consult albinism organisations, such as Standing Voice and national support groups, on how results are presented, and it will share findings with them.

## 10. Risks

- **Sparse data.** The phenomenon is rare. Mitigation: passage-level retrieval, seed-lexicon expansion, and hand annotation in place of statistical claims where counts are low.
- **Model coverage and noise.** OCR errors and spelling variation affect early printed text. Mitigation: use of an EEBO/ECCO-trained model, MacBERTh, and minimal normalisation.
- **Scope.** Three tiers of English material plus reception and comparison is still a lot. Mitigation: the weight of the project is on c. 1450–1900, the Old English work is bounded and lexical, Middle English is targeted, and non-English material is excluded by default (4.3). \[Review after year 1 and cut Middle English or the Old English case studies if necessary.\]
- **Defining the object.** The age problem (3.4) and the line between the core object and associated features may not resolve cleanly. Mitigation: settle the working definition in the year 1 pilot, keep borderline cases as tagged categories, and report results with and without them.
- **Over-reading convergence.** Overlap between lexicon chains may be mistaken for historical connection. Mitigation: the convergence principle (7.2), controls and validation against close reading.
- **Circularity.** Anchors shape results. Mitigation: two-ended search, controls and validation against close reading.

## 11. Supervision

Dr Barbara McGillivray (digital humanities, King's College London) has agreed to co-supervise. \[A primary supervisor in English or a related humanities field, such as medical humanities or disability studies, is still needed. Charlotte Baker (Lancaster) has not yet replied to an approach. An Old English or medieval specialist would now also be valuable given the Old English component.\]
