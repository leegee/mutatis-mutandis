# Event-Led Semantic Analysis of Early Modern English Pamphlet Discourse

## Abstract and Research Programme

This project develops a computational Digital Humanities framework for analysing semantic variation in early modern English pamphlet discourse (EEBO-TCP corpus) using instance-level embedding events and projection-based analysis of distributional semantic fields.

Rather than treating lexical items as stable analytical units or modelling semantic change as vector drift, the project reconceptualises meaning as a **distribution of contextual semantic events**. Each occurrence of a word is treated as a discrete event in embedding space, situated within a local neighbourhood of semantically related terms.

Meaning is therefore not represented as a single vector, centroid, or trajectory, but as a **field of recurrent relational structures across time, genre, and discourse context**.

The central object of analysis is a corpus-wide **semantic event ledger**, in which each token occurrence is preserved with full provenance, embedding representation, and local semantic neighbourhood structure. Higher-level representations such as heatmaps, scatter plots, and temporal comparisons are treated as **projections over this event ledger**, rather than primary analytical objects.

---

## Overview

Historical linguistics has long sought to track how words change meaning over time, but most computational approaches to this problem resolve it too quickly. By averaging together all the contexts in which a word appears - collapsing thousands of individual usages into a single representative point - they discard precisely the variation that might be most historically interesting. A word does not mean one thing in the 1580s and another thing in the 1650s: it means different things in different hands, different genres, different moments of crisis, and existing methods largely cannot see this.

This project proposes a different approach. Rather than treating a word's meaning as something to be summarised, it treats each occurrence as a discrete semantic event - a specific instance of a concept appearing in a specific discursive context, embedded in a specific transformer window, recoverable as a precise geometric object. The Early English Books Online corpus, spanning roughly 1475 to 1700, provides the substrate: a vast and historically consequential archive of printed English covering the Reformation, the Civil War, the Restoration, and the intellectual upheavals that accompanied them. The MacBERTh language model, trained specifically on early modern English, provides the contextual embeddings.

The central technical commitment is to preserve what most pipelines discard. Every occurrence of every concept of interest is stored as an individual observation, keyed by a stable identity that records not just what the word was but where it appeared, in what window, at what position. Geometric proximity between observations - computed using approximate nearest-neighbour search - is then used to recover neighbourhood structure: the relational field surrounding each event in embedding space. It is this neighbourhood structure, rather than any aggregate representation, that becomes the object of analysis.

From this foundation, the research asks a set of connected questions. How do neighbourhood fields shift across temporal slices of the corpus? Do concepts that historical scholarship treats as stable behave differently in embedding space from those it treats as contested? Does semantic fragmentation - the dispersal of a concept's occurrences across distinct geometric regions - precede, accompany, or follow periods of documented historical controversy? And to what extent do discourse conditions such as genre, polemic register, or political crisis shape the local semantic environment in ways that a word-level analysis would miss?

The project is therefore simultaneously a methodological argument and a historical one. The methodological argument is that instance-level neighbourhood structure recovers something that aggregation genuinely loses, and that this matters for how we model semantic change. The historical argument is that the upheavals of the early modern period left traces in distributional geometry that are not fully visible through existing methods - and that recovering them requires treating meaning not as a property of words but as a property of events.

---

## Central Claim

This thesis will demonstrate that instance-level neighbourhood structure in embedding space tracks documented semantic contestation in early modern English political concepts in ways that aggregate distributional methods cannot recover.

The validation of this claim proceeds in two stages: first, against a set of concepts whose semantic histories are well established in the scholarly literature; second, through close reading of the primary texts surfaced by the method, in the tradition of Skinnerian contextual interpretation. Agreement between geometric and philological findings constitutes evidence that the method works; divergence constitutes a finding in its own right, requiring either a revision of the received history or a refinement of the computational model.

---

## Research Focus

The project investigates how key moral, political, and religious concepts behave as distributions of semantic events within pamphlet discourse. Concept selection is guided by three bodies of historical scholarship that provide well-documented accounts of semantic change against which computational findings can be tested.

**Raymond Williams** (*Keywords*, 1976; revised 1983) traces the semantic careers of terms including *culture*, *interest*, *liberal*, *organic*, and *revolution* through close reading of the literary and intellectual record. Williams makes specific claims about when and how these concepts bifurcate, stabilise, or acquire new associations - claims precise enough to generate testable hypotheses about neighbourhood structure in the embedding space.

**Quentin Skinner** (*The Foundations of Modern Political Thought*, 1978; the liberty trilogy) provides granular accounts of *liberty*, *state*, *obligation*, and *sovereignty* in early modern political writing, with particular attention to the rhetorical contexts in which these terms operated. The irony of using Skinnerian close reading to validate a computational method is acknowledged; the thesis will engage directly with whether distributional geometry can capture what Skinner argues can only be recovered through contextual interpretation.

**J. G. A. Pocock** (*The Machiavellian Moment*, 1975; *Virtue, Commerce and History*, 1985) traces the semantic careers of *virtue*, *liberty*, *corruption*, *property*, and *commerce* through early modern English political writing with a precision that makes these ideal test concepts. Crucially, Pocock's sources are substantially EEBO texts, making his arguments directly checkable against the corpus.

**Reinhart Koselleck** (*Geschichtliche Grundbegriffe*, 1972–1997), whose conceptual histories of *Freiheit*, *Revolution*, *Geschichte*, and related terms in German provide a comparative framework, and whose methodological writings on *Begriffsgeschichte* - the history of concepts - offer theoretical grounding for the project's approach to semantic change as a historical phenomenon rather than merely a distributional one. Heuser's use of Koselleck as a validation framework provides a direct methodological precedent.

Core research questions include:

* How do individual occurrences of key concepts behave as local semantic events in embedding space?
* How do neighbourhood fields surrounding these events shift across temporal slices of the EEBO corpus?
* What forms of semantic stability, fragmentation, or reconfiguration emerge when meaning is modelled as distributions of instance-level relational structure?
* How do discourse conditions (genre, polemic intensity, political crisis) shape local semantic environments?
* Where existing drift methods aggregate contextual observations into a single representation per word, what is demonstrably lost? Can instance-level neighbourhood structure recover semantic distinctions that aggregation obscures?
* Do concepts fragment into distinct distributional clusters before, during, or after periods of documented historical controversy - and does the timing of fragmentation correspond to external textual evidence of contested meaning?
* How does semantic neighbourhood structure behave differently for concepts that historical scholarship treats as stable versus those it treats as contested? Does the geometry confirm, complicate, or contradict existing historiographical accounts?
* To what extent is neighbourhood structure a property of the concept, versus a property of the documents in which it appears? Can document-level and concept-level signals be disentangled at the observation layer?
* What is the relationship between lexical co-occurrence (recoverable from the corpus directly) and embedding-space neighbourhood (recoverable from geometric search)? Where they diverge, what does that divergence indicate about the limits of distributional semantics as a historical method?

---

## Methods

### 1. Corpus and Embedding Generation

The EEBO-TCP corpus is processed using contextual language models (MacBERTh-style architectures), producing token-level embeddings for each lexical occurrence. Each embedding encodes local syntactic and semantic context rather than abstract lexical identity. MacBERTh's training on early modern English is essential here: a modern BERT model would impose anachronistic semantic structure on a corpus whose orthography, syntax, and vocabulary differ substantially from contemporary English.

The corpus is segmented into coarse temporal slices, which function as organisational scaffolding for comparative analysis rather than as primary semantic units.

---

### 2. Semantic Event Ledger

Each token occurrence is represented as a **semantic event**, defined as:

* contextual embedding vector
* token form (including orthographic and OCR variation)
* document identifier (EEBO provenance)
* temporal slice identifier
* k-nearest semantic neighbours in embedding space

This produces a complete event ledger in which semantic structure is distributed across instances rather than aggregated at the level of word types or centroids.

---

### 3. Neighbourhood Field Extraction

For each event, k-nearest neighbours are computed in embedding space using normalised dot-product similarity. These neighbours define a **local semantic field**, representing the immediate relational environment of each occurrence.

Meaning is operationalised as the structure of these neighbourhood fields, rather than as a single vector position or trajectory.

---

### 4. Slice-Aware Comparative Analysis

Temporal slices are used only for grouping event distributions. The analysis focuses on changes in neighbourhood field structure across time, rather than movement of aggregated representations.

This avoids assuming semantic homogeneity within slices and preserves internal variation.

---

### 5. Validation Through Philological Close Reading

Computational findings are validated through close reading of the primary texts surfaced by the method. Where the geometry identifies a shift in neighbourhood structure - a fragmentation of *liberty*'s semantic field around the 1640s, for instance - the relevant token occurrences are retrieved from the EEBO texts and subjected to contextual interpretation.

This two-stage process - geometric detection followed by philological interpretation - is the operational form of the dialogue between distributional and contextual methods that the thesis proposes. It is also the answer to the validation problem that besets purely computational approaches: the geometry does not interpret itself, and the close reading provides both a check on the method and a site for historical argument.

---

### 6. Projection-Based Visual Analytics

All higher-level representations are treated as projections over the event ledger:

* scatter plots of embedding space (navigation layer over events)
* heatmaps of concept–neighbour strength (aggregated relational structure)
* event stream views (linear inspection of semantic occurrences)
* document-linked inspection layers (traceability to EEBO texts)

All projections are reversible mappings: no view introduces new semantic entities or assumptions.

---

## Theoretical Contribution

### 1. Semantic Events as the Primary Unit of Analysis

The project replaces the word type or lemma with the **semantic event** as the primary unit of computational historical semantics. Meaning is no longer an attribute of lexical items but a property of situated contextual occurrences.

Semantic change is therefore reformulated as **variation in distributions of events**, rather than drift in word-level representations. This is consistent with the spirit of Skinner's insistence that meaning is always meaning-in-use, and with Koselleck's account of concepts as sites of historical struggle rather than stable semantic containers.

---

### 2. From Trajectories to Relational Fields

Existing diachronic semantic models typically represent meaning as vector drift, centroid movement, or trajectory through embedding space. This project replaces these with a **relational field model**, in which meaning is defined by the stability and instability of neighbourhood structure, the recurrence of local semantic companions, and the reconfiguration of contextual adjacency across time.

Semantic change is thus understood as **structural reorganisation of relational environments**, not geometric displacement. The theoretical warrant for this formulation is Firthian rather than computational: JR Firth's principle that "a word is known by the company it keeps" - stated most directly in *Modes of Meaning* (1957) - is the foundation on which the distributional hypothesis rests, and from which the entire lineage of distributional semantics descends: Firth to Harris, Harris to the vector space models of the 1990s, and from there to word2vec and the contextual embeddings of BERT and its successors.

The argument this project makes is that existing distributional methods are, paradoxically, *insufficiently* Firthian. They invoke the principle of contextual meaning but then aggregate it away - collapsing the very particularity of co-occurrence that Firth was insisting on into a single representative point. Modelling each occurrence as a discrete event, and defining meaning through the structure of its neighbourhood rather than through any averaged representation, is the more faithful realisation of the Firthian programme.

---

### 3. Separation of Semantic Signal from Discourse Composition

By preserving instance-level embeddings and document provenance, the model distinguishes semantic change from genre mixture effects, rhetorical variation, and corpus composition shifts. This addresses a central limitation of distributional historical semantics: the conflation of semantic drift with corpus-level structural variation - a conflation that Pocock and Skinner would recognise as the computational equivalent of reading a change in the archive as a change in the language.

---

### 4. Computational Philology and Full Traceability

Each analytical output is traceable to a vector embedding, a token occurrence, and a document source. This enables a form of **computational philology**, where quantitative claims remain continuously grounded in retrievable textual evidence - and where the movement between the geometric and the textual is not a concession to humanistic scruple but a structural feature of the method.

---

## Positioning within Computational Historical Semantics

### Ryan Heuser and Vector-Based Semantic Change

Heuser's work (with colleagues including Le-Khac) establishes a canonical framework in which semantic change is modelled as word-level vector representations aligned across time and measured as geometric drift in embedding space. Heuser's use of Koselleck as a validation resource - taking concepts whose historical transformation is well-documented and asking whether distributional geometry tracks it - is a direct methodological precedent for this project's approach. This project departs from Heuser's representational model, however, by rejecting the assumption of stable word-level semantic objects. Where Heuser models semantic change as **movement of word representations**, this project models it as **reconfiguration of local semantic fields around repeated occurrences**.

---

### Barbara McGillivray and Contextual Semantic Modelling

McGillivray's work emphasises statistically robust diachronic modelling, careful control of corpus effects, and the integration of linguistic theory with computational methods. This project aligns with that methodological rigour but diverges in representational strategy. Rather than modelling words as aggregated contextual distributions over time, it models meaning at the level of **individual contextual events and their relational neighbourhoods** - a shift from word-level semantic distributions to event-level relational structure.

---

### Firth, Harris, and the Distributional Lineage

The theoretical foundations of this project lie in JR Firth's principle that meaning is constituted through context and co-occurrence - that a word is known by the company it keeps (*Modes of Meaning*, 1957). Zellig Harris's distributional hypothesis formalised this insight: words that occur in similar contexts tend to have similar meanings. This principle underlies every distributional semantic model from the vector space models of the 1990s through to word2vec, GloVe, and the contextual embeddings produced by BERT-family architectures.

The project's claim is that this lineage has not been followed to its logical conclusion. By aggregating contextual observations into a single per-word representation, most distributional methods discard the very co-occurrence particularity that Firth's principle is about. A contextual embedding model like MacBERTh produces a distinct representation for every occurrence of a word in context - which is the Firthian ideal. But the standard analytical move is then to average those representations, which retreats from it. This project refuses that retreat: the event-led architecture is the Firthian programme taken seriously at the level of implementation.

Halliday's development of Firth's work into systemic functional linguistics is a further relevant resource, particularly for the analysis of genre and register variation in the EEBO corpus - dimensions of discourse that shape neighbourhood structure and that the event-led model is specifically designed to preserve rather than average away.

---

### Hamilton, Leskovec & Jurafsky (2016)

Hamilton et al introduce a framework based on static embeddings per time slice, aligned across time and measured with cosine-based shift. This assumes stable word identities, meaning as a point in vector space, and semantic change as displacement of that point. This project explicitly rejects these assumptions: there are no stable word-level semantic objects, only events; meaning is not a point but a **field of local relations**; and change is not displacement but **reorganisation of neighbourhood structure**.

---

### Skinner, Pocock, and the Cambridge School

The Cambridge School of intellectual history, associated above all with Quentin Skinner and JGA Pocock, insists that the meaning of political concepts can only be recovered through meticulous attention to the rhetorical contexts in which they were deployed - the questions they were answering, the opponents they were addressing, the conventions they were observing or subverting. This is in apparent tension with any distributional approach, which derives meaning from statistical patterns rather than from authorial intention or rhetorical situation.

This project does not dissolve that tension but works with it. The computational method surfaces candidate moments of semantic shift; the philological method interrogates whether those moments correspond to genuine historical change in use and context. The dialogue between the two is not a methodological compromise but a research design: each stage disciplines the other.

---

## Synthesis

Across much of computational historical semantics, meaning has been modelled as the movement of lexical representations through embedding space. Across much of intellectual history, meaning has been modelled as the intention of historical agents operating within specific rhetorical contexts. This project proposes that neither framework is sufficient alone, and that the most productive approach is one in which distributional geometry and contextual interpretation are held in productive tension - each generating hypotheses for the other to test.

The result is a formulation of meaning as a continuously reconstituted field of relational structure instantiated across semantic events: recoverable geometrically, interpretable philologically, and grounded throughout in the retrievable evidence of the texts.

This yields three fundamental shifts - from word types to semantic events, from vectors to neighbourhood fields, and from trajectories to distributions of relational structure - and one methodological commitment: that computational claims about historical meaning must remain answerable to the texts from which they are derived.

---

## Contribution to Digital Humanities

This project contributes to Digital Humanities by introducing an event-led framework for semantic analysis in historical corpora; replacing centroid and trajectory models with neighbourhood field representations; enabling fully traceable semantic analysis grounded in textual evidence; establishing a validation methodology that integrates distributional and philological approaches; and reframing diachronic semantics as relational field dynamics rather than lexical drift.

---

## Final Conceptual Claim

Semantic change in early modern pamphlet discourse is best understood not as movement of words through a geometric space, but as the evolving structure of relational semantic fields generated by repeated contextual events across time and discourse conditions - recoverable through the geometry of instance-level embeddings, and interpretable through the close reading of the texts from which those embeddings are derived.
