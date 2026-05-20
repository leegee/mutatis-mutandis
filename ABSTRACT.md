# Event-Led Semantic Analysis of Early Modern English Pamphlet Discourse

## Abstract and Research Programme

This project develops a computational Digital Humanities framework for analysing semantic variation in early modern English pamphlet discourse (EEBO-TCP corpus) using instance-level embedding events and projection-based analysis of distributional semantic fields.

Rather than treating lexical items as stable analytical units or modelling semantic change as vector drift, the project reconceptualises meaning as a **distribution of contextual semantic events**. Each occurrence of a word is treated as a discrete event in embedding space, situated within a local neighbourhood of semantically related terms.

Meaning is therefore not represented as a single vector, centroid, or trajectory, but as a **field of recurrent relational structures across time, genre, and discourse context**.

The central object of analysis is a corpus-wide **semantic event ledger**, in which each token occurrence is preserved with full provenance, embedding representation, and local semantic neighbourhood structure. Higher-level representations such as heatmaps, scatter plots, and temporal comparisons are treated as **projections over this event ledger**, rather than primary analytical objects.

---

## Research focus

The project investigates how key moral, political, and religious concepts (e.g. *liberty*, *authority*, *conscience*, *obedience*) behave as distributions of semantic events within pamphlet discourse.

Core questions include:

* How do individual occurrences of key concepts behave as local semantic events in embedding space?
* How do neighbourhood fields surrounding these events shift across temporal slices of the EEBO corpus?
* What forms of semantic stability, fragmentation, or reconfiguration emerge when meaning is modelled as distributions of instance-level relational structure?
* How do discourse conditions (genre, polemic intensity, political crisis) shape local semantic environments?

---

## Methods

### 1. Corpus and embedding generation

The EEBO-TCP corpus is processed using contextual language models (MacBERTh-style architectures), producing token-level embeddings for each lexical occurrence. Each embedding encodes local syntactic and semantic context rather than abstract lexical identity.

The corpus is segmented into coarse temporal slices, which function as organisational scaffolding for comparative analysis rather than as primary semantic units.

---

### 2. Semantic event ledger

Each token occurrence is represented as a **semantic event**, defined as:

* contextual embedding vector
* token form (including orthographic and OCR variation)
* document identifier (EEBO provenance)
* temporal slice identifier
* k-nearest semantic neighbours in embedding space

This produces a complete event ledger in which semantic structure is distributed across instances rather than aggregated at the level of word types or centroids.

---

### 3. Neighbourhood field extraction

For each event, k-nearest neighbours are computed in embedding space using normalised dot-product similarity. These neighbours define a **local semantic field**, representing the immediate relational environment of each occurrence.

Meaning is operationalised as the structure of these neighbourhood fields, rather than as a single vector position or trajectory.

---

### 4. Slice-aware comparative analysis

Temporal slices are used only for grouping event distributions. The analysis focuses on changes in neighbourhood field structure across time, rather than movement of aggregated representations.

This avoids assuming semantic homogeneity within slices and preserves internal variation.

---

### 5. Projection-based visual analytics

All higher-level representations are treated as projections over the event ledger:

* scatter plots of embedding space (navigation layer over events)
* heatmaps of concept–neighbour strength (aggregated relational structure)
* event stream views (linear inspection of semantic occurrences)
* document-linked inspection layers (traceability to EEBO texts)

All projections are reversible mappings: no view introduces new semantic entities or assumptions.

---

## Theoretical contribution

### 1. Semantic events as the primary unit of analysis

The project replaces the word type or lemma with the **semantic event** as the primary unit of computational historical semantics. Meaning is no longer an attribute of lexical items but a property of situated contextual occurrences.

Semantic change is therefore reformulated as **variation in distributions of events**, rather than drift in word-level representations.

---

### 2. From trajectories to relational fields

Existing diachronic semantic models typically represent meaning as:

* vector drift
* centroid movement
* trajectory through embedding space

This project replaces these with a **relational field model**, in which meaning is defined by:

* stability and instability of neighbourhood structure
* recurrence of local semantic companions
* reconfiguration of contextual adjacency across time

Semantic change is thus understood as **structural reorganisation of relational environments**, not geometric displacement.

---

### 3. Separation of semantic signal from discourse composition

By preserving instance-level embeddings and document provenance, the model distinguishes semantic change from:

* genre mixture effects
* rhetorical variation
* corpus composition shifts

This addresses a central limitation of distributional historical semantics: the conflation of semantic drift with corpus-level structural variation.

---

### 4. Computational philology and full traceability

Each analytical output is traceable to:

* vector embedding
* token occurrence
* document source

This enables a form of **computational philology**, where quantitative claims remain continuously grounded in retrievable textual evidence.

---

## Positioning within computational historical semantics

### Ryan Heuser and vector-based semantic change

Heuser’s work (with colleagues including Le-Khac) establishes a canonical framework in which semantic change is modelled as:

* word-level vector representations
* alignment across time
* measurement of geometric drift in embedding space

This project departs from this model by rejecting the assumption of stable word-level semantic objects. Instead, it treats meaning as a **distribution of instance-level semantic events**, where no stable centroid or lexical vector is assumed.

Where Heuser models semantic change as **movement of word representations**, this project models it as **reconfiguration of local semantic fields around repeated occurrences**.

---

### Barbara McGillivray and contextual semantic modelling

McGillivray’s work emphasises:

* statistically robust diachronic modelling
* careful control of corpus effects
* integration of linguistic theory and computational methods

This project aligns with this methodological rigour but diverges in representational strategy.

Rather than modelling words as aggregated contextual distributions over time, it models meaning at the level of **individual contextual events and their relational neighbourhoods**.

The key shift is from:

> word-level semantic distributions
> to
> event-level relational structure

---

### Hamilton, Leskovec & Jurafsky (2016)

Hamilton et al. introduce a framework based on:

* static embeddings per time slice
* alignment across time
* cosine-based measurement of semantic shift

This assumes:

* stable word identities
* meaning as a point in vector space
* semantic change as displacement of that point

This project explicitly rejects these assumptions:

* there are no stable word-level semantic objects, only events
* meaning is not a point but a **field of local relations**
* change is not displacement but **reorganisation of neighbourhood structure**

---

## Synthesis

Across much of computational historical semantics, meaning has been modelled as:

> movement of lexical representations through embedding space

This project proposes a different formulation:

> meaning as a continuously reconstituted field of relational structure instantiated across semantic events

This yields three fundamental shifts:

1. from word types → semantic events
2. from vectors → neighbourhood fields
3. from trajectories → distributions of relational structure

---

## Contribution to Digital Humanities

This project contributes to Digital Humanities by:

* introducing an event-led framework for semantic analysis in historical corpora
* replacing centroid and trajectory models with neighbourhood field representations
* enabling fully traceable semantic analysis grounded in textual evidence
* unifying clustering, neighbour analysis, and visualisation under a single event-led architecture
* reframing diachronic semantics as relational field dynamics rather than lexical drift

---

## Final conceptual claim

Semantic change in early modern pamphlet discourse is best understood not as movement of words through a geometric space, but as:

> the evolving structure of relational semantic fields generated by repeated contextual events across time and discourse conditions
