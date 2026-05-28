# GPT Summary - Tier 2 Diagnostic Finding: FAISS Neighbourhood Collapse vs Graph Construction Failure Modes

## 1. Empirical observation

During year-filtered and concept-restricted runs (e.g. PREROGATIVE, 1645–1645), the following behaviour was observed:

* The embedding audit shows **high but non-degenerate similarity structure**

  * cosine similarity mean ≈ 0.90
  * std ≈ 0.06–0.07
  * p95 ≈ 0.98
* Nearest-neighbour results are highly concentrated:

  * a small set of event IDs dominate KNN outputs across many queries
* These same event IDs recur repeatedly across neighbourhoods (frequency spikes up to ~70+)

At the same time:

* The derived graph construction yields:

  * RAW EDGE PAIRS > 0
  * KEPT EDGES = 0 (after thresholding)
  * DEGREE MAP SIZE = 0
  * EMPTY GRAPH returned

This creates a contradiction at the *interpretation layer*, not the embedding layer.

---

## 2. Core diagnosis

The failure is not caused by FAISS, embeddings, or filtering bugs.

It is caused by a structural mismatch between:

> **what FAISS produces (event neighbourhood geometry)**
> vs
> **what the graph layer assumes (token co-occurrence structure)**

### Key distinction

The system currently assumes:

* FAISS neighbourhoods ≈ token co-occurrence context

But empirically:

* FAISS neighbourhoods ≈ **semantic attractor sets of events**

These are not equivalent objects.

---

## 3. Mechanism of collapse

The current graph construction reduces KNN structure into:

* tokenA ↔ tokenB co-occurrence counts inside neighbour lists

However, KNN lists are:

* dominated by repeated high-similarity event clusters
* often contain repeated or near-identical lexical realisations
* strongly biased toward dense semantic attractors

This produces:

* self-reinforcing token repetition
* weak cross-token diversity
* edge filtering failure when threshold ≥ 1–3

Hence:

> The graph is not “empty” semantically — it is **structurally unprojectable into token-token edges under current assumptions**.

---

## 4. What the diagnostics imply

Two key findings emerge:

### (A) Embedding space is NOT collapsed

* norms are stable
* cosine variance exists
* neighbourhood structure is meaningful

### (B) Graph projection is mismatched

* token co-occurrence projection loses most of the signal
* FAISS neighbourhood structure is not lexical in nature
* graph sparsification removes remaining weak edges

---

# 5. Three viable next directions

## Direction 1 — Event–Event overlap graph (recommended)

Reframe the graph at the level FAISS actually operates on:

### Definition

Let:

* KNN(A) = set of nearest neighbour events for event A

Then:

```text
edge(A, B) = |KNN(A) ∩ KNN(B)|
```

### Properties

* preserves FAISS geometry directly
* avoids token projection loss
* produces a **true contextual similarity manifold**
* naturally supports clustering and drift tracking

### Interpretation

This becomes:

> “How similar are two contextual usages of words?”

rather than:

> “Which words co-occur?”

---

## Direction 2 — Directional lexical attraction graph

Retain tokens, but stop treating KNN lists as co-occurrence sets.

### Definition

For token A:

```text
edge(A → B) = frequency of B in KNN(A)
```

### Properties

* preserves asymmetry (important for semantic drift)
* avoids symmetric co-occurrence collapse
* yields interpretable semantic fields

### Interpretation

This becomes:

> “Which concepts does a word pull toward in embedding space?”

rather than co-occurrence.

---

## Direction 3 — Tier 1-native window co-occurrence graph

Bypass FAISS entirely for graph construction.

### Definition

Use raw Tier 1 structure:

* window_id
* token stream within window

Define:

```text
edge(A, B) = co-occurrence within same window
```

(optionally weighted by distance or frequency)

### Properties

* fully interpretable
* historically grounded (document + year aware)
* stable across model changes
* avoids embedding artefacts entirely

### Interpretation

This becomes:

> “Rhetorical / syntactic co-occurrence in Early Modern textual windows”

---

# 6. Structural conclusion

The pipeline currently conflates three distinct spaces:

| Layer                 | Object type       | Meaning                         |
| --------------------- | ----------------- | ------------------------------- |
| Tier 1                | token-in-window   | textual co-occurrence           |
| Tier 2 (FAISS)        | event embeddings  | contextual semantic geometry    |
| Graph layer (current) | token-token edges | projected lexical co-occurrence |

The failure arises because:

> FAISS produces geometry over events, but the graph assumes geometry over tokens.

These are incompatible projections without an explicit aggregation model.

---

# 7. Key methodological implication

What has been empirically demonstrated is:

> KNN semantic neighbourhoods in contextual embedding space do not reliably project onto lexical co-occurrence graphs.

This is not a bug — it is a structural property of contextual embeddings.
