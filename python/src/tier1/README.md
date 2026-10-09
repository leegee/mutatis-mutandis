# Colab Passage Embedding Workflow

This workflow generates passage-level MacBERTh embeddings for historical pamphlet texts and imports them into Lance. It is separate from the existing Tier 1 event-embedding pipeline.

## Workflow

```text
PostgreSQL
    |
    v
colab_export_for.py
    |  Document-level token arrays in Parquet
    v
Google Drive: tier1/tokens/
    |
    v
colab_embed_passages.py
    |  MacBERTh passage embeddings
    v
Google Drive: tier1/passages/
    |
    v
Local: ../out/export/passages/
    |
    v
passages_parquet_to_lance.py
    |
    v
Lance: out/indexes/lance_pamphlets/
```

## 1. Export tokens locally

**File:** `colab_export_for.py`

Reads tokenised documents from PostgreSQL and exports them to Parquet shards for processing in Colab.

Each row represents one document and contains:

* `corpus`, `doc_id`, `pub_year`
* `token_idx`: original token indices
* `tokens`: document tokens in order

The exporter groups documents into shards, normally 500 documents per file.

Run from the Python project directory:

```bash
python colab_export_for.py
```

Upload the resulting token shards to `MyDrive/tier1/tokens/` in Google Drive.

## 2. Generate passage embeddings in Colab

**File:** `colab_embed_passages.py`

Runs `emanjavacas/MacBERTh` in Google Colab, preferably with a GPU. It reads the token Parquet shards and generates overlapping passages.

Current settings:

| Setting               |                                           Value |
| --------------------- | ----------------------------------------------: |
| Passage window        |                                        40 words |
| Passage hop           |                                        20 words |
| Embedding dimension   |                                             768 |
| Representations       | Layer 8, final layer, mean of final four layers |
| Output vector storage |                                         float16 |

Each output row represents a passage and preserves its document provenance and word/token boundaries:

* `corpus`, `doc_id`, `pub_year`
* `word_start`, `word_end`
* `token_start`, `token_end`
* `vec_l8`, `vec_last`, `vec_mean4`

Output files are written to `MyDrive/tier1/passages/`, using names such as `passages_00000.parquet`.

After processing, copy or download the completed passage shards to the local import directory:

```text
../out/export/passages/
```

Only completed output files should be imported. Check that each expected input shard has a corresponding passage output.

## 3. Import passages into Lance

**File:** `tier1/passages_parquet_to_lance.py`

Run locally from the Python project directory:

```bash
python -m tier1.passages_parquet_to_lance
```

The importer reads `../out/export/passages/passages_*.parquet` and writes three separate families of year-bucketed Lance tables:

* `l8`: layer 8 embeddings
* `last`: final-layer embeddings
* `mean4`: mean of the final four layers

Each table uses a single `vector` column, accompanied by passage identity, document provenance, word/token boundaries, publication year and model metadata.

Configuration is shared with the existing project:

* Lance root: `LANCE_INDEXES_DIR`
* Year bucket size: `LANCE_BUCKET_SIZE` (50 years)
* Model label: `LANCE_MODEL_NAME` (`macberth`)

The importer creates deterministic passage IDs and skips IDs already present, allowing interrupted imports to be resumed. Vector and scalar indexes are built after ingestion.

## Important distinctions

* Parquet is an intermediate transfer format; Lance is the local search store.
* Passage vectors are distinct from event vectors. Do not allocate artificial event IDs or insert passages into the PostgreSQL `events` table.
* `ACTIVE_SCALES` controls event-vector scales, not the three passage representations.
* Do not run the event writer's `purge_orphans()` routine against passage tables.
* Avoid simultaneous imports into the same Lance tables.

After importing, verify table row counts and vector dimensions before using the passage embeddings for search.

## Passage Embeddings: Purpose and Next Steps

### Purpose

The passage-embedding pipeline complements the existing event-vector pipeline by enabling semantic retrieval of historical text passages. It provides a discovery layer for finding relevant evidence even when historical authors express a concept using vocabulary different from the researcher's query.

Passage embeddings are generated using MacBERTh, with three alternative representations:

* **`l8`** — hidden states from layer 8.
* **`last`** — final hidden-layer representation.
* **`mean4`** — mean of the final four hidden layers.

Each representation is stored separately in Lance tables, with 50-year publication-date buckets and passage provenance, including corpus, document ID, publication year, word boundaries and token boundaries. Passage identity is stable across representations.

### Intended use

For example, a researcher investigating **LIBERTY, 1600–1650** could retrieve passages discussing freedom, rights, property, privilege or immunity without requiring every passage to contain the word *liberty*.

The intended workflow is:

1. Encode a query using a compatible MacBERTh representation.
2. Retrieve the nearest passage vectors using cosine similarity, filtered by publication date.
3. Display the original passage, document metadata and source location.
4. Inspect the evidence and identify relevant vocabulary, arguments and historical contexts.
5. Compare results across periods and connect interesting passages to the existing event-search pipeline.

Passage search and event search serve complementary purposes. Passage vectors locate relevant contexts and arguments; event vectors connect individual occurrences and support investigation of relationships between words and concepts. Together, they support provenance-preserving diachronic semantic search.

### Development priorities

1. **Validate the index:** confirm passage counts, document and year coverage, vector dimensions, and the absence of non-finite or zero vectors.
2. **Implement basic retrieval:** accept a query vector, representation and year range; return ranked passages with similarity scores and provenance.
3. **Evaluate the representations:** compare `l8`, `last` and `mean4` against a small benchmark of historically informed queries, including LIBERTY, LEGITIMACY and TYRANNY.
4. **Support multiple seeds:** retrieve results independently for several query terms or example passages and combine rankings using reciprocal rank fusion (RRF), rather than relying solely on a single averaged query vector.
5. **Integrate event search:** allow researchers to move from a relevant passage to its constituent words and occurrences, then explore related events.
6. **Handle overlapping windows:** the 40-word passages overlap by 20 words, so group or deduplicate overlapping results where appropriate while preserving their source provenance.

### Design principles

* Keep the three representations separate until evaluation establishes their relative usefulness.
* Treat retrieval quality as an empirical question: MacBERTh is a contextual language model, not a model specifically trained for sentence-level semantic retrieval.
* Preserve source text, document identity, dates and token-level provenance throughout the search process.
* Present embeddings as tools for discovering historical evidence, not as substitutes for historical interpretation.

**Immediate objective:** build a minimal end-to-end demonstration that accepts a research query and date range, returns the top 20 passages, and allows each result to be inspected in its original documentary context. Evaluate its ability to discover useful evidence beyond the existing event-search pipeline before developing a more sophisticated interface.

## Sample Queries

### Justifying violence against opponents

Research concept: how writers justify, excuse or condemn violence against people regarded as enemies.

Query:

> When is it lawful or necessary to use force against those who threaten the religion, peace or safety of the kingdom?

What it might uncover: passages using resistance, self-preservation, defence, rebellion, necessity, malignants or enemies of the kingdom.

Why it is useful: the vocabulary may distinguish a defence of violence from a condemnation of it. The system must retrieve the context, not simply match a reference to violence.

### Identifying an internal enemy

Research concept: how writers portray a group as secretly threatening a community or its institutions.

Query

> People accused of secretly undermining the established religion, government or common good.

What it might uncover: plots, conspiracies, machinations, sedition, corruption, designs and enemies within.

Why it is useful: the same underlying accusation may be expressed without any one stable term for conspiracy or subversion.

### Dehumanising or demonising a group

Research concept: rhetoric that represents opponents as inherently wicked, dangerous or unworthy of ordinary treatment.

Query

> A group of people portrayed as wicked, dangerous, corrupt or enemies of humanity who deserve punishment.

What it might uncover: expressions of moral pollution, disease, monstrosity, satanic influence or collective guilt.

Why it is useful: it tests whether semantic retrieval connects different rhetorical strategies without treating every negative description as equivalent.

### Allegations of religious disloyalty

Research concept: accusations that a religious group is disloyal to the state or secretly serves another power.

Query

> People accused of placing loyalty to a foreign power or religious authority above their duty to their own country.

What it might uncover: claims about divided allegiance, foreign influence, treachery, popery, corruption of the state or subjection to an external authority.

Why it is useful: it tests whether the search retrieves the underlying accusation when the specific target group or historical terminology is absent from the query.

### The language of political legitimacy

Research concept: when obedience to authority is presented as a duty, and when resistance becomes justified.

Query

> en a ruler abuses lawful authority, whether the people may withdraw obedience or resist the ruler.

What it might uncover: prerogative, tyranny, trust, consent, liberty, lawful resistance and arguments about the people's rights.

Why it is useful: this extends the searches you have already run for royal power and liberty, but gives you a more explicit conceptual test.
