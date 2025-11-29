MacBERTh (1450–1950)

https://huggingface.co/emanjavacas/MacBERTh
https://jdmdh.episciences.org/9690/pdf

Out-of-the-box capabilities:
MacBERTh is the strongest general-purpose model for EME and LME. Unlike vanilla BERT, BERT-Adapted, or TuringBERT, it is pretrained from scratch on a large historical corpus, internalising:

* Historical spelling variation and obsolete morphology
* Period-specific syntax
* Genre conventions of pamphlets, sermons, newsbooks, and petitions
* Deep bidirectional context
* Stable phrase- and sentence-level semantics
* Direct tasks supported without additional training:
* OCR correction
* Tokenisation, lemmatisation, and POS tagging
* Semantic drift analysis (phrase reuse detection, cluster mapping)
* RAG-style analysis: clustering by ideology, author, phrase, or semantic content
* TEI-adjacent tagging
* Argument-structure extraction (better clause/phrase boundaries allow automated extraction of premises, claims, authority citations, scriptural references)
* Geo-spatial semantics
* Extraction of semantic neighbours from embeddings

Tasks requiring fine-tuning / pre-training on the specific corpus:

* Sentiment classification in historically grounded categories, eg:
  * Godly vs ungodly, liberty vs bondage
* Political alignment classification:
  * Royalist vs Parliamentarian vs Radical sectary
* Genre classification:
  * Sermon vs polemic vs petition vs newsbook
* Specific forms: 
  * answer, remonstrance, tract, diurnall
* Nearest-neighbour search at the sentence level to detect borrowed tropes or recurring rhetorical formulas, eg:
  * "Ancient rights and liberties of the people"


  1. Sentiment classification (historically grounded)

Labels: godly / ungodly, liberty / bondage, positive / negative / neutral in period terms

Data needed:

Small-scale: ~2,000–5,000 labelled paragraphs

Medium-scale: ~10,000+ paragraphs for better generalisation

Labelling strategy:

Use human annotation for seed set (≈500–1,000 examples)

Bootstrapped labelling with keyword lists (e.g., “tyranny,” “popery,” “freedom”) for expansion

Ensure balance across labels

Output: fine-tuned MacBERTh classifier that maps historical phrasing to sentiment categories

2. Political alignment classification

Labels: Royalist, Parliamentarian, Radical sectary

Data needed:

Seed: 500–1,000 pamphlets clearly associated with each faction

Full fine-tuning: 2,000–5,000 labelled pamphlets or paragraphs

Labelling strategy:

Use metadata where available (author, publisher, known faction)

Manual annotation of ambiguous texts to improve classifier

Output: model predicts factional alignment for unseen pamphlets

3. Genre classification

Labels: Sermon, polemic, petition, newsbook

Data needed:

Seed: 500–1,000 examples per genre

Fine-tuning: 2,000–3,000 per genre

Labelling strategy:

Use library/catalogue metadata when possible

Annotate ambiguous or hybrid texts manually

Output: model distinguishes genres even when vocabulary overlaps

4. Specific pamphlet forms

Labels: answer, remonstrance, tract, diurnall

Data needed:

Small: 500 examples per form

Medium: 1,000–2,000 per form

Labelling strategy:

Use TEI metadata or publication notices where available

Manual annotation for unclear cases

Output: fine-grained classification of sub-types within genres

5. Sentence-level nearest-neighbour / trope detection

Labels: Not strictly labelled; embeddings are unsupervised

Data needed:

The larger the corpus, the better — even a few thousand sentences works for local analysis

Labelling strategy:

Optional: manually curate a set of “known repeated tropes” to validate clustering

Use embeddings to detect recurring phrases or rhetorical formulas

Output: fine-tuned embeddings that cluster historical tropes effectively, allowing RAG-style retrieval and semantic search


