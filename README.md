# `pamphlets`

# Notes

## Questions The System Might Help Answer

* Did The Ranters exist?
* Natural clusters
* Semantic drift in keywords
* Semantic drift/evolution of least frequent terms whose use grows over the period

### Semantic drift in keywords

Whilst Austin/Skinner allow an understanding of how the text functioned at its time of writing in its native environment, Koselleck helps to understand how the same concepts change their function over time, and over social and ideological conditions.

* Diachronic tracking of concepts
* Attention to semantic layering
* Focus on conflict rather than consensus
* Analysis of conceptual pairs/poles and asymmetries

Initial corpus: TEI-encoded EEBO-TCP Phase I - including rich metadata

* Date (can be mechanically extracted)
* Genre (can be mechanically inferred)
* Printer / place (possibly mechanically extracted)
* Political alignment (where known/inferred)

Semantic drift analysis using temporally-aligned dynamic embeddings of conceptual tokens (eg liberty) and cosine-distance trajectories to quantify shifts in lexical semantics across seventeenth-century pamphlet discourse.

Dynamic embeddings are required because static models conflate historically distinct usages into a single semantic representation, whereas this aspect of the project aims to trace how religious/political concepts are reconfigured through time within pamphlet discourse.

To compare embeddings across temporal slices, each earlier slice is aligned to the subsequent slice using Orthogonal Procrustes, ie rotating the vector space to minimise orientation differences whilst preserving relative distances, ensuring that measured cosine-distance trajectories reflect genuine semantic drift rather than random variation.

Polemical clustering by concept usage an rhetorical posture (genre) to show who uses the same word (where) differently at the same time (and location) to capture asymmetrical counter-concepts and reveal conceptual struggle.

Where drift is seen to accelerate, apply a Skinnerian close reading to identify illocutionary force.

## Visualisation

2D graph where x = time slicse, y = semantic proximity to target word, cells = words.

## Bibliography

* Austin, JL: "How To Do Things With Words"

* Harris, Zellig: [Methods in Structural Linguistics] (1951) (meaning inferred from distribution)

* Firth, John R: [Synopsis of Linguistic Theory 1930–55. In: F. R. Palmer (ed.) Studies in Linguistic Analysis. Oxford: Blackwell, pp 1–32]() ("You shall know a word by the company it keeps.", 1957)

* Raymond, Joab:

* Skinner, Quentin:

* Williams, Raymond: [Keywords: A Volcabulary of Culture And Society]() (1976)

* Williams, Raymond: [Culture And Society 1780-1950]() (1958)

* Koselleck, Reinhart: [Futures Past]()

* Koselleck, Reinhart: [Geschichtliche Grundbegriffe]() (1972, 1997)

* Koselleck, Reinhart: [Social History and Conceptual History]()

* Heuser, Ryan: [Computing Kosselleck: Modelling Semantic Revolutions 1720-1960 (talk)](https://www.youtube.com/watch?v=7L-PO-AqG60)

* Heuser, Ryan: [Computing Koselleck: Modelling Semantic Revolutions, 1720–1960](https://www.cambridge.org/core/books/abs/explorations-in-the-digital-history-of-ideas/computing-koselleck-modelling-semantic-revolutions-17201960/1ED34828C706CB5A7882E0A825C6F72F) from Part II - Case Studies in the Digital History of Ideas, Published online by Cambridge University Press:  09 November 2023

## People and Projects

* [Manuscript Pamphleteering in Early Stuart England](https://tei-c.org/activities/projects/manuscript-pamphleteering-in-early-stuart-england/)

* [Heuser, Ryan](https://www.english.cam.ac.uk/people/Ryan.Heuser)

* [MacBERTHh](https://huggingface.co/emanjavacas/MacBERTh)

* [Bodleian Repo](https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/A50955)

* [Early Modern Manuscripts Online (EMMO)](https://folgerpedia.folger.edu/Early_Modern_Manuscripts_Online_%28EMMO%29?utm_source=chatgpt.com)

* [Early English Books Online Text Creation Partnership (EEBO TCP), Bodleian Digital Library Systems & Services](https://digital.humanities.ox.ac.uk/project/early-english-books-online-text-creation-partnership-eebo-tcp)

