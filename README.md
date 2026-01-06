# pamphlets

To install dependencies:

```bash
bun install
```

To run:

```bash
bun run 
```

This project was created using `bun init` in bun v1.2.22. [Bun](https://bun.com) is a fast all-in-one JavaScript runtime.

## Questions The System Might Help Answer

* Did The Ranters exist?
* Natural clusters
* Semantic drift in keywords
* Semantic drift/evolution of least frequent terms whose use grows over the period

### Semantic drift in keywords

Semantic drift analysis using temporally-aligned dynamic embeddings and cosine-distance trajectories to quantify shifts in lexical semantics across seventeenth-century pamphlet discourse.

Dynamic embeddings are required because static models conflate historically distinct usages into a single semantic representation, whereas this aspect of the project aims to trace how religious/political concepts are reconfigured through time within pamphlet discourse.

To compare embeddings across temporal slices, each earlier slice is aligned to the subsequent slice using Orthogonal Procrustes, which rotates the vector space to minimise orientation differences while preserving relative distances, ensuring that measured cosine-distance trajectories reflect genuine semantic drift rather than random variation.

## Bibliography

* Raymond, Joab

* [Manuscript Pamphleteering in Early Stuart England](https://tei-c.org/activities/projects/manuscript-pamphleteering-in-early-stuart-england/)

* [Early Modern Manuscripts Online (EMMO)](https://folgerpedia.folger.edu/Early_Modern_Manuscripts_Online_%28EMMO%29?utm_source=chatgpt.com)

* [Early English Books Online Text Creation Partnership (EEBO TCP), Bodleian Digital Library Systems & Services](https://digital.humanities.ox.ac.uk/project/early-english-books-online-text-creation-partnership-eebo-tcp)

* [Bodleian Repo](https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/A50955)
