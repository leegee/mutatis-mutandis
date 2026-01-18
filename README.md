## Abstract

Computational modelling, building on the work of Ryan Heuser, complements rather than replaces close reading, rendering the semantic drift of Williams’ keywords visible through diachronic visualisation, while close reading interprets their rhetorical and ideological significance, informed by Sinfield’s concept of ideological fault lines. Following the conceptual frameworks of Skinner and Koselleck, these shifts mark historically situated changes in the conceptual limits of vernacular political discourse, and show how salient pivotal terms could be meaningfully expressed and comprehended while retaining moral force despite shedding explicit theological grounding. Secularisation is treated as a gradual, uneven linguistic process, unfolding within what Joad Raymond describes as the informal, responsive pamphlet culture of early modern England, prior to its formal consolidation in law. This dual approach demonstrates the possibilities of combining computational and interpretive methods to reveal ideological transformation, following a rich and varied tradition of scholarship attentive to semantic, conceptual, and political change.

## Keywords and Fault Lines: Computationally Mapping Ideological and Semantic Change in the Distributed Secularisation of Early Modern England

This project treats early modern English pamphlets as sites of ideological fault lines in Alan Sinfield's sense: points where dominant systems of meaning are placed under historical pressure and must negotiate internal contradiction rather than resolve it.

Following Joad Raymond, pamphlets are treated not as transparent expressions of political doctrine but as an area in which theological, moral, and juridical vocabularies are actively reworked in response to crisis, operating outside the constraints of statute, sermon and institutional doctrine. The significance of pamphlets lies less in the positions they advocate than in the semantic work they perform: justifying authority, law, liberty, and right under conditions of religious fragmentation and constitutional uncertainty.

Building on the computational work of Ryan Heuser, this study uses diachronic distributional semantics to make visible semantic revolutions across the early modern period, connecting lexical neighbourhood shifts to broader social and political transformations. In conjunction with the work of Quentin Skinner, it treats language as both performative and historically situated: semantic drift reflects both the limits and possibilities of argument within specific speech communities. Drawing on Reinhart Koselleck, the project interprets these shifts as evidence of changing conceptual landscapes where key terms such as law, liberty and conscience evolve under pressures, prefiguring modern notions of secular authority and juridical rationality. In the spirit of Raymond Williams' Keywords, these terms are approached as historically contingent sites of debate and negotiation, whose shifting semantic load signals broader cultural and political transformation.

Computational modelling does not intend to replace close reading or ideological critique; rather it renders patterns of semantic drift empirically visible, both literally and figurativly highlighting points where concepts retain moral force whilst dropping explicit theological anchoring, the locations in history where modern English secular law was born. The result is an account of secularisation as a distributed, gradual linguistic process, unfolding unevenly across vernacular political argument before formal consolidation in law. This approach demonstrates the explanatory power of digital methods while explicitly situating the analysis within the historiographical and conceptual lineage established by Sinfield, Raymond, Williams, Heuser, Skinner, and Koselleck.

### Semantic drift in "keywords"

Illustrating semantic drift in keywords (Williams, 1976) through the faultlines (Sinfield, 1992) between:

* theological and juridical language

* clerical authority and lay readership

*  Latin legal-theological tradition and vernacular political argument

*  sermon, polemic, statute, and petition

Whilst Austin/Skinner will allow an understanding of how the text functioned at its time of writing in its native environment,
Koselleck will help to understand how the same concepts change function over both time and over social and ideological conditions.

* Diachronic tracking of concepts
* Attention to semantic layering
* Focus on conflict rather than consensus
* Analysis of conceptual pairs/poles and asymmetries

Revealing diachronic neighbourhoods should reveal the underlying semantic change.

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

| Term | Sinfieldian Contradiction | Early Stabiliser | Pressure Point | Likely Semantic Drift | Fault Line / Observational Significance |
|------|---------------------------|------------------|----------------|-----------------------|-------------------------------------------|
| Law | Must appear transcendent and moral, yet function as historically contingent power | Divine ordinance, natural law, providence | Religious fragmentation, civil war, competing claims of authority | God / commandment / scripture to custom / reason / liberty / nation | Moral authority preserved while shedding theological grounding; semantic neighbourhood shifts reveal functional re-legitimation |
| Authority | Absolute enough to govern, but accountable enough to be argued for | Divine right, ordination | Parliamentarianism, resistance theory, consent | Divine / sacred / ordained to civil / parliamentary / delegated | Authority justified procedurally rather than ontologically; neighbourhood drift exposes ideological negotiation |
| Liberty | Morally defensible but not anarchic; collective yet increasingly individual | Christian freedom (freedom from sin) | Conscience, toleration, property | Grace / obedience / soul to right / property / subject / English | Moral force persists without theological frame; drift indicates secularisation without disenchantment |
| Conscience | Binding yet private; authoritative yet resistant to institutional capture | Sin, salvation, divine judgement | Sectarianism, toleration debates | God / soul / damnation to judgement / liberty / inward / persuasion | Moral interiority detaches from ecclesiastical authority; fault line visible in subjective reasoning |
| Right | Natural yet historically asserted and contested | Divine or customary sanction | Petitions, grievances, property claims | God / law / nature to subject / liberty / property | Naturalisation of historically produced claims; semantic drift shows shifting legitimatory frameworks |

## Notes

Early modern pamphlets repeatedly convert moral vocabulary into instruments of behavioural discipline,
translating theological anxiety into social coercion
at precisely the moment when customary forms of authority were under strain.

## Questions The System Might Help Answer

* Drift from theocracy to democracy
* Did The Ranters exist?
* Natural diachronic clusters: drift in keywords, IVF
* Geospatial spread of ideas and ideology
* Author, printer networks

## Visualisation

2D graph where x = time slicse, y = semantic proximity to target word, cells = words.

### Keyword trajectory over time

The rise and fall of vocab

Relative freq per slice

### Co-occurrence networks

Conceptual clusters = ideological associations

nodes = words,  edges = co-occurrence strength

### Semantic shift, embedding maps

Meaning, usage of a word changing over time

UMAP or t-SNE, plot words as points, colour-coded by slice

### Faultline heatmaps

Areas of ideological tension (eg secular vs religious language across corpus

rows = pamphlets or slices, columns = semantic clusters, values = normalized frequency or association strength

### Phrase, collocation clouds (dynamic word clouds)

Dominant collocations and shifting emphasis in discourse.

Collocations per slice, overlaid in time-lapse/interactive visualization (clouds or layered bar charts)

### Topic Modeling Evolution

Rise and fall of ideological themes

Streamgraphs

### Narrative flow

Plot cosine similarity in embeddings?

Visualise as a line graph or heatmap to show narrative tension?

### Ideological Network Across Authors

Nodes = authors, edges = textual similarity between their pamphlets.

Visualise clusters or “communities” to show who is ideologically aligned.

### Geographical Mapping

#### Explicit

    Metadata -> map

Remember that radical presses were moved.

#### Ideological

Corpus -> Embeddings -> cosine similiarity to poles, keyword scores, PCA/UMAP reduction, supervised learning -> map via metadata

If sufficient sources can be mapped, does their orthography reveal their writers'/printers' locale?

I wonder if we have sufficient sources of known ideological stance to use as training data?

Most keywords will be contested, ambigious.

## Bibliography

See [Bibliography](./BIBLIOGRAPHY.md)

## People and Projects

* [Manuscript Pamphleteering in Early Stuart England](https://tei-c.org/activities/projects/manuscript-pamphleteering-in-early-stuart-england/)

* [Heuser, Ryan](https://www.english.cam.ac.uk/people/Ryan.Heuser)

* [MacBERTHh](https://huggingface.co/emanjavacas/MacBERTh)

* [Bodleian Repo](https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/A50955)

* [Early Modern Manuscripts Online (EMMO)](https://folgerpedia.folger.edu/Early_Modern_Manuscripts_Online_%28EMMO%29?utm_source=chatgpt.com)

* [Early English Books Online Text Creation Partnership (EEBO TCP), Bodleian Digital Library Systems & Services](https://digital.humanities.ox.ac.uk/project/early-english-books-online-text-creation-partnership-eebo-tcp)

## Restoring the Database

Make sure the table space is on an SSD:

    CREATE TABLESPACE eebo_space LOCATION 'D:/postgres-data-2/eebo';

Create a temp tablespace if not already and use it for sorting/indexing:

    CREATE TABLESPACE temp_space LOCATION 'D:/postgres-data-2/temp';
    ALTER SYSTEM SET temp_tablespaces = 'temp_space';

Increase memory for faster index creation

    ALTER SYSTEM SET maintenance_work_mem = '16GB';  -- big enough for token indexes
    ALTER SYSTEM SET work_mem = '256MB';             -- per sort operation
    SELECT pg_reload_conf();

Kill all connections:

    SELECT pg_terminate_backend(pid)
    FROM pg_stat_activity
    WHERE datname='eebo';

Restore with 4 worker jobs:

    pg_restore -v -d eebo -j 4 "./db-backup/eebo_backup.dump"

Monitor:

    -- Active queries (shows index creation)
    SELECT pid, now() - query_start AS duration, state, query
    FROM pg_stat_activity
    WHERE state <> 'idle';

    -- Size of largest tables and indexes
    SELECT relname, pg_size_pretty(pg_total_relation_size(relid))
    FROM pg_stat_user_tables
    ORDER BY pg_total_relation_size(relid) DESC;

Clean up:

    ALTER SYSTEM RESET maintenance_work_mem;
    ALTER SYSTEM RESET work_mem;
    ALTER SYSTEM RESET temp_tablespaces;
    SELECT pg_reload_conf();





# CPU-Bound

For now the methodology is focuosed on my ancient CPU-only (Radeon...), 64 GB setup.
