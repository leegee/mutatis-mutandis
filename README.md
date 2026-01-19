# _Mutatis Mutandis_

# Abstract: About This Project

Computational modelling, building on the work of Firth and Ryan Heuser, is consciously used to complement rather than replaces close reading, rendering visible the semantic drift of Williams’ *keywords* through diachronic visualisations, whilst close reading interprets their rhetorical and ideological significance, both in historical and present perspective, informed by Sinfield’s concept of ideological Faultlines.

Following the conceptual frameworks of Skinner and Koselleck, these shifts mark historically-situated changes in the conceptual limits of vernacular political discourse, and show how salient pivotal terms could be meaningfully expressed and comprehended while retaining moral force despite shedding explicit theological grounding. The medium for this discussion is generically termed pamphlets: as illustrated by Joad Raymond, they functioned as the period's most responsive medium in which concepts are tested and rearticulate as ideas evolve prior to any possibility of stablisation and the possibility of incorporation into law or political doctrine.

Secularisation is treated as a gradual, uneven linguistic process, unfolding within what Joad Raymond describes as the informal, responsive pamphlet culture of early modern England, prior to its formal consolidation into law.

This dual approach demonstrates the possibilities of combining computational and interpretive methods to reveal ideological transformation discoverable through semantic drift, situating the study within a rich and varied tradition of scholarship attentive to semantic, conceptual, and political change.

The software pipeline — including orthological maps, RDBMS, RAG implementation, fastText and MacBERTh models, modelling, visualisation, web-based search interface, and Docker deployment — will be delivered alongside a case study of as-yet-to-be-determined keywords, such as *liberty*, *justice*, *conscience*, and *moral*.

...

**Hart vs Finnis**

Hart and Finnis offer competing accounts of legal authority, but both presuppose a historically inherited moral vocabulary whose formation, contestation, and secularisation their theories do not themselves attempt to explain. It is to the formation of such terms, the process of their evolution, to which this thesis addresses itself.
...

Keyword choice: (Hart argued against) Austin (a student of Benthem): legal authority comes back to what he calls 'the sovereign'.
...

There is a prior historical formation of moral and legal concepts that both Hart and Finnis take for granted without seeking to explain. It is this formation, and the processes whereby it becomes linguistically and rhetorically stabilised, that this thesis addresses.
...
Sinfield, Cultural Materialism pp.32-33

> But in effect I have been addressing the production of ideology. Societies need to produce materially to continue — they need food, shelter, warmth; goods to exchange with other societies; a transport and information infrastructure to carry those processes. Also, they have to produce ideologically (Althusser makes this argument at the start of his essay on ideological state apparatuses). They need knowledges to keep material production going — diverse technical skills and wisdoms in agriculture, industry, science, medicine, economics, law, geography, languages, politics, and so on. And they need understandings, intuitive and explicit, of a system of social relationships within which the whole process can take place more or less evenly. Ideology produces, makes plausible, concepts and systems to explain who we are, who the others are, how the world works. The strength of ideology derives from the way it gets to be common sense; it “goes without saying.” For its production is not an external process, stories are not outside ourselves, something we just hear or read about. Ideology makes sense for us — of us — because it is already proceeding when we arrive in the world, and *we come to consciousness in _its_ terms* [my emphasis]. As the world shapes itself around and through us, certain interpretations of experience strike us as plausible: they fit with what we have experienced already, and are confirmed by others around us. So we complete what Colin Sumner calls a “circle of social reality”: “understanding produces its own social reality at the same time as social reality produces its own understanding.” This is apparent when we observe how people in other cultures than our own make good sense of the world in ways that seem strange to us: their outlook is supported by their social context. For them, those frameworks of perception, maps of meaning, work. The conditions of plausibility are therefore crucial. They govern our understandings of the world and how to live in it


Both Hart and Finnis rely upon a historically inherited moral–legal vocabulary whose emergence, consolidation, and limits they presuppose rather than explain. This thesis intervenes at that prior level, not to resolve their disagreement, but to examine the linguistic and rhetorical formation of the concepts on which it depends. In doing so, it aims to clarify — and where necessary recover — the terms with which such formations can be adequately described.

...

**Expanded:**

## Abstract: About This Project (AHRC Version)

This project traces semantic drift in Williams’ *keywords* bycombining Skinnerian close reading with computational modelling after the work of Ryan Heuser . Diachronic visualisations make patterns of semantic and orthographic change visible, while close reading interprets rhetorical and ideological significance, both from contemporary historical and present perspective, informed by Alan Sinfield’s concept of ideological *Faultlines*.

Following the conceptual frameworks of Skinner and Koselleck, these shifts reveal historically-situated changes in the conceptual limits of vernacular political discourse, and show how salient pivotal terms could be meaningfully expressed and comprehended whilst retaining moral force despite shedding explicit theological grounding. ***

Secularisation is treated as a gradual, uneven linguistic process, unfolding within what Joad Raymond describes as the informal, responsive pamphlet culture of early modern England, prior to its formal consolidation into law.

Read in long perspective, these shifts illuminate moments when inherited moral vocabularies are repurposed under conditions of institutional strain — a pattern with contemporary resonance in a period when, as Lord Dyson has recently observed, the British constitution faces “turmoil”, symptomatic of wider global forces, in which political actors increasingly press against constitutional boundaries, placing institutional norms under strain.

This dual approach demonstrates the possibilities of combining computational and interpretive methods to reveal ideological transformation, situating the study within a diverse tradition of scholarship attentive to semantic, conceptual, and political change.

The project will deliver a software pipeline — including orthological maps, RDBMS, blank RDBS schema, FAISS-based RAG implementation with modern web PWA SPA GUI, trained fastText and fine-tuned MacBERTh models, diachronic visualisation, and a web-based search interface with which to query and visualise the raw and processed data, and a Docker install file — alongside a case study of selected keywords, such as *liberty*, *justice*, *conscience*, and *moral*, and a system that defaults to idempotently reproducing this paper's results, thus including a DB dump populated with raw and tokenised EEBO-TCP and similar pamphlets along with all the orthological maps, dictionaries, mappings, embeddings and Xs the sytem generates.

...a period when, as Lord Dyson in Counsel Magazine[1] says of today, the British constitution faces “turmoil as a symptom of wider global forces” when political actors stretch constitutional boundaries such that our values and institutions are in danger

[1] [Counsel Magaine](https://www.counselmagazine.co.uk/articles/a-conversation-with-lord-dyson?utm_source=chatgpt.com) - see the [Bibliography](./Bibliography.md).

## A Domain-specific Modification of the Heuser Mechanism to run con-currently and primarily

Heuser’s vector-field approach necessarily smooths semantic space by allowing concepts to emerge from aggregated distributional proximity. While highly productive for large-scale pattern detection, this smoothing does risk collapsing historically salient conceptual distinctions - particularly in polemical corpora where antagonistic terms naturally share contexts. The present study therefore inverts this procedure: canonical concepts are defined in advance and held stable, and distributional modelling is used not to discover semantic identity but to trace shifts in the discursive fields surrounding fixed conceptual anchors.

## Keywords and Fault Lines: Computationally Mapping Ideological and Semantic Change in the Distributed Secularisation of Early Modern England

This project treats early modern English pamphlets as sites of ideological fault lines in Alan Sinfield's sense: points where dominant systems of meaning are placed under historical pressure and must negotiate internal contradiction rather than resolve it.

Following the conceptual frameworks of Skinner and Koselleck, these shifts reveal historically-situated changes in the conceptual limits of vernacular political discourse, and show how salient pivotal terms could be meaningfully expressed and comprehended whilst retaining moral force despite shedding explicit theological grounding.

Read in long perspective, such transformations help contextualise contemporary constitutional debate: as Lord Dyson has recently noted, the British constitution is undergoing a period of “turmoil”, symptomatic of wider global forces, in which political actors increasingly test the limits of constitutional convention, placing long-standing institutional norms under pressure.

Building on the computational work of Ryan Heuser, this study uses diachronic distributional semantics to make visible semantic revolutions across the early modern period, connecting lexical neighbourhood shifts to broader social and political transformations. In conjunction with the work of Quentin Skinner, it treats language as both performative and historically situated: semantic drift reflects both the limits and possibilities of argument within specific speech communities. Drawing on Reinhart Koselleck, the project interprets these shifts as evidence of changing conceptual landscapes where key terms such as law, liberty and conscience evolve under pressures, prefiguring modern notions of secular authority and juridical rationality. In the spirit of Raymond Williams' Keywords, these terms are approached as historically contingent sites of debate and negotiation, whose shifting semantic load signals broader cultural and political transformation.

Computational modelling does not intend to replace close reading or ideological critique; rather it renders patterns of semantic drift empirically visible, both literally and figurativly highlighting points where concepts retain moral force whilst dropping explicit theological anchoring, the locations in history where modern English secular law was born. The result is an account of secularisation as a distributed, gradual linguistic process, unfolding unevenly across vernacular political argument before formal consolidation in law. This approach demonstrates the explanatory power of digital methods while explicitly situating the analysis within the historiographical and conceptual lineage established by Sinfield, Raymond, Williams, Heuser, Skinner, and Koselleck.

### Semantic drift in "keywords"

Illustrating semantic drift in keywords (Williams, 1976) through the faultlines (Sinfield, 1992) between:

* theological and juridical language
* clerical authority and lay readership
* Latin legal-theological tradition and vernacular political argument
* sermon, polemic, statute, and petition

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


| Term       | Sinfieldian Contradiction                                                         | Early Stabiliser                          | Pressure Point                                                    | Likely Semantic Drift                                               | Fault Line / Observational Significance                                                                                         |
| ------------ | ----------------------------------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------- | --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Law        | Must appear transcendent and moral, yet function as historically contingent power | Divine ordinance, natural law, providence | Religious fragmentation, civil war, competing claims of authority | God / commandment / scripture to custom / reason / liberty / nation | Moral authority preserved while shedding theological grounding; semantic neighbourhood shifts reveal functional re-legitimation |
| Authority  | Absolute enough to govern, but accountable enough to be argued for                | Divine right, ordination                  | Parliamentarianism, resistance theory, consent                    | Divine / sacred / ordained to civil / parliamentary / delegated     | Authority justified procedurally rather than ontologically; neighbourhood drift exposes ideological negotiation                 |
| Liberty    | Morally defensible but not anarchic; collective yet increasingly individual       | Christian freedom (freedom from sin)      | Conscience, toleration, property                                  | Grace / obedience / soul to right / property / subject / English    | Moral force persists without theological frame; drift indicates secularisation without disenchantment                           |
| Conscience | Binding yet private; authoritative yet resistant to institutional capture         | Sin, salvation, divine judgement          | Sectarianism, toleration debates                                  | God / soul / damnation to judgement / liberty / inward / persuasion | Moral interiority detaches from ecclesiastical authority; fault line visible in subjective reasoning                            |
| Right      | Natural yet historically asserted and contested                                   | Divine or customary sanction              | Petitions, grievances, property claims                            | God / law / nature to subject / liberty / property                  | Naturalisation of historically produced claims; semantic drift shows shifting legitimatory frameworks                           |

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

Initial Target Keywords

* liberty
* justice
* reasonable
* common
* conscience
* god
* king
* divine
* sovereign
* paternal
* state
* nation
* obligation
* authority
* duty
* right
* rule

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
## CPU-Bound

For now the methodology is focuosed on my ancient CPU-only (Radeon...), 64 GB setup.

