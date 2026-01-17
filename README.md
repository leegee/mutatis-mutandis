# Keywords and Fault Lines: Computationally Mapping Ideological and Semantic Change in the Distributed Secularisation of Early Modern England

This project treats early modern English pamphlets as sites of ideological fault lines in Alan Sinfield's sense: points where dominant systems of meaning are placed under historical pressure and must negotiate internal contradiction rather than resolve it.

Following Joad Raymond, pamphlets are treated not as transparent expressions of political doctrine but as an area in which theological, moral, and juridical vocabularies are actively reworked in response to crisis, operating outside the constraints of statute, sermon and institutional doctrine. The significance of pamphlets lies less in the positions they advocate than in the semantic work they perform: justifying authority, law, liberty, and right under conditions of religious fragmentation and constitutional uncertainty.

Building on the computational work of Ryan Heuser, this study uses diachronic distributional semantics to make visible semantic revolutions across the early modern period, connecting lexical neighbourhood shifts to broader social and political transformations. In conjunction with the work of Quentin Skinner, it treats language as both performative and historically situated: semantic drift reflects both the limits and possibilities of argument within specific speech communities. Drawing on Reinhart Koselleck, the project interprets these shifts as evidence of changing conceptual landscapes where key terms such as law, liberty and conscience evolve under pressures, prefiguring modern notions of secular authority and juridical rationality. In the spirit of Raymond Williams' Keywords, these terms are approached as historically contingent sites of debate and negotiation, whose shifting semantic load signals broader cultural and political transformation.

Computational modelling does not intend to replace close reading or ideological critique; rather it renders patterns of semantic drift empirically visible, highlighting points where concepts retain moral force while dropping explicit theological anchoring. The result is an account of secularisation as a distributed, gradual linguistic process, unfolding unevenly across vernacular political argument before formal consolidation in law. This approach demonstrates the explanatory power of digital methods while explicitly situating the analysis within the historiographical and conceptual lineage established by Sinfield, Raymond, Williams, Heuser, Skinner, and Koselleck.

## Geographical Mapping

### Explicit

    Metadata -> map

Remember that radical presses were moved.

### Ideological

Corpus -> Embeddings -> cosine similiarity to poles, keyword scores, PCA/UMAP reduction, supervised learning -> map via metadata

If sufficient sources can be mapped, does their orthography reveal their writers'/printers' locale?

I wonder if we have sufficient sources of known ideological stance to use as training data?

Most keywords will be contested, ambigious.

## Notes

Early modern pamphlets repeatedly convert moral vocabulary into instruments of behavioural discipline,
translating theological anxiety into social coercion 
at precisely the moment when customary forms of authority were under strain.

## Questions The System Might Help Answer

* Drift from theocracy to democracy
* Did The Ranters exist?
* Natural clusters
* Semantic drift in keywords
* Semantic drift/evolution of least frequent terms whose use grows over the period

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

## Bibliography
- Austin, J. L. (1962). *How to Do Things with Words: The William James Lectures delivered in Harvard University in 1955*. Oxford: Clarendon Press / Oxford University Press. :contentReference[oaicite:0]{index=0}

- Harris, Zellig (1951/1957). “A Synopsis of Linguistic Theory, 1930–55”. In *Studies in Linguistic Analysis*, edited volume, Oxford: Blackwell, pp. 1–32. :contentReference[oaicite:1]{index=1}

- Firth, J. R. (1957). “A Synopsis of Linguistic Theory, 1930–55”. In F. R. Palmer (ed.), *Studies in Linguistic Analysis*. Oxford: Blackwell, pp. 1–32. :contentReference[oaicite:2]{index=2}

- Firth, J. R. (1957). “You shall know a word by the company it keeps.” In F. R. Palmer (ed.), *Studies in Linguistic Analysis*. Oxford: Blackwell, pp. 1–32. :contentReference[oaicite:3]{index=3}

- Raymond, Joab. *[Details missing — please provide full title, year, and publication/publisher if known]*

- Sinfield, Alan (1992). Faultlines: Cultural Materialism and the Politics of Dissident Reading. Oxford: Clarendon Press (Oxford University Press).

- Skinner, Quentin. *[Details missing — please provide full title, year, and publication/publisher if known]*

- Williams, Raymond (1976). *Keywords: A Vocabulary of Culture and Society*. London: Croom Helm. :contentReference[oaicite:4]{index=4}

- Williams, Raymond (1958). *Culture and Society 1780–1950*. London: Chatto & Windus (first UK ed.); New York: Columbia University Press (US ed.). :contentReference[oaicite:5]{index=5}

- Koselleck, Reinhart. *Futures Past: On the Semantics of Historical Time*. Cambridge, MA: MIT Press. *[Publishing details may vary — please provide year and edition]*

- Koselleck, Reinhart (1972, 1997). *Geschichtliche Grundbegriffe: Historisches Lexikon zur politisch‑sozialen Sprache in Deutschland*. Stuttgart: Klett‑Cotta. *[Often cited with multiple volumes across dates]*

- Koselleck, Reinhart. *Social History and Conceptual History*. Stanford, CA: Stanford University Press. *[Please provide year if known]*

- Heuser, Ryan (2023). “Computing Koselleck: Modelling Semantic Revolutions 1720–1960 (talk)”. YouTube video. https://www.youtube.com/watch?v=7L-PO-AqG60

- Heuser, Ryan (2023). “Computing Koselleck: Modelling Semantic Revolutions, 1720–1960”. In *Explorations in the Digital History of Ideas*, Part II – Case Studies in the Digital History of Ideas, edited by Peter de Bolla. Published online by Cambridge University Press: 09 November 2023. https://www.cambridge.org/core/books/abs/explorations-in-the-digital-history-of-ideas/computing-koselleck-modelling-semantic-revolutions-17201960/1ED34828C706CB5A7882E0A825C6F72F

- Hobbes, Thomas (1651, 1996). Leviathan, or The Matter, Forme, & Power of a Common-Wealth Ecclesiasticall and Civil. London: Andrew Crooke. [Modern critical edition: edited by Richard Tuck, Cambridge: Cambridge University Press, 1996]

- Locke, John (1689, 1980). Two Treatises of Government. London: Awnsham & John Churchill. [Second Treatise often cited separately; modern edition: edited by Peter Laslett, Cambridge: Cambridge University Press, 1980]

- Gramsci, Antonio (1929–1935, 1971, 1992). Selections from the Prison Notebooks. New York: International Publishers. [English edition, translated and edited by Quintin Hoare and Geoffrey Nowell Smith; see especially Part I, “Hegemony and Political Society,” for foundational discussion of cultural and ideological hegemony]

## People and Projects

* [Manuscript Pamphleteering in Early Stuart England](https://tei-c.org/activities/projects/manuscript-pamphleteering-in-early-stuart-england/)

* [Heuser, Ryan](https://www.english.cam.ac.uk/people/Ryan.Heuser)

* [MacBERTHh](https://huggingface.co/emanjavacas/MacBERTh)

* [Bodleian Repo](https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/A50955)

* [Early Modern Manuscripts Online (EMMO)](https://folgerpedia.folger.edu/Early_Modern_Manuscripts_Online_%28EMMO%29?utm_source=chatgpt.com)

* [Early English Books Online Text Creation Partnership (EEBO TCP), Bodleian Digital Library Systems & Services](https://digital.humanities.ox.ac.uk/project/early-english-books-online-text-creation-partnership-eebo-tcp)



