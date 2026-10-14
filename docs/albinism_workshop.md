Workshop abstract

Title: Searching for pale bodies across time: a pilot two-ended semantic retrieval of hair and eye descriptions in historical insular texts

Background. Changes in the cultural meaning of extreme bodily whiteness, such as very white hair, unusually pale skin and pale or fiery eyes, are hard to study at scale. The modern term “albino” is recent, so term-based search cannot reach earlier material, and the descriptive vocabulary spreads across clauses and varies with period and genre.

Method. We describe a pilot that searches from both ends of the timeline. Descriptive anchors (for example white hair “like wool” or “like snow”) are used to retrieve passages from earlier texts, while the modern term and its historical neighbours are used to search backward. Retrieval works at sentence and short-passage level, with hair, skin and eye features annotated separately. We use [a historical language model, for example MacBERTh, for the period it covers (confirm coverage) and a modern model for later texts]. We run the same queries in each model separately and compare the passages they retrieve and their neighbours, rather than aligning vectors across models. Quotation of biblical sources is tagged and kept in the data, not filtered out, so that it can be analysed as reception.

Preliminary results. A first query based on “hair as white as wool” retrieved [number] candidate passages from [corpus], including [number] from Welsh mythological material. [Add what you found: precision on a hand-checked sample, example passages, and any surprises.]

Limitations and next steps. The phenomenon is rare, so counts are small, and the category “albino-like” is our own construction, not a historical one. We therefore treat retrieval as a way of generating candidates for close reading and validate it against a hand-annotated sample. We also annotate whether each passage records a reaction to the described features, since surviving texts favour encounters that provoked one.

Significance. The approach lets humanities researchers survey recurring symbolic complexes across centuries without presupposing their equivalence, and it is designed to be reusable for other rare descriptive categories.