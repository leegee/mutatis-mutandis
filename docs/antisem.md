## Sources

### Dataset from the Institute for the Study of Contemporary Antisemitism (ISCA) at Indiana University:

https://zenodo.org/records/14448399

### Reddit and BlueSky Observations pertaining to Antisemitism Research (June 2024 to August 2025)

https://zenodo.org/records/17798789

Hidden in plain sight: antisemitic content in QAnon subreddits

https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0318988

### Useful related ISCA / Jikeli sets

Antisemitism after 10/7 (Nov–Dec 2023, keyword "Israel")
https://zenodo.org/records/17523796

Calling out antisemitism (re-annotation layer)
https://zenodo.org/records/17524915

Antisemitism on X – trends before/after Oct 7
https://zenodo.org/records/15025646

X + TikTok annotations around Oct 2025 hostage release
https://zenodo.org/records/18203016


## Prior Art, via Claude

Mustafa & Japkowicz, “Monitoring the evolution of antisemitic discourse on extremist social media using BERT” (arXiv 2403.05548). For each batch of incoming posts, they generate contextual embeddings with a fine-tuned BERT and feed them into a divisive clustering method that updates the existing clustering incrementally. The aim is to extract antisemitic themes and terminology over time and capture how they evolve. This is essentially your pamphlet pipeline applied to social media: embed, cluster, and watch the clusters shift per time slice. Read it first, especially its comparison table of prior work.


Kikkisetti, Mustafa, Melillo, Corizzo, Boukouvalas, Gill & Japkowicz (2024), “Using LLMs to discover emerging coded antisemitic hate-speech in extremist social media,” arXiv 2401.10841. This is the closest match to your goal. The Mustafa paper cites it for its term-extraction method. It extracts representative expressions from posts, filters out incoherent and previously seen ones, scores semantic similarity to known antisemitic terms with a fine-tuned language model, and removes expressions that explicitly mention Jewish topics, leaving only coded ones. Its working hypothesis is that coded antisemitic terminology begets more coded terminology, so it harvests terms used in contexts similar to known coded terms. The known terms come from sources such as the AJC “Translate Hate” glossary. That is your frame-based substitute discovery, applied to social media.

Chandra et al. (2021) is the multimodal paper I mentioned earlier. The Mustafa paper describes it as a RoBERTa classifier sorting antisemitic content into political, economic, religious and racial categories.

Barna & Knap (2021) is a Hungarian COVID-era study. It used LDA topic modelling with qualitative discourse analysis.

Tripodi et al., “Tracing Antisemitic Language Through Diachronic Embedding Projections: France 1789-1914” (arXiv 1906.01440). It uses diachronic word embeddings and embedding projections over six streams of antisemitic discourse in French texts. It is a historical-corpus precedent for the same diachronic-embedding technique you use on EEBO, and likely more methodologically relevant to your pamphlet work than the social media papers.

--

Chandra et al., “Subverting the Jewtocracy” (2021). They present the first multimodal detection work for online antisemitism, with labelled datasets of 3,102 Twitter posts and 3,509 Gab posts, and a system using both text and images. It is useful for the multimodal problem, and it is an early benchmark.

Jikeli et al., “Antisemitic Messages? A Guide to High Quality Annotation and a Labeled Dataset of Tweets” (Indiana University). It offers 6,941 English tweets from 2019 to 2021, of which 1,250 (18%) were labelled antisemitic, annotated against the IHRA definition. The annotation guide, which asks annotators to specify which part of the definition applies, speaks directly to the definitional care I raised earlier.

A German dataset covering antisemitism and conspiracy theories, and the Decoding Antisemitism project. The latter analyses media comments in English, German and French. I didn’t open either source, so check the papers directly.

Dog whistle work: Mendelsohn et al., “From Dogwhistles to Bullhorns” (ACL 2023), and Kruk et al. on LLM dog whistle disambiguation. These are the most relevant for coded vocabulary.

Domain models and benchmarks: HateBERT (Caselli et al.), HateXplain, and BERTweet or XLM-R fine-tunes. These are common baselines.
