**Title**
Discovering lexical neighbourhoods in historical English with matched baselines and a historically pre-trained language model

**Authors**
Lee Goddard, unaffiliated

**Structured abstract**

**Purpose**
To present a practical, statistically disciplined pipeline for discovering the lexical neighbourhood (substitutes and collocates) of seed words and short phrases in large historical English corpora.

**Design/methodology/approach**
The method combines MacBERTh (a BERT model pre-trained on English 1450–1950) with document-matched baselines, multiple independent baseline replicates, light orthographic stemming, and exploratory paired bootstrap intervals. Substitutes are obtained by masking the seed and scoring the model’s probability mass; collocates are obtained by comparing window presence rates. Both analyses share the same experimental controls.

**Findings**
On three historically prominent phrases—“white as wool” (Isaiah 1:18), “salt of the earth” (Matthew 5:13), and “body politic”—the pipeline recovers the expected lexical clusters with high baseline-replicate ranking stability while suppressing generic background vocabulary.

**Originality**
The combination of matched-document baselines, multi-replicate stability metrics, and uncertainty quantification applied simultaneously to both distributional substitutes and frequency collocates is rarely implemented in published historical-NLP work.

**Contribution to the field of Digital Humanities**
The approach offers a reproducible, low-barrier tool for exploratory lexical analysis of Early Modern and Late Modern English that respects the orthographic and documentary structure of the data.

**Keywords**
historical English, MacBERTh, collocates, substitutes, matched baseline, EEBO, digital philology, body politic

---

**1. Introduction**

Scholars working with large historical corpora such as Early English Books Online (EEBO) frequently wish to discover the lexical neighbourhood of a word or short phrase: what other words appear in the same semantic slot (substitutes) and what words co-occur in its immediate context (collocates). Classic collocation methods remain valuable, yet they are sensitive to genre and topic effects that are especially pronounced in early-modern print. Contextualised language models trained on historical text offer a complementary route through the masked-language-modelling objective (Manjavacas and Fonteyn, 2021, 2022; Fonteyn and Manjavacas, 2022).

Several recent projects have used historically pre-trained transformers for semantic querying or sense exploration (Fonteyn and Manjavacas, 2022; Ryan et al., 2023). The present note contributes a concrete experimental design that strengthens the statistical credibility of such analyses. Three design choices are central:

- every seed occurrence is paired with baseline windows drawn from the *same* document (matched baseline);
- the baseline is redrawn multiple times and both mean scores and a baseline-replicate ranking-stability metric are reported;
- a paired bootstrap supplies exploratory 95 % intervals on the score itself, allowing items whose interval includes zero to be filtered.

The pipeline also handles multi-word phrases and pools orthographic variants by a light stemming procedure. Three short case studies—two biblical and one secular—illustrate the behaviour of the method.

**2. Method**

**2.1 Data and model**
Occurrences are sampled from a PostgreSQL index of EEBO-TCP (and optionally later material). The language model is MacBERTh (Manjavacas and Fonteyn, 2021, 2022).

**2.2 Seed sampling and context windows**
For a single word or a short phrase (with optional gap between slots), up to a user-specified number of occurrences are drawn, limited per document. A fixed window of tokens on either side is retrieved. Orthographic variants of the seed itself are recorded and later excluded from the candidate lists.

**2.3 Substitutes**
Each occurrence is tokenised and the entire seed span is replaced by a single `[MASK]` token. The model’s softmax distribution at the mask position is aggregated into groups that share a light stem (u/v, i/j, final -e, plurals, -ed/-ing/-eth, doubled consonants). Probability mass belonging to the seed’s own stem family is reported but excluded from ranking.

For each candidate group \(j\) we compute a baseline-relative predictive score

\[
s_j = \bar{p}_j \cdot \log_2 \Bigl( \frac{\bar{p}_j + \varepsilon}{B_j + \varepsilon} \Bigr),
\]

where \(\bar{p}_j\) is the mean model-assigned probability across seed occurrences and \(B_j\) is the corresponding mean under the baseline replicates. The score is calculated only over the eligible (grouped) vocabulary; it is therefore a KL-inspired contribution rather than a full Kullback–Leibler divergence over the model’s entire output distribution. We report the total probability mass captured by the eligible groups so that the coverage of the candidate vocabulary is transparent.

**2.4 Collocates**
Presence of each stemmed content word inside the seed windows is compared with its presence rate in the baseline windows. We rank candidates by a presence-rate enrichment score

\[
s = p_{\text{seed}} \cdot \log_2 \Bigl( \frac{p_{\text{seed}}}{p_{\text{base}}} \Bigr),
\]

where \(p_{\text{seed}}\) and \(p_{\text{base}}\) are the proportions of windows containing the word. This is a frequency-weighted log-ratio, not a categorical KL divergence.

**2.5 Matched baselines and replicates**
By default each seed window is paired with one or more non-overlapping windows drawn from the same document. Same-document sampling partially controls for document-level vocabulary, genre conventions and broad subject matter; it does not eliminate local topic, authorial style, or position effects within the document. The baseline set is redrawn independently several times (default five). Scores are averaged across replicates; a stability figure records the fraction of replicates in which an item ranked among the top \(k\).

**2.6 Uncertainty and stability**
A paired bootstrap that resamples seed occurrences while cycling through the pre-computed baseline replicates yields exploratory 95 % percentile intervals. These intervals are conditional on the observed baseline draws and do not fully propagate uncertainty arising from baseline sampling or document clustering.

We also report baseline-replicate ranking stability: the mean pairwise Jaccard similarity of the top-\(k\) sets across replicates, and the fraction of replicates in which each candidate appears in the top \(k\). This measures sensitivity to the particular baseline draws, not semantic validity.

**3. Case studies**

All three phrases were queried across the full available span with the default matched-baseline settings.

**3.1 “White as wool”** (25 occurrences)
Top substitutes: *scarlet*, *clean*, *snow*, *black*, *forgiven*, *purple*, *pardoned*, *red*.
Top collocates: *scarlet*, *sins*, *snow*, *red*, *crimson*, *hairs*.

The recovered vocabulary maps directly onto Isaiah 1:18 and the related description in Revelation 1:14. Baseline-replicate ranking stability across the five replicates was essentially perfect.

**3.2 “Salt of the earth”** (300 occurrences)
Top substitutes: *light/lights*, *apostles*, *church/churches*, *stars*, *life*, *word*, *sun*, *truth*, *lamps*, *preachers*.
Top collocates: *light*, *world*, *season*, *mat(thew)*, *savour*, *ministers*, *wherewith*, *lost*, *unsavoury*, *disciples*.

The dominant cluster is the Matthean pairing itself (Matthew 5:13–14) together with the warning about losing savour. A secondary cluster reflects the early-modern ministerial application of the image. Mean pairwise top-50 baseline-replicate ranking stability was 0.65 (substitutes) and 0.73 (collocates).

**3.3 “Body politic”** (217 occurrences)
Top substitutes: *state*, *church*, *kingdom*, *corporation*, *community*, *commonwealth*, *people*, *soul*, *politique*, *civil*.
Top collocates: *naturall*, *head*, *members*, *part*, *corporation*, *whole*, *common*, *society*, *purge*, *humours*.

The method recovers the classic early-modern organic metaphor of the state: the standing contrast with the *body natural*, the king as *head*, the citizens as *members*, and the associated medical-political vocabulary (*disease*, *purge*, *humours*, *blood*). Baseline-replicate ranking stability remained high (0.70 for substitutes, 0.60 for collocates).

In all three cases the matched-document baseline reduced the influence of broader background vocabulary relative to a period-wide sample, leaving the historically specific neighbourhood of each phrase.

**4. Discussion**

The components of the pipeline are individually familiar. Masked prediction with historical transformers has been demonstrated (Fonteyn and Manjavacas, 2022; Ryan et al., 2023), and collocation analysis has a long history in corpus linguistics. The contribution lies in the joint experimental regime: matched baselines that partially control for document-level vocabulary, multi-replicate ranking stability, and exploratory bootstrap intervals that make uncertainty visible. These controls are especially useful when the analyst is exploring rather than testing a pre-specified hypothesis.

The three case studies span religious and secular discourse. “White as wool” and “salt of the earth” recover well-known biblical clusters; “body politic” recovers a central early-modern political metaphor. The consistency of the results across these domains suggests that the design is not limited to a single register.

Limitations remain. The method inherits whatever biases exist in the underlying corpus and in MacBERTh’s training data. Very rare seeds still yield wide intervals. The matched-document baseline reduces the influence of document-level vocabulary relative to a purely period-wide sample, but residual local dependence remains; the intervals are therefore presented as exploratory and conditional on the sampled baseline replicates. Users must inspect the underlying passages—a step facilitated by the accompanying code, which records document identifiers, corpus, title, and year for every seed occurrence.

**5. Conclusion**

The pipeline offers a practical, reproducible way to explore lexical neighbourhoods in historical English while partially controlling for document-level vocabulary through matched baselines. The recovery of the expected clusters for “white as wool”, “salt of the earth”, and “body politic” shows that the design works across both religious and political discourse. Code and documentation will be released so that others can apply the same controls to their own seeds and periods.

**References**

Fonteyn, L. and Manjavacas, E. (2022). Software demonstration: meaning-based querying of historical corpora with MacBERTh. DHBenelux 2022. https://doi.org/10.5281/zenodo.6514385

Manjavacas, E. and Fonteyn, L. (2021). MacBERTh: Development and evaluation of a historically pre-trained language model for English (1450–1950). *Proceedings of the Workshop on Natural Language Processing for Digital Humanities (NLP4DH)*, 23–36. Association for Computational Linguistics.

Manjavacas, E. and Fonteyn, L. (2022). Adapting vs. pre-training language models for historical languages. *Journal of Data Mining & Digital Humanities*. https://doi.org/10.46298/jdmdh.9152

Ryan, Y. C., Liimatta, A., Mäkelä, E., et al. (2023). Using ECCO-BERT and the Historical Thesaurus of English to explore concepts and agency in historical writing: Interpreting the eighteenth-century luxury debate. Digital Humanities 2023. https://doi.org/10.5281/zenodo.8108032
