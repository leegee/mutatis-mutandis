**Results summary for the phrase “white as wool” (25 occurrences, full corpus 100–2020, scope=all)**

```
python src/passage_search_pipeline/passages_4_mask_fillers.py --phrase "white/whyte as wool" --start-year 100 --end-year 2020 --min-occurrences 1 --scope all
-min-occurrences 1 --scope all
2026-10-14 21:38:14 Loading MacBERTh...
2026-10-14 21:38:15 Building groups...
2026-10-14 21:44:15
=== white as wool | 100-2020 | scope=all (25 occurrences) ===
2026-10-14 21:44:15   check: red like crimson , they shall be as [MASK] . if thou fearest god will hardly be
2026-10-14 21:44:16          -> wooll, wool, white, scarlet, snow
2026-10-14 21:44:32   baselines: 5 replicates; seed family (as, whit, whyt, wool) takes p=0.537 of the mass and is excluded below.
2026-10-14 21:44:32
  SUBSTITUTES (alternatives in the seed's slot; variants pooled)
2026-10-14 21:44:32   stability: mean pairwise top-50 overlap across baselines = 0.80
2026-10-14 21:44:32   filler                      mean_p   base_p   score  95% CI           support  top%  stab
2026-10-14 21:44:32   scarlet                     0.0476  0.00000  0.5775  [0.052,1.501]     10/25    59%   5/5
2026-10-14 21:44:32   clean                       0.0490  0.00112  0.4408  [0.024,1.211]     10/25    41%   5/5
2026-10-14 21:44:32   snow                        0.0284  0.00013  0.2897  [0.069,0.619]     11/25    28%   5/5
2026-10-14 21:44:32   black                       0.0120  0.00001  0.1116  [0.030,0.217]      7/25    32%   5/5
2026-10-14 21:44:32   forgiven/forgiuen           0.0048  0.00000  0.0422  [0.006,0.104]      3/25    53%   5/5
2026-10-14 21:44:32   purple                      0.0043  0.00000  0.0373  [0.009,0.081]      4/25    41%   5/5
2026-10-14 21:44:32   pardoned/pardons            0.0048  0.00011  0.0307  [0.002,0.095]      5/25    55%   5/5
2026-10-14 21:44:32   red                         0.0039  0.00003  0.0276  [0.013,0.050]      8/25    18%   5/5
2026-10-14 21:44:32   comely                      0.0033  0.00005  0.0242  [0.003,0.055]      3/25    41%   5/5
2026-10-14 21:44:32   soft                        0.0029  0.00026  0.0190  [0.000,0.052]      5/25    43%   4/5
2026-10-14 21:44:32   gold                        0.0019  0.00010  0.0112  [0.001,0.022]      4/25    20%   4/5
2026-10-14 21:44:32   healed/healing              0.0018  0.00007  0.0097  [0.000,0.027]      3/25    44%   5/5
2026-10-14 21:44:32   sin/sins/sinne              0.0027  0.00032  0.0096  [0.001,0.033]      3/25    46%   5/5
2026-10-14 21:44:32
  COLLOCATES (windows containing the word, min 3; --require words omitted)
2026-10-14 21:44:32   stability: mean pairwise top-50 overlap across baselines = 1.00
2026-10-14 21:44:32   word                       windows   p_seed   p_base   score  95% CI           stab
2026-10-14 21:44:32   scarlet                         11    0.440   0.0192  1.9870  [0.973,3.236]     5/5
2026-10-14 21:44:32   sins                            15    0.600   0.0885  1.8502  [0.706,3.637]     5/5
2026-10-14 21:44:32   snow                            10    0.400   0.0192  1.7514  [0.770,2.978]     5/5
2026-10-14 21:44:32   red                              9    0.360   0.0192  1.5215  [0.489,2.724]     5/5
2026-10-14 21:44:32   crimson                          7    0.280   0.0192  1.0819  [0.317,2.114]     5/5
2026-10-14 21:44:32   hairs                            6    0.240   0.0192  0.8740  [0.165,1.987]     5/5
2026-10-14 21:44:32   ancient                          5    0.200   0.0269  0.6123  [0.038,1.522]     5/5
2026-10-14 21:44:32   dan                              4    0.160   0.0192  0.4891  [0.042,1.298]     5/5
2026-10-14 21:44:32   sinner                           4    0.160   0.0192  0.4891  [0.042,1.195]     5/5
2026-10-14 21:44:32   tender                           3    0.120   0.0192  0.3170  [0.020,0.874]     5/5
2026-10-14 21:44:32
Saved 23 rows to: D:\src\mutatis-mutandis\out\masked_fillers.csv
```

The model and the collocate statistics both recover the classic biblical image extremely cleanly.

### Substitutes (what the model thinks belongs in the masked slot)
Top items by contribution to divergence from the matched baseline:

| Rank | Filler              | mean_p | score  | support | notes |
|------|---------------------|--------|--------|---------|-------|
| 1    | scarlet             | 0.048  | 0.58   | 10/25   | strongest alternative |
| 2    | clean               | 0.049  | 0.44   | 10/25   | purification sense |
| 3    | snow                | 0.028  | 0.29   | 11/25   | the parallel image in Isa 1:18 |
| 4    | black               | 0.012  | 0.11   | 7/25    | colour contrast |
| 5–8  | forgiven / purple / pardoned / red | … | … | … | theological and colour terms |

The seed family itself (“white/whyte … as … wool/wooll”) accounts for more than half the probability mass and is correctly excluded. Stability across the five baseline replicates is high (mean pairwise top-50 Jaccard ≈ 0.80).

### Collocates (words that co-occur in the windows far more than in matched baselines)
| Word     | windows | p_seed | score |
|----------|---------|--------|-------|
| scarlet  | 11      | 0.44   | 1.99  |
| sins     | 15      | 0.60   | 1.85  |
| snow     | 10      | 0.40   | 1.75  |
| red      | 9       | 0.36   | 1.52  |
| crimson  | 7       | 0.28   | 1.08  |
| hairs    | 6       | 0.24   | 0.87  |
| ancient / dan / sinner / tender | … | … | … |

Again, stability is essentially perfect (1.00).

### Interpretation
Virtually every high-scoring item is drawn from the well-known Isaianic / penitential cluster:

> “though your sins be as **scarlet**, they shall be as white as **snow**; though they be **red** like **crimson**, they shall be as **wool**.”

“Hairs” and “ancient” point to the related image in Revelation 1:14 (“his head and his hairs were white like wool, as white as snow”). The model’s substitutes (“forgiven”, “pardoned”, “clean”, “healed”) show that it has also internalised the theological force of the simile, not merely the surface colours.

The matched-document baseline + multi-replicate design is working as intended: the genre/period effects that would otherwise dominate (religious vocabulary in general) are largely cancelled out, leaving the specific collocational and substitutable neighbourhood of the phrase.

### Key take-aways from the numbers

* Scarlet / snow / red / crimson dominate both lists with very high lifts and tight stability. This is almost pure recovery of Isaiah 1:18 (and its many early-modern citations).
* Sins is the strongest pure collocate (15/25 windows), confirming the penitential frame.
* Hairs + ancient point to the secondary Revelation 1:14 cluster (“hairs … white like wool, as white as snow”).
* Theological substitutes (forgiven, pardoned, clean, healed) appear with solid support even though they are not surface colour terms — the model has learned the force of the simile.
* All top items have perfect or near-perfect stability across the five matched-document baseline replicates, and the confidence intervals for the highest ranks stay well above zero.

The analysis is working exactly as designed: the matched baseline successfully suppresses the general religious vocabulary of the period and surfaces the specific neighbourhood of the phrase.

**The 95% CI is a bootstrap confidence interval on the score itself.**

### What is being intervalled
The score for each candidate is:

\[
\text{score} = \overline{p}_{\text{seed}} \cdot \log_2 \Bigl( \frac{\overline{p}_{\text{seed}}}{\overline{p}_{\text{base}}} \Bigr)
\]

(i.e. the contribution of that word to the pointwise KL divergence between the seed distribution and the baseline).

### How the interval is constructed
1. There are \(n\) seed occurrences (here \(n=25\)).
2. There are \(R\) independent baseline replicates (here \(R=5\)).
3. For each of 300 bootstrap draws (`--bootstrap 300`):
   - Resample the \(n\) seed occurrences **with replacement**.
   - Pair them with one of the baseline replicates (cycling through the \(R\) baselines).
   - Recompute the score on that resampled pair.
4. The 2.5th and 97.5th percentiles of those 300 scores become `ci_lo` and `ci_hi`.

This is therefore a **paired bootstrap** that accounts for both:
- sampling variability among the seed occurrences, and
- variability across the different baseline draws.

### How to read it
- If the entire interval lies **above zero**, the positive association is statistically credible under the bootstrap (the default behaviour of the script is to drop any item whose interval includes ≤ 0, unless you pass `--keep-uncertain`).
- A very wide interval (e.g. scarlet’s [0.05, 1.50]) usually means the mass is concentrated in only a few of the 25 occurrences (see the high `top_share`).
- A narrower interval that still stays well above zero (e.g. red’s [0.013, 0.050]) indicates a more consistent, moderate signal across many occurrences.

In short: the 95% CI quantifies how stable the reported score is under resampling of both the seed windows and the matched baselines.

