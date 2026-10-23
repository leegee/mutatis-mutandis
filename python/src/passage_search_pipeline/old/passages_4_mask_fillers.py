"""
tier1_passages/passages_4_mask_fillers.py

Vocabulary discovery around seed words, two ways:

  substitute  Mask the seed with a single [MASK] and sum MacBERTh's MLM
              distribution over the vocabulary. Words the model finds
              plausible *in the seed's slot*, beyond what it predicts for a
              random masked word. (Synonyms, antonyms, near-synonyms.)

  collocate   Words that appear in the seed's context windows far more often
              than in baseline windows. (Topical neighbours.)

Design points
-------------
* Matched baseline (default). Each seed occurrence is paired with baseline
  window(s) from the SAME document, at a non-overlapping position, so genre,
  topic and spelling conventions of the seed-bearing documents don't count as
  seed-specific vocabulary. `--baseline period` gives the older behaviour
  (random windows from the period).
* Replicates. The baseline is redrawn --replicates times. Every score is the
  mean over replicates, and a `stability` figure says in how many replicates
  the item reached the top --stability-k.
* Support and uncertainty. Substitutes keep per-occurrence probabilities:
  `support` is the number of occurrences giving the word at least
  --support-p; `top_share` is the largest single occurrence's share of the
  word's total mass. A paired bootstrap (occurrences x baselines) gives a 95%
  interval. Items with support < --min-support, or an interval reaching 0,
  are dropped (--keep-uncertain keeps them).
* Variants. Words are grouped by a light stem (u/v, i/j, -ie/-y, plural,
  -ed/-ing/-eth, final -e, doubled consonant), so resisted/resisting/resiste
  are one family. The seed's own family is excluded from the alternatives and
  reported as probability mass.

Sources
-------
  --source db   (default) sample occurrences from the corpus database
  --source csv  use passages from a passage_vocabulary_benchmark.csv

Scope (db only)
---------------
  --scope pamphlet  (default) the filtered pamphlet_corpus view
  --scope all       every English document in `documents` (all of EEBO, or
                    other corpora if you change the years)

Examples
--------
    python passages_4_mask_fillers.py --seed resist,resiste
    python passages_4_mask_fillers.py --seed resist,resiste --scope all \
        --start-year 1550 --end-year 1700 --max-per-doc 10 \
        --forbid god,lord,sathan,devil,christ,faith,temptation,sinne
    python passages_4_mask_fillers.py --seed libertie,liberty --period-size 25
    python passages_4_mask_fillers.py --seed resist --exclude-forms resistance
    python passages_4_mask_fillers.py --source csv --seed defence
    python passages_4_mask_fillers.py --phrase "take/tooke/taketh heed/heede" \
        --scope all --start-year 1550 --end-year 1700 --phrase-gap 1
    python passages_4_mask_fillers.py --phrase "white/whyte as white/whyte wool/wolle" \
        --scope all --start-year 1550 --end-year 1700 --phrase-gap 1

Single words and phrases
------------------------
Use --seed for single words (comma-separated spelling variants). Use --phrase
for multi-word seeds (db only): space separates words, "/" separates spelling
variants within a word. --phrase-gap N allows up to N other tokens between
consecutive words. The whole phrase is replaced by ONE [MASK] for the
substitute analysis; its words are left out of the collocates and excluded
from the alternatives.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from lib.corpus_logging import logger
from lib.corpus_config import OUT_DIR
from lib.macberth import load_macberth
from lib.passage_stem import (
    STOP,
    stem_key,
    Groups,
    build_groups,
    content_word,
    span_of,
)
from lib.passage_db import (
    SCOPES,
    fetch_hits,
    fetch_phrase_hits,
    fetch_doc_lengths,
    matched_hits,
    fetch_doc_meta,
    fetch_random_hits,
    fetch_windows,
    windows_to_passages,
    random_positions,
    one_position_per_passage,
)

MAX_LEN = 512
EPS = 1e-5


# ---------------------------------------------------------------------------
# CSV source
# ---------------------------------------------------------------------------

def load_passages(path: Path) -> list[dict]:
    """One row per passage_id; passages repeat across queries/reps."""
    logger.info(f"Loading passpages from {path}")
    seen = set()
    passages = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["passage_id"] in seen:
                continue
            seen.add(row["passage_id"])
            text = row["passage_text"] or ""
            if not text.strip():
                continue
            passages.append({
                "doc_id": row["doc_id"],
                "word_start": int(row["word_start"]),
                "words": text.split(),
            })
    return passages


def find_occurrences(passages, variants):
    """
    (passage_index, word_index) for each occurrence of any variant,
    de-duplicated on absolute (doc_id, position) because windows overlap.
    """
    seen = set()
    out = []
    for p_idx, p in enumerate(passages):
        for w_idx, w in enumerate(p["words"]):
            if w.lower() in variants:
                key = (p["doc_id"], p["word_start"] + w_idx)
                if key not in seen:
                    seen.add(key)
                    out.append((p_idx, w_idx))
    return out


def filter_context(passages, occ, require, forbid):
    """
    Keep occurrences whose window contains at least one required word
    (if any are given) and none of the forbidden words.
    """
    if not require and not forbid:
        return occ
    out = []
    for p, w in occ:
        s, e = span_of(passages[p], w)
        window = {x.lower() for i, x in enumerate(passages[p]["words"])
                  if not s <= i <= e}
        if require and not (window & require):
            continue
        if forbid and (window & forbid):
            continue
        out.append((p, w))
    return out


def _masked_batches(macberth, passages, occurrences, batch_size):
    """
    Yield (B, V) softmax distributions at the masked position, per batch.

    The target word becomes ONE [MASK] even if it tokenises to several
    wordpieces, so predictions are always single wordpieces.
    """
    tok = macberth.tokenizer
    dev = macberth.device

    for off in range(0, len(occurrences), batch_size):
        seqs, mask_pos = [], []

        for p_idx, w_idx in occurrences[off:off + batch_size]:
            enc = tok(
                passages[p_idx]["words"],
                is_split_into_words=True,
                truncation=True,
                max_length=MAX_LEN,
            )
            ids = enc["input_ids"]
            s, e = span_of(passages[p_idx], w_idx)
            pieces = [i for i, wid in enumerate(enc.word_ids())
                      if wid is not None and s <= wid <= e]
            if not pieces:          # word fell off the truncated end
                continue
            ids = ids[:pieces[0]] + [tok.mask_token_id] + ids[pieces[-1] + 1:]
            seqs.append(ids)
            mask_pos.append(pieces[0])

        if not seqs:
            continue

        maxlen = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), maxlen), tok.pad_token_id, dtype=torch.long)
        attention = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for r, s in enumerate(seqs):
            input_ids[r, :len(s)] = torch.tensor(s)
            attention[r, :len(s)] = 1

        out = macberth.predict_masked(
            input_ids=input_ids.to(dev),
            attention_mask=attention.to(dev),
        )
        rows = torch.arange(len(seqs), device=out.logits.device)
        cols = torch.tensor(mask_pos, device=out.logits.device)
        yield out.logits[rows, cols].float().softmax(-1)


@torch.inference_mode()
def grouped_probs(macberth, passages, occurrences, groups, batch_size, keep_rows):
    """
    Probability mass per word group at each masked position.

    Returns (result, n, vocab_sum):
      result     (n, G) float32 matrix if keep_rows, else (G,) summed vector
                 (None if nothing could be masked)
      n          number of masked positions used
      vocab_sum  (V,) summed per-vocabulary-entry probability, for naming
    """
    chunks, total, vocab_sum, n = [], None, None, 0

    for probs in _masked_batches(macberth, passages, occurrences, batch_size):
        key = str(probs.device)
        if key not in groups.cache:
            groups.cache[key] = (
                torch.from_numpy(groups.sel).to(probs.device),
                torch.from_numpy(groups.group_of[groups.sel]).to(probs.device),
            )
        sel, gidx = groups.cache[key]

        g = torch.zeros((probs.shape[0], groups.n), dtype=torch.float32,
                        device=probs.device)
        g.index_add_(1, gidx, probs[:, sel])

        vs = probs.sum(0).double().cpu().numpy()
        vocab_sum = vs if vocab_sum is None else vocab_sum + vs

        if keep_rows:
            chunks.append(g.cpu().numpy())
        else:
            gs = g.sum(0).double().cpu().numpy()
            total = gs if total is None else total + gs
        n += g.shape[0]

    if n == 0:
        return None, 0, None
    if keep_rows:
        return np.concatenate(chunks, axis=0), n, vocab_sum
    return total, n, vocab_sum


@torch.inference_mode()
def top_tokens(macberth, passages, occ_one, k=5):
    for probs in _masked_batches(macberth, passages, [occ_one], 1):
        ids = probs[0].topk(k).indices.tolist()
        return macberth.tokenizer.convert_ids_to_tokens(ids)
    return []


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def mean_jaccard(sets):
    vals = [len(a & b) / len(a | b) for a, b in combinations(sets, 2) if a | b]
    return sum(vals) / len(vals) if vals else None


def score_substitutes(M, B, excluded, args, rng):
    """
    M  (n, G) per-occurrence group probabilities
    B  (R, G) per-replicate baseline mean probabilities
    excluded  group indices to drop (the seed's own family)

    Score = mean_p * log2(mean_p / base_p): contribution of the word to the
    divergence from the baseline, averaged over replicates.

    Returns (items, overlap) where items are dicts, best first.
    """
    n, G = M.shape
    R = B.shape[0]
    mean_p = M.mean(0).astype(np.float64)

    S = mean_p[None, :] * np.log2((mean_p[None, :] + EPS) / (B + EPS))
    if len(excluded):
        S[:, list(excluded)] = -np.inf
    score = S.mean(0)

    K = args.stability_k
    tops = [set(np.argsort(-S[r])[:K].tolist()) for r in range(R)]
    overlap = mean_jaccard(tops)

    order = np.argsort(-score)[:args.candidates]
    cand = np.array([g for g in order if np.isfinite(score[g]) and score[g] > 0],
                    dtype=int)
    if len(cand) == 0:
        return [], overlap

    P = M[:, cand].astype(np.float64)
    support = (P >= args.support_p).sum(0)
    top_share = P.max(0) / np.maximum(P.sum(0), 1e-12)

    draws = np.empty((args.bootstrap, len(cand)))
    for b in range(args.bootstrap):
        idx = rng.integers(0, n, n)
        mp = P[idx].mean(0)
        base_b = B[b % R][cand]
        draws[b] = mp * np.log2((mp + EPS) / (base_b + EPS))
    ci_lo, ci_hi = np.percentile(draws, [2.5, 97.5], axis=0)

    base_mean = B.mean(0)
    items = []
    for j, g in enumerate(cand):
        items.append({
            "group": int(g),
            "mean_p": mean_p[g], "base_p": base_mean[g],
            "score": score[g], "ci_lo": ci_lo[j], "ci_hi": ci_hi[j],
            "support": int(support[j]), "top_share": top_share[j],
            "stability": sum(int(g) in t for t in tops) / R,
        })

    keep = []
    for it in items:
        if it["support"] < args.min_support:
            continue
        if not args.keep_uncertain and it["ci_lo"] <= 0:
            continue
        keep.append(it)
    return keep, overlap


def score_collocates(passages, occ, base_lists, excl_keys, min_c, args, rng):
    """
    Presence of each (stem-grouped) word in the seed windows versus the
    baseline windows, once per window. Score = p_seed * log2(p_seed / p_base).
    """
    surface = defaultdict(Counter)

    def bag(words, skip=(), record=False):
        keys = set()
        for i, w in enumerate(words):
            if i in skip:
                continue
            lw = w.lower()
            if not lw.isalpha() or len(lw) < 2 or lw in STOP:
                continue
            k = stem_key(lw)
            if k in excl_keys:
                continue
            if record:
                surface[k][lw] += 1
            keys.add(k)
        return keys

    seed_bags = []
    for p, w in occ:
        s, e = span_of(passages[p], w)
        seed_bags.append(bag(passages[p]["words"], range(s, e + 1), record=True))
    seed_c = Counter()
    for b in seed_bags:
        seed_c.update(b)
    cand_keys = [k for k, c in seed_c.items() if c >= min_c]
    if not cand_keys:
        return [], None

    col = {k: i for i, k in enumerate(cand_keys)}
    n, W = len(seed_bags), len(cand_keys)
    P = np.zeros((n, W), dtype=bool)
    for i, b in enumerate(seed_bags):
        for k in b:
            j = col.get(k)
            if j is not None:
                P[i, j] = True

    R = len(base_lists)
    PB = np.zeros((R, W))
    for r, bps in enumerate(base_lists):
        c = Counter()
        for p in bps:
            c.update(bag(p["words"]))
        nb = len(bps)
        PB[r] = [(c.get(k, 0) + 0.5) / (nb + 1) for k in cand_keys]

    ps = P.mean(0)
    S = ps[None, :] * np.log2(ps[None, :] / PB)
    score = S.mean(0)

    K = args.stability_k
    tops = [set(np.argsort(-S[r])[:K].tolist()) for r in range(R)]
    overlap = mean_jaccard(tops)

    order = np.argsort(-score)[:args.candidates]
    cand = np.array([j for j in order if score[j] > 0], dtype=int)
    if len(cand) == 0:
        return [], overlap

    Pc = P[:, cand].astype(np.float64)
    draws = np.empty((args.bootstrap, len(cand)))
    for b in range(args.bootstrap):
        idx = rng.integers(0, n, n)
        psb = Pc[idx].mean(0)
        pb = PB[b % R][cand]
        with np.errstate(divide="ignore", invalid="ignore"):
            d = psb * np.log2(np.maximum(psb, 1e-12) / pb)
        draws[b] = np.where(psb > 0, d, 0.0)
    ci_lo, ci_hi = np.percentile(draws, [2.5, 97.5], axis=0)

    pb_mean = PB.mean(0)
    items = []
    for jj, j in enumerate(cand):
        k = cand_keys[j]
        it = {
            "word": surface[k].most_common(1)[0][0],
            "p_seed": ps[j], "p_base": pb_mean[j], "score": score[j],
            "ci_lo": ci_lo[jj], "ci_hi": ci_hi[jj],
            "support": int(P[:, j].sum()),
            "stability": sum(int(j) in t for t in tops) / R,
        }
        if it["support"] < min_c:
            continue
        if not args.keep_uncertain and it["ci_lo"] <= 0:
            continue
        items.append(it)
    return items, overlap


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse(macberth, label, period, scope_label, passages, occ, exclude_forms,
            groups, baseline_for, args, rows, checked, nprng):
    tok = macberth.tokenizer
    R = args.replicates

    # diagnostic: list every seed occurrence
    logger.info(f"\n== Seed occurrences ({len(occ)}) ==")
    for p_idx, w_idx in occ:
        p = passages[p_idx]
        s, e = span_of(p, w_idx)
        lo = max(0, s - 12)
        hi = min(len(p["words"]), e + 13)
        window = " ".join(
            ("[MASK]" if s <= (lo + j) <= e else w)
            for j, w in enumerate(p["words"][lo:hi])
        )
        logger.info(f"    {p['doc_id']}:{p['word_start'] + w_idx}  {window}")

    logger.info(f"\n=== {label} | {period} | scope={scope_label} ({len(occ)} occurrences) ===")
    if len(occ) < args.min_occurrences:
        logger.info(f"  Skipped: {len(occ)} occurrences < --min-occurrences "
              f"{args.min_occurrences}. Widen the years, use --scope all, raise "
              f"--max-per-doc, loosen --require/--forbid, or lower the threshold.")
        return

    excl_keys = {stem_key(f.lower()) for f in exclude_forms if f.isalpha()}
    excl_groups = [groups.index[k] for k in excl_keys if k in groups.index]

    # Sanity check once per seed: junk here means the MLM head isn't loaded.
    if label not in checked:
        checked.add(label)
        p_idx, w_idx = occ[0]
        words = list(passages[p_idx]["words"])
        s, e = span_of(passages[p_idx], w_idx)
        lo, hi = max(0, s - 8), min(len(words), e + 9)
        shown = words[lo:s] + ["[MASK]"] + words[e + 1:hi]
        logger.info(f"  check: {' '.join(shown)}")
        logger.info(f"         -> {', '.join(top_tokens(macberth, passages, occ[0]))}")

    # Seed side (computed once).
    M, n, vocab_sum = grouped_probs(macberth, passages, occ, groups,
                                    args.batch_size, keep_rows=True)
    if M is None:
        logger.info("  No occurrences could be masked.")
        return
    vocab_mean = vocab_sum / n

    # Baseline replicates.
    bases, base_lists = [], []
    for r in range(R):
        bp, pos = baseline_for(r, excl_keys)
        total, nb, _ = grouped_probs(macberth, bp, pos, groups,
                                     args.batch_size, keep_rows=False)
        if total is None:
            continue
        bases.append(total / nb)
        base_lists.append(bp)
    if not bases:
        logger.info("  No usable baseline positions.")
        return
    B = np.stack(bases)
    R = len(bases)

    fam = M.mean(0)[excl_groups].sum() if excl_groups else 0.0
    logger.info(f"  baselines: {R} replicates; seed family "
          f"({', '.join(sorted(excl_keys))}) takes p={fam:.3f} of the mass "
          f"and is excluded below.")

    # ---- substitutes -----------------------------------------------------
    subs, ov_s = score_substitutes(M, B, excl_groups, args, nprng)
    logger.info(f"\n  SUBSTITUTES (alternatives in the seed's slot; variants pooled)")
    if ov_s is not None:
        logger.info(f"  stability: mean pairwise top-{args.stability_k} overlap "
              f"across baselines = {ov_s:.2f}")
    logger.info(f"  {'filler':<26}{'mean_p':>8}{'base_p':>9}{'score':>8}  "
          f"{'95% CI':<15}{'support':>9}{'top%':>6}{'stab':>6}")
    for rank, it in enumerate(subs, start=1):
        g = it["group"]
        ms = sorted(groups.members[g], key=lambda i: -vocab_mean[i])
        top = vocab_mean[ms[0]]
        name = "/".join(tok.convert_ids_to_tokens(int(i)) for i in ms[:3]
                        if vocab_mean[i] >= 0.1 * top)
        rows.append({
            "kind": "substitute", "seed": label, "period": period,
            "scope": scope_label, "n_occurrences": n, "rank": rank,
            "filler": name, "mean_prob": f"{it['mean_p']:.5f}",
            "baseline_prob": f"{it['base_p']:.6f}",
            "lift": f"{(it['mean_p'] + EPS) / (it['base_p'] + EPS):.2f}",
            "score": f"{it['score']:.5f}",
            "ci_lo": f"{it['ci_lo']:.5f}", "ci_hi": f"{it['ci_hi']:.5f}",
            "support": it["support"], "top_share": f"{it['top_share']:.3f}",
            "stability": f"{it['stability']:.2f}",
        })
        if rank <= args.top:
            ci = f"[{it['ci_lo']:.3f},{it['ci_hi']:.3f}]"
            logger.info(f"  {name:<26}{it['mean_p']:>8.4f}{it['base_p']:>9.5f}"
                  f"{it['score']:>8.4f}  {ci:<15}{it['support']:>5}/{n:<3}"
                  f"{it['top_share']:>6.0%}{round(it['stability'] * R):>4}/{R}")

    # ---- collocates ------------------------------------------------------
    # Required words are over-represented by construction, so leave them out.
    min_c = args.min_collocate_count or max(3, round(0.03 * n))
    col_excl = excl_keys | {stem_key(w) for w in args.require_set}
    cols, ov_c = score_collocates(passages, occ, base_lists, col_excl, min_c,
                                  args, nprng)
    logger.info(f"\n  COLLOCATES (windows containing the word, min {min_c}; "
          f"--require words omitted)")
    if ov_c is not None:
        logger.info(f"  stability: mean pairwise top-{args.stability_k} overlap "
              f"across baselines = {ov_c:.2f}")
    logger.info(f"  {'word':<26}{'windows':>8}{'p_seed':>9}{'p_base':>9}{'score':>8}  "
          f"{'95% CI':<15}{'stab':>6}")
    for rank, it in enumerate(cols, start=1):
        rows.append({
            "kind": "collocate", "seed": label, "period": period,
            "scope": scope_label, "n_occurrences": n, "rank": rank,
            "filler": it["word"], "mean_prob": f"{it['p_seed']:.5f}",
            "baseline_prob": f"{it['p_base']:.6f}",
            "lift": f"{it['p_seed'] / it['p_base']:.2f}",
            "score": f"{it['score']:.5f}",
            "ci_lo": f"{it['ci_lo']:.5f}", "ci_hi": f"{it['ci_hi']:.5f}",
            "support": it["support"], "top_share": "",
            "stability": f"{it['stability']:.2f}",
        })
        if rank <= args.top:
            ci = f"[{it['ci_lo']:.3f},{it['ci_hi']:.3f}]"
            logger.info(f"  {it['word']:<26}{it['support']:>8}{it['p_seed']:>9.3f}"
                  f"{it['p_base']:>9.4f}{it['score']:>8.4f}  {ci:<15}"
                  f"{round(it['stability'] * R):>4}/{R}")


def make_periods(start, end, size):
    if not size:
        return [(start, end)]
    out, y = [], start
    while y <= end:
        out.append((y, min(y + size - 1, end)))
        y += size
    return out


def parse_phrase(spec):
    """
    'take/tooke/taketh heed/heede' ->
        ('take heed', [{'take', 'tooke', 'taketh'}, {'heed', 'heede'}])
    """
    parts = spec.split()
    slots = []
    for part in parts:
        forms = {v.strip().lower() for v in part.split("/") if v.strip()}
        if forms:
            slots.append(forms)
    if len(slots) < 2:
        raise ValueError(f"A phrase needs at least two words: {spec!r}")
    label = " ".join(part.split("/")[0].strip().lower() for part in parts)
    return label, slots


def csv_list(s):
    return {w.strip().lower() for w in s.split(",") if w.strip()}


def main() -> None:
    p = argparse.ArgumentParser(description="Discover vocabulary around seed words.")
    p.add_argument("--seed", action="append", default=[],
                   help="Seed word, or comma-separated spelling variants. Repeatable.")
    p.add_argument("--phrase", action="append", default=[],
                   help="Seed phrase (db only): words separated by spaces, spelling "
                        "variants by '/', e.g. \"take/tooke/taketh heed/heede\". "
                        "Repeatable.")
    p.add_argument("--phrase-gap", type=int, default=0,
                   help="Max other tokens allowed between consecutive phrase words.")
    p.add_argument("--exclude-forms", default="",
                   help="Extra words (e.g. derivations such as 'resistance') to keep "
                        "out of the results. The seed's own inflections and spellings "
                        "are always excluded.")
    p.add_argument("--source", choices=("db", "csv"), default="db")
    p.add_argument("--scope", choices=tuple(SCOPES), default="pamphlet",
                   help="db only: pamphlet_corpus view, or all English documents.")
    p.add_argument("--input", default=OUT_DIR / "passage_vocabulary_benchmark.csv",
                   help="CSV path (--source csv).")
    p.add_argument("--occurrences-csv", default=OUT_DIR / "passage_occurrences.csv",
                help="Write one row per seed occurrence (doc_id, position, window).")
    p.add_argument("--output", default=OUT_DIR / "masked_fillers.csv")
    p.add_argument("--start-year", type=int, default=1600)
    p.add_argument("--end-year", type=int, default=1650)
    p.add_argument("--period-size", type=int, default=0,
                   help="Split the years into periods of this many years (0 = one period).")
    p.add_argument("--match", choices=("token", "canonical"), default="token",
                   help="Match seeds against tokens.token (lowercased) or tokens.canonical.")
    p.add_argument("--require", default="",
                   help="Comma-separated words; keep only occurrences whose window "
                        "contains at least one.")
    p.add_argument("--forbid", default="",
                   help="Comma-separated words; drop occurrences whose window "
                        "contains any (e.g. god,lord,sathan,devil,christ,faith).")
    p.add_argument("--oversample", type=int, default=6,
                   help="With --require/--forbid, fetch this many times "
                        "--max-occurrences before filtering.")
    p.add_argument("--max-occurrences", type=int, default=400,
                   help="Occurrences kept per seed per period.")
    p.add_argument("--max-per-doc", type=int, default=3)
    p.add_argument("--context", type=int, default=40,
                   help="Words of context either side of the seed.")

    p.add_argument("--baseline", choices=("matched", "period"), default="matched",
                   help="matched: baseline windows from the same documents as the "
                        "seed occurrences. period: random windows from the period.")
    p.add_argument("--baseline-per-hit", type=int, default=1,
                   help="matched: baseline windows per seed occurrence.")
    p.add_argument("--baseline-n", type=int, default=500,
                   help="period/csv: baseline windows/positions per replicate.")
    p.add_argument("--replicates", type=int, default=5,
                   help="Number of independent baseline draws.")
    p.add_argument("--bootstrap", type=int, default=300,
                   help="Bootstrap draws for the 95%% intervals.")
    p.add_argument("--stability-k", type=int, default=50,
                   help="Top-K used for the stability measures.")
    p.add_argument("--candidates", type=int, default=300,
                   help="Candidates carried into the uncertainty step.")

    p.add_argument("--min-occurrences", type=int, default=20,
                   help="Skip a seed with fewer occurrences than this (0 = never skip).")
    p.add_argument("--min-support", type=int, default=3,
                   help="A substitute must reach --support-p in at least this many "
                        "occurrences.")
    p.add_argument("--support-p", type=float, default=0.005,
                   help="Per-occurrence probability counted as 'supporting'.")
    p.add_argument("--min-collocate-count", type=int, default=0,
                   help="A collocate must appear in at least this many windows "
                        "(0 = auto: max(3, 3%% of occurrences)).")
    p.add_argument("--keep-uncertain", action="store_true",
                   help="Keep items whose 95%% interval includes 0.")

    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--seed-rng", type=int, default=0)
    args = p.parse_args()

    if args.start_year > args.end_year:
        p.error("--start-year must not exceed --end-year")
    if not args.seed and not args.phrase:
        p.error("give at least one --seed or --phrase")
    if args.phrase and args.source != "db":
        p.error("--phrase needs --source db")
    if args.phrase_gap < 0:
        p.error("--phrase-gap must not be negative")
    if args.replicates < 1:
        p.error("--replicates must be at least 1")

    require = csv_list(args.require)
    forbid = csv_list(args.forbid)
    args.require_set = require
    extra_exclude = csv_list(args.exclude_forms)
    seed_groups = [
        (s.split(",")[0].strip().lower(), csv_list(s), None)
        for s in args.seed
    ]
    for spec in args.phrase:
        try:
            plabel, slots = parse_phrase(spec)
        except ValueError as exc:
            p.error(str(exc))
        seed_groups.append((plabel, set().union(*slots), slots))
    nprng = np.random.default_rng(args.seed_rng)

    logger.info("Loading MacBERTh...")
    macberth = load_macberth()
    logger.info("Building groups...")
    groups = build_groups(macberth.tokenizer, macberth.model.config.vocab_size)

    conn = None
    csv_passages = None
    if args.source == "db":
        from lib.corpus_db import get_connection
        conn = get_connection(application_name="passage-mask-fillers")
        periods = make_periods(args.start_year, args.end_year, args.period_size)
        scope_label = args.scope
    else:
        csv_passages = load_passages(Path(args.input))
        logger.info(f"Loaded {len(csv_passages)} unique passages from {args.input}")
        periods = [(None, None)]
        scope_label = "csv"

    if require:
        logger.info(f"Requiring one of: {', '.join(sorted(require))}")
    if forbid:
        logger.info(f"Forbidding: {', '.join(sorted(forbid))}")

    rows: list[dict] = []
    checked: set[str] = set()
    period_cache: dict = {}

    occ_path = None
    write_header = False
    if args.occurrences_csv:
        occ_path = Path(args.occurrences_csv)
        write_header = not occ_path.exists()
        occ_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        for lo, hi in periods:
            period = f"{lo}-{hi}" if lo is not None else "csv"

            for label, variants, slots in seed_groups:
                doc_len = {}
                if conn is not None:
                    want = args.max_occurrences * (
                        args.oversample if (require or forbid) else 1)
                    if slots:
                        hits, surfaces = fetch_phrase_hits(
                            conn, args.scope, slots, args.match, lo, hi, want,
                            args.max_per_doc, args.phrase_gap,
                        )
                    else:
                        hits, surfaces = fetch_hits(
                            conn, args.scope, variants, args.match, lo, hi, want,
                            args.max_per_doc,
                        )
                    passages, occ = windows_to_passages(
                        fetch_windows(conn, hits, args.context)
                    )
                    exclude = variants | surfaces | extra_exclude
                else:
                    passages = csv_passages
                    occ = find_occurrences(passages, variants)
                    exclude = variants | extra_exclude

                fetched = len(occ)
                occ = filter_context(passages, occ, require, forbid)[:args.max_occurrences]
                if require or forbid:
                    logger.info(f"\n[{label}] {len(occ)} of {fetched} occurrences pass --require/--forbid.")

                if conn is not None and args.baseline == "matched" and occ:
                    seed_hits = [passages[pi]["hit"] for pi, _ in occ]
                    doc_len = fetch_doc_lengths(conn, [d for d, _ in seed_hits])

                def baseline_for(r, excl_keys, *, label=label, lo=lo, hi=hi, passages=passages, occ=occ, doc_len=doc_len):
                    rb = random.Random(f"{args.seed_rng}:{label}:{lo}:{r}")
                    if conn is None:
                        return csv_passages, random_positions(
                            csv_passages, excl_keys, args.baseline_n, rb)
                    if args.baseline == "matched":
                        seed_hits = [passages[pi]["hit"] for pi, _ in occ]
                        hits = matched_hits(seed_hits, doc_len, args.context, rb,
                                            args.baseline_per_hit)
                        bp, _ = windows_to_passages(
                            fetch_windows(conn, hits, args.context))
                    else:
                        ck = (lo, hi, r)
                        if ck not in period_cache:
                            hits = fetch_random_hits(conn, args.scope, lo, hi, args.baseline_n, args.context, rb)
                            period_cache[ck], _ = windows_to_passages( fetch_windows(conn, hits, args.context))
                        bp = period_cache[ck]
                    return bp, one_position_per_passage(bp, excl_keys, rb)

                analyse(macberth, label, period, scope_label, passages, occ, exclude, groups, baseline_for, args, rows, checked, nprng)

                # write occurrences csv for this seed/period
                if occ_path is not None and occ:
                    doc_ids = [passages[p_idx]["doc_id"] for p_idx, _ in occ]
                    meta = fetch_doc_meta(conn, doc_ids) if conn is not None else {}

                    with occ_path.open("a", encoding="utf-8-sig", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=[
                            "seed", "period", "pub_year", "scope",
                            "doc_id", "corpus", "title", "pub_year",
                            "token_idx", "word_start", "window_text",
                        ])
                        if write_header:
                            w.writeheader()
                            write_header = False          # only once

                        for p_idx, w_idx in occ:
                            p = passages[p_idx]
                            m = meta.get(p["doc_id"], {})
                            window = " ".join(p["words"])
                            w.writerow({
                                "seed": label,
                                "period": period,
                                "pub_year": m.get("pub_year"),
                                "scope": scope_label,
                                "doc_id": p["doc_id"],
                                "corpus": m.get("corpus", ""),
                                "title": m.get("title", ""),
                                "pub_year": m.get("pub_year", ""),
                                "token_idx": p["word_start"] + w_idx,
                                "word_start": p["word_start"],
                                "window_text": window,
                            })
                    logger.info(f"  wrote {len(occ)} occurrence rows → {occ_path}")
    finally:
        if conn is not None:
            conn.close()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "kind", "seed", "period", "pub_year", "scope", "n_occurrences", "rank", "filler",
            "mean_prob", "baseline_prob", "lift", "score", "ci_lo", "ci_hi",
            "support", "top_share", "stability",
        ])
        w.writeheader()
        w.writerows(rows)
    logger.info(f"\nSaved {len(rows)} rows to: {out.resolve()}")

    logger.info(f"Done")

if __name__ == "__main__":
    main()
