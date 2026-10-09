"""
tier1/passage_mask_fillers.py

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
    python passage_mask_fillers.py --seed resist,resiste
    python passage_mask_fillers.py --seed resist,resiste --scope all \\
        --start-year 1550 --end-year 1700 --max-per-doc 10 \\
        --forbid god,lord,sathan,devil,christ,faith,temptation,sinne
    python passage_mask_fillers.py --seed libertie,liberty --period-size 25
    python passage_mask_fillers.py --seed resist --exclude-forms resistance
    python passage_mask_fillers.py --source csv --seed defence
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from lib.corpus_config import OUT_DIR
from lib.macberth import load_macberth

MAX_LEN = 512
EPS = 1e-5

# Function words, including Early Modern forms (adapted from
# eebo_word_frequencies.py).
STOP = set("""
a an the and or but nor yet so if unless whether as than at by for from in
into of off on onto out over through to up upon with within without about
above across after against along among around before behind below beneath
beside between beyond down during except inside near outside since toward
towards under underneath until via i me my mine myself we us vs our ours
ourselves you your yours yourself yourselves he him his himself she her hers
herself it its itself they them their theirs themselves this that these those
who whom whose which what whatever whichever when where why how all any
anybody anyone anything both each either everybody everyone everything few
many much neither nobody none no one nothing other others several some
somebody someone something such here there everywhere nowhere somewhere am is
are was were be bee been being have has had having do de doe does did done
doing can cannot could may might must shall should will would ought need dare
not never again already also always ever just merely only quite rather really
very too almost enough especially even still then now once often sometimes
usually because although though while whereas therefore thus hence indeed
perhaps maybe yes thou thee thy thine thyself ye hath hast doth dost art
wilt wouldst shouldst couldst whither whence hither thither hereof thereof
wherein whereof whereby herewith therewith unto amidst amongst betwixt per
mr est ad two three four five six seven eight nine ten haue hee shee wee vpon
vnto vntill whilest yea nay
""".split())


# ---------------------------------------------------------------------------
# Word grouping
# ---------------------------------------------------------------------------

def fold_key(t: str) -> str:
    """Collapse u/v and i/j so 'preuent' and 'prevent' share a key."""
    return t.lower().replace("v", "u").replace("j", "i")


@lru_cache(maxsize=None)
def stem_key(word: str) -> str:
    """
    Light stem for grouping variants of one word. Deliberately conservative:
    a false split costs little (members are listed), a false merge hides
    a word.

      preuent/prevent/prevents/preventing/prevented -> one key
      resist/resiste/resisted/resisteth/resists      -> one key
      libertie/liberty/liberties; kingdome/kingdoms; repell/repelled
    """
    w = fold_key(word)

    # plurals, third person, -ie
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("ie") and len(w) > 4:
        w = w[:-2] + "y"
    elif w.endswith("es") and len(w) > 4:
        w = w[:-2]
    elif w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        w = w[:-1]

    # verb endings
    for suf, minimum in (("ing", 4), ("eth", 3), ("ed", 3)):
        if w.endswith(suf) and len(w) - len(suf) >= minimum:
            w = w[:-len(suf)]
            break

    # final -e
    if w.endswith("e") and len(w) > 3:
        w = w[:-1]

    # doubled final consonant (repell -> repel)
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiou":
        w = w[:-1]

    return w


@dataclass
class Groups:
    group_of: np.ndarray        # vocab id -> group index, -1 if not a candidate
    members: list               # group -> [vocab ids]
    keys: list                  # group -> stem key
    index: dict                 # stem key -> group
    sel: np.ndarray = field(init=False)
    cache: dict = field(default_factory=dict)   # device -> torch tensors

    def __post_init__(self):
        self.sel = self.group_of >= 0

    @property
    def n(self) -> int:
        return len(self.members)


def build_groups(tokenizer, vocab_size: int) -> Groups:
    """Whole alphabetic non-stopword vocabulary entries, grouped by stem."""
    tokens = tokenizer.convert_ids_to_tokens(
        list(range(min(vocab_size, len(tokenizer))))
    )
    group_of = np.full(vocab_size, -1, dtype=np.int64)
    index, members, keys = {}, [], []
    for i, t in enumerate(tokens):
        if t is None or not t.isalpha() or len(t) < 2 or t.lower() in STOP:
            continue
        k = stem_key(t)
        if k not in index:
            index[k] = len(members)
            members.append([])
            keys.append(k)
        group_of[i] = index[k]
        members[index[k]].append(i)
    return Groups(group_of, members, keys, index)


def content_word(w: str, excl_keys: set[str]) -> bool:
    lw = w.lower()
    return lw.isalpha() and len(lw) > 1 and lw not in STOP and stem_key(lw) not in excl_keys


# ---------------------------------------------------------------------------
# CSV source
# ---------------------------------------------------------------------------

def load_passages(path: Path) -> list[dict]:
    """One row per passage_id; passages repeat across queries/representations."""
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
        window = {x.lower() for i, x in enumerate(passages[p]["words"]) if i != w}
        if require and not (window & require):
            continue
        if forbid and (window & forbid):
            continue
        out.append((p, w))
    return out


# ---------------------------------------------------------------------------
# Database source
# ---------------------------------------------------------------------------

# Fixed SQL fragments (never built from user text).
SCOPES = {
    "pamphlet": {
        "table": "pamphlet_corpus",
        "join": "d.corpus = t.corpus AND d.doc_id = t.doc_id",
        "where": "",
    },
    "all": {
        "table": "documents",
        "join": "d.doc_id = t.doc_id",
        "where": "AND (d.lang = 'eng' OR d.lang IS NULL)",
    },
}

# token: uses idx_tokens_token_lower. canonical: uses idx_tokens_canonical, so
# the match is exact and variants are matched as given (lower-cased).
MATCH_SQL = {
    "token": "lower(t.token) = ANY(%(variants)s)",
    "canonical": "t.canonical = ANY(%(variants)s)",
}


def fetch_hits(conn, scope, variants, match, lo, hi, n, per_doc):
    """
    Randomly sample up to n occurrences in the scope's documents for years
    lo..hi, at most per_doc from any one document.

    Returns ([(doc_id, token_idx)], {surface forms seen}).
    """
    sc = SCOPES[scope]
    query = f"""
        WITH hits AS (
            SELECT t.doc_id, t.token_idx, lower(t.token) AS surface,
                   row_number() OVER (
                       PARTITION BY t.doc_id ORDER BY random()
                   ) AS rn
            FROM tokens t
            JOIN {sc['table']} d ON {sc['join']}
            WHERE {MATCH_SQL[match]}
              AND d.pub_year BETWEEN %(lo)s AND %(hi)s
              {sc['where']}
        )
        SELECT doc_id, token_idx, surface
        FROM hits
        WHERE rn <= %(per_doc)s
        ORDER BY random()
        LIMIT %(n)s
    """
    with conn.cursor() as cur:
        cur.execute(query, {
            "variants": sorted(variants), "lo": lo, "hi": hi,
            "per_doc": per_doc, "n": n,
        })
        rows = cur.fetchall()
    return [(r[0], r[1]) for r in rows], {r[2] for r in rows}


def fetch_doc_lengths(conn, doc_ids):
    """doc_id -> token_count, for matched baseline sampling."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT doc_id, token_count FROM documents WHERE doc_id = ANY(%s)",
            (sorted(set(doc_ids)),),
        )
        return {d: tc for d, tc in cur.fetchall() if tc}


def matched_hits(seed_hits, doc_len, context, rng, per_hit):
    """
    For each seed occurrence, pick per_hit baseline centres in the SAME
    document whose windows do not overlap the seed's window.
    """
    out = []
    for doc, idx in seed_hits:
        tc = doc_len.get(doc)
        if not tc or tc <= 2 * context + 1:
            continue
        for _ in range(per_hit):
            for _try in range(30):
                c = rng.randint(context, tc - context - 1)
                if abs(c - idx) > 2 * context:
                    out.append((doc, c))
                    break
    return out


def fetch_random_hits(conn, scope, lo, hi, n, context, rng):
    """
    Random (doc_id, token_idx) centres from the whole period (the
    `--baseline period` mode). Documents are drawn with replacement, weighted
    by length (capped so a few huge texts can't dominate).
    """
    sc = SCOPES[scope]
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT d.doc_id, d.token_count
            FROM {sc['table']} d
            WHERE d.pub_year BETWEEN %s AND %s
              AND d.token_count > %s
              {sc['where']}
            """,
            (lo, hi, 2 * context + 1),
        )
        docs = cur.fetchall()
    if not docs:
        return []
    picked = rng.choices(docs, weights=[min(tc, 50_000) for _, tc in docs], k=n)
    return [(d, rng.randint(context, tc - context - 1)) for d, tc in picked]


WINDOW_SQL = """
    SELECT w.ord, t.token_idx, t.token
    FROM unnest(%s::int[], %s::text[], %s::int[], %s::int[])
         AS w(ord, doc_id, lo, hi)
    JOIN tokens t
      ON t.doc_id = w.doc_id
     AND t.token_idx BETWEEN w.lo AND w.hi
    ORDER BY w.ord, t.token_idx
"""


def fetch_windows(conn, hits, context, batch=200):
    """
    Context windows (context tokens either side) around each hit.
    Returns [(hit, token_idxs, words)] in hit order.
    """
    out = []
    for off in range(0, len(hits), batch):
        chunk = hits[off:off + batch]
        by_ord = defaultdict(list)
        with conn.cursor() as cur:
            cur.execute(WINDOW_SQL, (
                list(range(len(chunk))),
                [h[0] for h in chunk],
                [max(0, h[1] - context) for h in chunk],
                [h[1] + context for h in chunk],
            ))
            for o, idx, tok in cur.fetchall():
                by_ord[o].append((idx, tok))

        for o, hit in enumerate(chunk):
            rows = by_ord.get(o)
            if not rows:
                continue
            idxs = [r[0] for r in rows]
            words = [r[1] if r[1].strip() else "." for r in rows]
            out.append((hit, idxs, words))
    return out


def windows_to_passages(windows):
    """Passages (each remembering its hit) plus the target position in each."""
    passages, occ = [], []
    for (doc_id, tidx), idxs, words in windows:
        try:
            w_idx = idxs.index(tidx)
        except ValueError:
            continue
        passages.append({
            "doc_id": doc_id, "word_start": idxs[0], "words": words,
            "hit": (doc_id, tidx),
        })
        occ.append((len(passages) - 1, w_idx))
    return passages, occ


# ---------------------------------------------------------------------------
# Baseline positions
# ---------------------------------------------------------------------------

def random_positions(passages, excl_keys, n, rng):
    """Random content-word positions pooled over all passages (csv mode)."""
    pool = [
        (p_idx, w_idx)
        for p_idx, p in enumerate(passages)
        for w_idx, w in enumerate(p["words"])
        if content_word(w, excl_keys)
    ]
    rng.shuffle(pool)
    return pool[:n]


def one_position_per_passage(passages, excl_keys, rng):
    """One random content word per baseline window (db modes)."""
    out = []
    for p_idx, p in enumerate(passages):
        cand = [i for i, w in enumerate(p["words"]) if content_word(w, excl_keys)]
        if cand:
            out.append((p_idx, rng.choice(cand)))
    return out


# ---------------------------------------------------------------------------
# Masked prediction
# ---------------------------------------------------------------------------

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
            pieces = [i for i, wid in enumerate(enc.word_ids()) if wid == w_idx]
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

    def bag(words, skip=None, record=False):
        keys = set()
        for i, w in enumerate(words):
            if i == skip:
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

    seed_bags = [bag(passages[p]["words"], w, record=True) for p, w in occ]
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

    print(f"\n=== {label} | {period} | scope={scope_label} ({len(occ)} occurrences) ===")
    if len(occ) < args.min_occurrences:
        print(f"  Skipped: {len(occ)} occurrences < --min-occurrences "
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
        lo, hi = max(0, w_idx - 8), min(len(words), w_idx + 9)
        shown = words[lo:hi]
        shown[w_idx - lo] = "[MASK]"
        print(f"  check: {' '.join(shown)}")
        print(f"         -> {', '.join(top_tokens(macberth, passages, occ[0]))}")

    # Seed side (computed once).
    M, n, vocab_sum = grouped_probs(macberth, passages, occ, groups,
                                    args.batch_size, keep_rows=True)
    if M is None:
        print("  No occurrences could be masked.")
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
        print("  No usable baseline positions.")
        return
    B = np.stack(bases)
    R = len(bases)

    fam = M.mean(0)[excl_groups].sum() if excl_groups else 0.0
    print(f"  baselines: {R} replicates; seed family "
          f"({', '.join(sorted(excl_keys))}) takes p={fam:.3f} of the mass "
          f"and is excluded below.")

    # ---- substitutes -----------------------------------------------------
    subs, ov_s = score_substitutes(M, B, excl_groups, args, nprng)
    print(f"\n  SUBSTITUTES (alternatives in the seed's slot; variants pooled)")
    if ov_s is not None:
        print(f"  stability: mean pairwise top-{args.stability_k} overlap "
              f"across baselines = {ov_s:.2f}")
    print(f"  {'filler':<26}{'mean_p':>8}{'base_p':>9}{'score':>8}  "
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
            print(f"  {name:<26}{it['mean_p']:>8.4f}{it['base_p']:>9.5f}"
                  f"{it['score']:>8.4f}  {ci:<15}{it['support']:>5}/{n:<3}"
                  f"{it['top_share']:>6.0%}{round(it['stability'] * R):>4}/{R}")

    # ---- collocates ------------------------------------------------------
    # Required words are over-represented by construction, so leave them out.
    min_c = args.min_collocate_count or max(3, round(0.03 * n))
    col_excl = excl_keys | {stem_key(w) for w in args.require_set}
    cols, ov_c = score_collocates(passages, occ, base_lists, col_excl, min_c,
                                  args, nprng)
    print(f"\n  COLLOCATES (windows containing the word, min {min_c}; "
          f"--require words omitted)")
    if ov_c is not None:
        print(f"  stability: mean pairwise top-{args.stability_k} overlap "
              f"across baselines = {ov_c:.2f}")
    print(f"  {'word':<26}{'windows':>8}{'p_seed':>9}{'p_base':>9}{'score':>8}  "
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
            print(f"  {it['word']:<26}{it['support']:>8}{it['p_seed']:>9.3f}"
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


def csv_list(s):
    return {w.strip().lower() for w in s.split(",") if w.strip()}


def main() -> None:
    p = argparse.ArgumentParser(description="Discover vocabulary around seed words.")
    p.add_argument("--seed", action="append", required=True,
                   help="Seed word, or comma-separated spelling variants. Repeatable.")
    p.add_argument("--exclude-forms", default="",
                   help="Extra words (e.g. derivations such as 'resistance') to keep "
                        "out of the results. The seed's own inflections and spellings "
                        "are always excluded.")
    p.add_argument("--source", choices=("db", "csv"), default="db")
    p.add_argument("--scope", choices=tuple(SCOPES), default="pamphlet",
                   help="db only: pamphlet_corpus view, or all English documents.")
    p.add_argument("--input", default=OUT_DIR / "passage_vocabulary_benchmark.csv",
                   help="CSV path (--source csv).")
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
    if args.replicates < 1:
        p.error("--replicates must be at least 1")

    require = csv_list(args.require)
    forbid = csv_list(args.forbid)
    args.require_set = require
    extra_exclude = csv_list(args.exclude_forms)
    seed_groups = [
        (s.split(",")[0].strip().lower(), csv_list(s))
        for s in args.seed
    ]
    nprng = np.random.default_rng(args.seed_rng)

    print("Loading MacBERTh...")
    macberth = load_macberth()
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
        print(f"Loaded {len(csv_passages)} unique passages from {args.input}")
        periods = [(None, None)]
        scope_label = "csv"

    if require:
        print(f"Requiring one of: {', '.join(sorted(require))}")
    if forbid:
        print(f"Forbidding: {', '.join(sorted(forbid))}")

    rows: list[dict] = []
    checked: set[str] = set()
    period_cache: dict = {}

    try:
        for lo, hi in periods:
            period = f"{lo}-{hi}" if lo is not None else "csv"

            for label, variants in seed_groups:
                doc_len = {}
                if conn is not None:
                    want = args.max_occurrences * (
                        args.oversample if (require or forbid) else 1)
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
                    print(f"\n[{label}] {len(occ)} of {fetched} occurrences "
                          f"pass --require/--forbid.")

                if conn is not None and args.baseline == "matched" and occ:
                    seed_hits = [passages[pi]["hit"] for pi, _ in occ]
                    doc_len = fetch_doc_lengths(conn, [d for d, _ in seed_hits])

                def baseline_for(r, excl_keys, *, label=label, lo=lo, hi=hi,
                                 passages=passages, occ=occ, doc_len=doc_len):
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
                            hits = fetch_random_hits(conn, args.scope, lo, hi,
                                                     args.baseline_n, args.context, rb)
                            period_cache[ck], _ = windows_to_passages(
                                fetch_windows(conn, hits, args.context))
                        bp = period_cache[ck]
                    return bp, one_position_per_passage(bp, excl_keys, rb)

                analyse(macberth, label, period, scope_label, passages, occ,
                        exclude, groups, baseline_for, args, rows, checked, nprng)
    finally:
        if conn is not None:
            conn.close()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "kind", "seed", "period", "scope", "n_occurrences", "rank", "filler",
            "mean_prob", "baseline_prob", "lift", "score", "ci_lo", "ci_hi",
            "support", "top_share", "stability",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows to: {out.resolve()}")


if __name__ == "__main__":
    main()
