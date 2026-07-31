#!/usr/bin/env python
"""
tier1_4_substitute_vectors.py - Tier 1.4: Unsupervised Substitute-Vector Construction

Construct a type-level "substitutability" profile for every concept-set
target word in the EEBO corpus, entirely unsupervised.

Motivation
----------

Tier 1 can pool *hidden states* at masked positions, producing contextual
embeddings that are excellent at grouping orthographic variants of the
same lexeme (e.g. "liberty" / "libertie") — those occupy near-identical
contexts, so their pooled embeddings land close together almost by
construction. They are much weaker at surfacing genuine near-synonyms
with divergent register or collocational habits (e.g. "liberty" /
"freedome"), because two distributionally-different words can still
refer to compatible ideas without their *pooled context clouds*
overlapping much.

This tier asks a more direct question instead: "if this word were
erased, how plausible does the model think each vocabulary item is in
its place?" For every masked occurrence of a target word, we don't
pool the encoder's hidden state — we pull the **MLM head's logits** at
the masked position, softmax them, and keep the top-k vocabulary items
as a sparse probability distribution. Summing (and finally averaging)
that distribution across every occurrence of a wordform gives a
type-level "substitute profile": a fingerprint of everything the model
considers interchangeable with that word, in the contexts it actually
appears in.

Two words with heavily overlapping substitute profiles are
distributional substitutes for one another — candidates for synonymy —
regardless of spelling. This is a stronger, more direct probe of
substitutability than embedding-space nearest neighbours, though the
two signals are complementary (see notes below).

Target-word scope
------------------

Tier 1.4 masks the *same* restricted set of target words as Tier 1:
only tokens whose normalized form appears in one of ``CONCEPT_SETS``'
``forms`` lists (via ``is_mask_target``), not every content word in the
corpus. This keeps the two tiers directly comparable — a substitute
profile only gets built for a word if Tier 1 also built a pooled
contextual embedding for it, so the "agreement between Tier 1 and Tier
1.4" cross-check described below is always comparing profiles that
exist on both sides. It also keeps the accumulator's memory footprint
bounded by the (small, curated) concept vocabulary rather than by every
distinct content wordform in the corpus.

Architecture
------------

Input:

    PostgreSQL pamphlet_tokens
        ↓
    document buffering                         (shared with Tier 1;
                                                  carries pub_year)
        ↓
    token filtering (CONCEPT_SETS target
    words only, via is_mask_target)             (shared with Tier 1)
        ↓
    window selection (best-centered)            (reused from Tier 1's
                                                   EmbeddingPipeline)
        ↓
    masked forward pass → MLM head logits        ( logits, not hidden states)
        ↓
    top-k softmax → per-occurrence substitute
    distribution
        ↓
    type-level aggregation, BOTH:
      (a) corpus-wide (summed across every
          occurrence of a wordform, any year)
      (b) per-year        (summed across every
          occurrence of a wordform IN THAT YEAR)
        ↓
    Tier 1.4 substitute-profile store
    (checkpointed accumulator + exportable
    sparse matrix + per-year profiles for
    downstream querying)

This is a sibling store to Tier 1, not a replacement — it shares the
masking/windowing infrastructure and target-word scope but produces a
fundamentally different artifact (sparse, vocab-indexed, type-level)
rather than dense, instance-level contextual embeddings.

Time resolution
----------------

The corpus this tier currently targets is scoped tightly around the
English Civil War (1625–1689, ~65 years), so bucketing is done at
**per-year** granularity rather than into wider fixed bins — the
transitions of interest (Personal Rule, Civil Wars, Interregnum,
Restoration) happen within a handful of years of each other and would
be blurred by coarser binning. Expect real, honest sparsity: many
(wordform, year) cells will have few or zero occurrences, especially
for rarer concept words in thin publishing years. This store does not
try to hide that by interpolating or smoothing at write time — the
per-year accumulator (``agg_by_year`` / ``year_counts``) stores exactly
what was observed, nothing more. Smoothing (pooling a window of
adjacent years) is instead applied on demand at query time via
``SubstituteVectorStore.pooled_profile``, so callers can choose their
own reliability/resolution tradeoff per word rather than have one
fixed bucket width baked into the stored data. ``--year-min``/
``--year-max`` scope the extraction query itself (default 1625/1689);
the corpus-wide profile (``agg``/``type_counts``, unchanged from
before) still pools across the *entire* processed range and remains
the profile ``export_sparse_matrix`` and the Tier 1 cross-comparison
use by default.

Known limitations / follow-up work
-----------------------------------

* **Restricted to CONCEPT_SETS target words.** Only wordforms listed in
  ``CONCEPT_SETS`` are masked and profiled, mirroring Tier 1. Broadening
  this to every content word (the corpus-wide, open-vocabulary version
  of this tier) would mean swapping the ``is_mask_target`` filter below
  back for ``is_content_token`` — but note that materially increases
  both the number of masked forward passes per document and the size of
  the in-memory accumulator (bounded by distinct content wordforms
  rather than by the concept vocabulary), so it isn't a drop-in change
  without re-checking memory/runtime budgets.
* **Per-year cells are frequently sparse by design.** A single stray
  occurrence in a thin year will produce a profile that looks exactly
  as "confident" as a well-attested year unless the caller checks the
  accompanying count. Always carry ``year_counts`` alongside any
  per-year divergence figure reported downstream, and prefer
  ``pooled_profile(..., window>0)`` over raw single-year lookups for
  words that don't clear a reasonable per-year occurrence floor.
* **Documents with a NULL/unknown pub_year are still included** in the
  corpus-wide profile (``agg``/``type_counts``) but contribute nothing
  to the per-year accumulator, since there's no year to key on.
* **Multi-subword target words are skipped** (``--allow-multi-subword``
  is not yet implemented). A target word split into multiple WordPiece
  pieces has a joint probability that isn't simply the average of each
  piece's marginal distribution (each piece's distribution is over a
  different, position-conditioned sub-vocabulary). Reconstructing a
  proper joint substitute distribution for multi-piece words needs a
  small beam search over the masked span; left as future work rather
  than approximated silently here.
* **Frequency matters.** Rare wordforms get noisy, low-count profiles.
  Use ``--min-count`` when exporting/querying to filter these out.
* **A word's own vocabulary id will usually dominate its own profile**
  (the model correctly guesses "liberty" belongs where "liberty" was).
  This is expected and harmless for *pairwise* comparisons between two
  different types, but means the raw top-1 substitute for a word is
  rarely informative on its own — the interesting signal is further
  down the ranked list, or in how much *overlap* two different words'
  full profiles have.
* **Substitute-vector and pooled-embedding similarity are
  complementary, not redundant.** Substitute vectors can occasionally
  surface syntactically-compatible-but-semantically-empty pairs.
  Treating agreement between Tier 1 (pooled embeddings) and Tier 1.4
  (substitute profiles) as a confidence signal is recommended before
  trusting a pair as genuinely cognate/substitutable.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from lib.macberth import load_macberth
from lib.corpus_db import get_connection
from lib.corpus_logging import logger
from lib.eebo_config import ZARR_PATH, EMBED_BATCH_SIZE, CONCEPT_SETS

# Reuse Tier 1's window-selection machinery and shared filtering logic
# directly, rather than re-implementing it, so both tiers are
# guaranteed to agree on what "the window containing this token" means
# and on what counts as a maskable target word.
from tier1_0_corpus2zarr import (
    WINDOW_CONFIGS,
    is_mask_target,
    EmbeddingPipeline,
)

from lib.DocBuffer import DocBuffer


# Where the Tier 1.4 store lives. This should really be its own entry in
# lib.eebo_config alongside ZARR_PATH — using a sibling directory here as
# a placeholder so this script runs standalone; swap for a proper config
# constant once you've decided on final layout.
TIER1_4_PATH = ZARR_PATH.parent / "tier1_4_substitutes"


class SubstitutePipeline:
    """
    Builds masked jobs and runs the MLM head to produce per-occurrence
    substitute distributions, then aggregates them per document.

    Mirrors the structure of Tier 1's ``EmbeddingPipeline`` but pulls
    ``logits`` at the mask position instead of pooling hidden states,
    and — like Tier 1 — only targets tokens matching ``is_mask_target``
    (i.e. words listed in ``CONCEPT_SETS``), not every content word.
    """

    def __init__(self, tokenizer, model, device, configs=None,
                 top_k: int = 50, batch_size: int = EMBED_BATCH_SIZE,
                 single_subword_only: bool = True):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        # Default to a single scale (medium) for cost; pass multiple
        # configs to accumulate independent context "votes" per
        # occurrence, echoing Tier 1's multi-scale philosophy.
        self.configs = configs or [c for c in WINDOW_CONFIGS if c["name"] == "medium"]
        self.top_k = top_k
        self.batch_size = batch_size
        self.single_subword_only = single_subword_only

    def _encode(self, tokens):
        enc = self.tokenizer(tokens, is_split_into_words=True, truncation=False, return_tensors="pt")
        word_ids = enc.word_ids() or [None] * len(enc["input_ids"][0])
        return enc["input_ids"][0].tolist(), enc["attention_mask"][0].tolist(), word_ids

    def build_jobs(self, buf: DocBuffer) -> list[dict]:
        input_ids, _attention_mask, word_ids = self._encode(buf.tokens)

        word_id_positions: dict[int, list[int]] = defaultdict(list)
        for i, wid in enumerate(word_ids):
            if wid is not None and wid >= 0:
                word_id_positions[wid].append(i)

        jobs = []
        for config in self.configs:
            windows = EmbeddingPipeline._compute_windows(word_ids, config["size"], config["stride"])
            if not windows:
                continue

            for bpos, _corpus_token_idx in enumerate(buf.corpus_token_idxs):
                token = buf.tokens[bpos]

                # Only mask the same restricted set of target words as
                # Tier 1 (CONCEPT_SETS forms), not every content word.
                if not is_mask_target(token):
                    continue

                abs_positions = word_id_positions.get(bpos, [])
                if not abs_positions:
                    continue
                if self.single_subword_only and len(abs_positions) != 1:
                    # Multi-piece target word — skipped for now, see
                    # module docstring "Known limitations".
                    continue

                window = EmbeddingPipeline._best_window_for_token(windows, bpos)
                if window is None:
                    continue

                encoded_start, encoded_end = window["encoded_start"], window["encoded_end"]
                mask_positions = [
                    p - encoded_start for p in abs_positions
                    if encoded_start <= p < encoded_end
                ]
                if not mask_positions:
                    continue

                window_ids = input_ids[encoded_start:encoded_end]
                masked_ids = list(window_ids)
                for pos in mask_positions:
                    masked_ids[pos] = self.tokenizer.mask_token_id

                jobs.append({
                    "input_ids": masked_ids,
                    "attention_mask": [1] * len(masked_ids),
                    "mask_position": mask_positions[0],  # single-subword -> exactly one
                    "wordform": token.strip().lower(),
                })
        return jobs

    def run(self, buf: DocBuffer):
        """Returns (doc_agg, doc_counts) aggregated across occurrences in this document."""
        jobs = self.build_jobs(buf)
        if not jobs:
            return {}, {}

        doc_agg: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        doc_counts: dict[str, int] = defaultdict(int)

        for i in range(0, len(jobs), self.batch_size):
            chunk = jobs[i:i + self.batch_size]
            self._forward_and_accumulate(chunk, doc_agg, doc_counts)

        return doc_agg, doc_counts

    def _forward_and_accumulate(self, jobs: list[dict], doc_agg, doc_counts):
        max_len = max(len(j["input_ids"]) for j in jobs)

        def pad(seq, pad_value=0):
            return seq + [pad_value] * (max_len - len(seq))

        input_ids_t = torch.tensor([pad(j["input_ids"]) for j in jobs], dtype=torch.long).to(self.device)
        attn_mask_t = torch.tensor([pad(j["attention_mask"]) for j in jobs], dtype=torch.long).to(self.device)

        with torch.inference_mode():
            out = self.model(input_ids=input_ids_t, attention_mask=attn_mask_t, return_dict=True)

        # out.logits: (batch, seq_len, vocab_size). We only ever need the
        # distribution at each job's own mask position, so index per-job
        # rather than trying to vectorize across a shared "pos" (jobs in
        # a batch can have different mask positions).
        for b, job in enumerate(jobs):
            pos = job["mask_position"]
            logits = out.logits[b, pos]
            k = min(self.top_k, logits.shape[-1])
            top_logits, top_ids = torch.topk(logits, k)
            top_probs = torch.softmax(top_logits, dim=-1)

            wordform = job["wordform"]
            doc_counts[wordform] += 1
            target = doc_agg[wordform]
            for vid, prob in zip(top_ids.tolist(), top_probs.tolist()):
                target[vid] += prob


class SubstituteVectorStore:
    """
    Type-level substitute-profile accumulator.

    Unlike Tier 1's Zarr store (one row per *instance*), this store is
    aggregated at the *type* level: one accumulated distribution per
    distinct wordform, summed across every masked occurrence seen. State
    is bounded by the concept vocabulary, not corpus size, so the
    accumulator is kept fully in memory and checkpointed to disk
    periodically as a pickle (cheap to (de)serialize; avoids repeated
    sparse-matrix conversion during the run). ``export_sparse_matrix``
    performs the one-time conversion into a queryable scipy CSR matrix
    once extraction is complete (or at any checkpoint, if you want an
    interim queryable snapshot).

    Two parallel accumulators are kept:

    * ``agg`` / ``type_counts`` — corpus-wide, exactly as before. Every
      occurrence of a wordform is summed here regardless of year (or
      even if pub_year is unknown/NULL).
    * ``agg_by_year`` / ``year_counts`` — keyed by ``(wordform, year)``,
      summing only the occurrences that fell in that specific year.
      Deliberately *not* pre-smoothed or bucketed wider than a single
      year; see module docstring "Time resolution". Use
      ``pooled_profile`` at query time to pool a window of years for
      words that are too sparse at single-year resolution.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.path / "accumulator.pkl"
        self.processed_docs_path = self.path / "processed_docs.json"

        self.agg: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.type_counts: dict[str, int] = defaultdict(int)

        # Per-year accumulator. Keys are (wordform, year) tuples; values
        # mirror the shape of `agg`/`type_counts` but scoped to one year.
        self.agg_by_year: dict[tuple[str, int], dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.year_counts: dict[tuple[str, int], int] = defaultdict(int)

        self.processed_docs: set[str] = set()

        self._load()

    def _load(self):
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, "rb") as f:
                state = pickle.load(f)
            self.agg = defaultdict(lambda: defaultdict(float), state["agg"])
            self.type_counts = defaultdict(int, state["type_counts"])
            # .get(...) with a default so checkpoints written before the
            # per-year accumulator existed still load cleanly (they just
            # start with an empty per-year store, which will backfill as
            # new documents are processed — old docs stay marked
            # processed and won't be re-scanned, so a --clear + full
            # rerun is needed to backfill per-year data for docs already
            # ingested under the old, year-less checkpoint format).
            self.agg_by_year = defaultdict(
                lambda: defaultdict(float), state.get("agg_by_year", {})
            )
            self.year_counts = defaultdict(int, state.get("year_counts", {}))
            logger.info(f"Resumed Tier 1.4 accumulator: {len(self.agg):,} types, "
                        f"{len(self.agg_by_year):,} (type, year) cells")
        if self.processed_docs_path.exists():
            with open(self.processed_docs_path) as f:
                self.processed_docs = set(json.load(f))
            logger.info(f"Resumed Tier 1.4 progress: {len(self.processed_docs):,} docs already processed")

    def get_doc_ids(self) -> set[str]:
        return self.processed_docs

    def merge_doc(self, doc_id: str, doc_agg: dict, doc_counts: dict, pub_year: int | None = None):
        for wordform, dist in doc_agg.items():
            target = self.agg[wordform]
            for vid, prob in dist.items():
                target[vid] += prob
        for wordform, c in doc_counts.items():
            self.type_counts[wordform] += c

        if pub_year is not None:
            for wordform, dist in doc_agg.items():
                key = (wordform, pub_year)
                target = self.agg_by_year[key]
                for vid, prob in dist.items():
                    target[vid] += prob
            for wordform, c in doc_counts.items():
                self.year_counts[(wordform, pub_year)] += c

        self.processed_docs.add(doc_id)

    def pooled_profile(self, wordform: str, center_year: int, window: int = 0):
        """
        Sum this word's per-year substitute distributions across
        [center_year - window, center_year + window] (inclusive) and
        return (normalized_dist, total_count).

        window=0 gives exact single-year resolution. Widen window to
        trade time-resolution for reliability on sparse words. Caller
        decides the tradeoff explicitly per word/year rather than this
        store picking one bucket width for everything — see
        `pooled_profile_auto` for an auto-widening convenience wrapper.

        Returns ({}, 0) if nothing was observed in the requested range.
        """
        total: dict[int, float] = defaultdict(float)
        count = 0
        for y in range(center_year - window, center_year + window + 1):
            key = (wordform, y)
            c = self.year_counts.get(key, 0)
            if not c:
                continue
            count += c
            for vid, p in self.agg_by_year[key].items():
                total[vid] += p

        if count == 0:
            return {}, 0

        return {vid: p / count for vid, p in total.items()}, count

    def pooled_profile_auto(self, wordform: str, center_year: int,
                             min_count: int = 10, max_window: int = 10):
        """
        Convenience wrapper around `pooled_profile`: starts at window=0
        (exact year) and widens by 1 year on each side until either
        `min_count` occurrences are pooled or `max_window` is hit,
        whichever comes first. Returns (normalized_dist, total_count,
        window_used) so callers/plots can report how much smoothing was
        actually needed for a given word/year.
        """
        for window in range(0, max_window + 1):
            dist, count = self.pooled_profile(wordform, center_year, window)
            if count >= min_count:
                return dist, count, window
        # Return whatever we got at the widest window tried, even if it
        # never reached min_count — still useful, just flagged by the
        # caller via the returned count/window as low-confidence.
        return dist, count, max_window

    def checkpoint(self):
        tmp = self.checkpoint_path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(
                {
                    "agg": dict(self.agg),
                    "type_counts": dict(self.type_counts),
                    "agg_by_year": dict(self.agg_by_year),
                    "year_counts": dict(self.year_counts),
                },
                f, protocol=pickle.HIGHEST_PROTOCOL
            )
        tmp.replace(self.checkpoint_path)

        tmp2 = self.processed_docs_path.with_suffix(".json.tmp")
        with open(tmp2, "w") as f:
            json.dump(sorted(self.processed_docs), f)
        tmp2.replace(self.processed_docs_path)

        logger.info(f"Checkpointed Tier 1.4: {len(self.agg):,} types, "
                    f"{len(self.agg_by_year):,} (type, year) cells, "
                    f"{len(self.processed_docs):,} docs")

    def export_sparse_matrix(self, vocab_size: int, min_count: int = 5):
        """
        Build a queryable scipy CSR matrix (n_types x vocab_size), each
        row normalized to a mean substitute-probability distribution,
        plus a type index and per-type occurrence counts.
        """
        types = sorted(w for w, c in self.type_counts.items() if c >= min_count)
        type_to_row = {w: i for i, w in enumerate(types)}

        rows, cols, data = [], [], []
        for w in types:
            row = type_to_row[w]
            dist = self.agg[w]
            count = self.type_counts[w]
            for vid, prob_sum in dist.items():
                rows.append(row)
                cols.append(vid)
                data.append(prob_sum / count)  # mean substitute probability

        matrix = sp.csr_matrix((data, (rows, cols)), shape=(len(types), vocab_size))

        sp.save_npz(self.path / "substitute_matrix.npz", matrix)
        with open(self.path / "types.json", "w") as f:
            json.dump(types, f)
        with open(self.path / "type_counts.json", "w") as f:
            json.dump({w: self.type_counts[w] for w in types}, f)

        logger.info(f"Exported Tier 1.4 substitute matrix: {matrix.shape[0]:,} types "
                    f"(min_count={min_count}) x {matrix.shape[1]:,} vocab")
        return matrix, types

    def export_year_table(self, year_min: int, year_max: int, min_count: int = 5):
        """
        Export a dense (word x year) occurrence-count table covering
        [year_min, year_max] inclusive, as both JSON and CSV. This is
        primarily a *sparsity overview* — it makes it easy to see, at a
        glance or via a heatmap, which (word, year) cells have enough
        data to trust before computing per-year divergence figures, and
        which are gaps that need `pooled_profile`/`pooled_profile_auto`.

        Only words that clear `min_count` in their corpus-wide total
        (same filter as `export_sparse_matrix`) are included as rows,
        so the two exports stay aligned on the same word set.
        """
        words = sorted(w for w, c in self.type_counts.items() if c >= min_count)
        years = list(range(year_min, year_max + 1))

        table = {
            w: {y: self.year_counts.get((w, y), 0) for y in years}
            for w in words
        }

        with open(self.path / "year_counts.json", "w") as f:
            json.dump(table, f)

        import csv
        with open(self.path / "year_counts.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["wordform"] + [str(y) for y in years])
            for w in words:
                writer.writerow([w] + [table[w][y] for y in years])

        n_cells = len(words) * len(years)
        n_nonzero = sum(1 for w in words for y in years if table[w][y] > 0)
        logger.info(f"Exported Tier 1.4 year-count table: {len(words):,} words x "
                    f"{len(years)} years ({n_nonzero:,}/{n_cells:,} cells non-empty, "
                    f"{100 * n_nonzero / n_cells:.1f}% coverage)")
        return table

    def export_vocab_strings(self, tokenizer):
        """
        Export the tokenizer vocabulary as a plain list of surface
        strings, indexed by vocab id, so substitute-matrix columns and
        raw profile dicts (agg / agg_by_year) can be read as actual
        English tokens downstream (e.g. by tier1_4_report.py) without
        needing MacBERTh loaded. This is also the natural anchor point
        for any future cross-model comparison (e.g. against a modern
        BERT) — both sides need their vocab ids resolved to strings
        before a JS-divergence or overlap comparison across two
        different tokenizers means anything.
        """
        vocab = tokenizer.convert_ids_to_tokens(range(tokenizer.vocab_size))
        with open(self.path / "vocab_strings.json", "w") as f:
            json.dump(vocab, f)
        logger.info(f"Exported Tier 1.4 vocab strings: {len(vocab):,} tokens")
        return vocab


class SubstituteCorpusProcessor:
    def __init__(self, conn, pipeline: SubstitutePipeline, store: SubstituteVectorStore,
                 report_every: int = 100, checkpoint_every: int = 2000,
                 year_min: int | None = None, year_max: int | None = None):
        self.conn = conn
        self.pipeline = pipeline
        self.store = store
        self.report_every = report_every
        self.checkpoint_every = checkpoint_every
        # Scopes the (non --doc-id) extraction query to this range and is
        # used to key the per-year accumulator. A single --doc-id run
        # bypasses this filter (an explicit single-doc request should
        # always run, whatever year it happens to be) but still buckets
        # that doc's own pub_year into agg_by_year normally.
        self.year_min = year_min
        self.year_max = year_max

    def process(self, doc_id: str | None = None, limit: int | None = None):
        """
        doc_id: process exactly one specific document (by id).
        limit:  process at most this many *new* documents (first N
                unprocessed docs encountered in query order), regardless
                of how many total docs the query would otherwise return.
                Useful for smoke-testing on a handful of documents before
                committing to a full corpus run.
        """
        already_processed = self.store.get_doc_ids()

        cur = self.conn.cursor(name="tier1_4_cursor")
        cur.itersize = 10000

        if doc_id:
            # Explicit single-doc request: run regardless of year range.
            cur.execute(
                "SELECT doc_id, token_idx, vector_id, token, pub_year FROM pamphlet_tokens "
                "WHERE doc_id = %s ORDER BY token_idx", (doc_id,)
            )
        else:
            clauses = []
            params: list = []
            if self.year_min is not None:
                clauses.append("pub_year >= %s")
                params.append(self.year_min)
            if self.year_max is not None:
                clauses.append("pub_year <= %s")
                params.append(self.year_max)
            where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
            cur.execute(
                f"SELECT doc_id, token_idx, vector_id, token, pub_year FROM pamphlet_tokens "
                f"{where}ORDER BY doc_id, token_idx",
                params,
            )

        buf = None
        docs_processed = 0

        for row_doc_id, token_idx, vid, token, pub_year in cur:
            if row_doc_id in already_processed:
                continue

            if buf is None or row_doc_id != buf.doc_id:
                if buf:
                    self._flush(buf)
                    docs_processed += 1
                    if docs_processed % self.report_every == 0:
                        logger.info(f"Processed {docs_processed} documents")
                    if docs_processed % self.checkpoint_every == 0:
                        self.store.checkpoint()
                    if limit is not None and docs_processed >= limit:
                        logger.info(f"Reached --limit={limit}, stopping")
                        self.store.checkpoint()
                        cur.close()
                        return

                buf = DocBuffer(doc_id=row_doc_id, pub_year=pub_year)

            # Only buffer tokens that are actually maskable target words
            # (CONCEPT_SETS forms), mirroring Tier 1's target-word scope.
            if is_mask_target(token):
                buf.append(token, vid, token_idx)

        if buf and buf.doc_id not in already_processed:
            self._flush(buf)

        self.store.checkpoint()
        logger.info(f"Finished: {docs_processed} new documents processed this run")

    def _flush(self, buf: DocBuffer):
        doc_agg, doc_counts = self.pipeline.run(buf)
        if doc_agg:
            self.store.merge_doc(buf.doc_id, doc_agg, doc_counts, pub_year=buf.pub_year)
        else:
            # Still record the doc as processed even if it produced no
            # eligible (single-subword, concept-target) jobs, so resumes
            # don't keep re-scanning empty documents.
            self.store.processed_docs.add(buf.doc_id)


def query_nearest_substitutes(matrix, types, query_word: str, top_n: int = 15):
    """
    Small convenience query: rank all types by similarity of their
    substitute profile to ``query_word``'s, using Jensen-Shannon
    divergence (lower = more similar) over the dense row vectors.
    Rows are already small (vocab_size-length) so densifying one row
    at a time is cheap.
    """
    from scipy.spatial.distance import jensenshannon

    if query_word not in types:
        raise ValueError(f"{query_word!r} not found in exported types "
                          f"(check spelling, or it may be below --min-count, "
                          f"or it isn't in CONCEPT_SETS at all)")

    idx = types.index(query_word)
    query_vec = np.asarray(matrix[idx].todense()).ravel()
    query_vec = query_vec / (query_vec.sum() + 1e-12)

    scores = []
    for i, w in enumerate(types):
        if w == query_word:
            continue
        row = np.asarray(matrix[i].todense()).ravel()
        row_sum = row.sum()
        if row_sum == 0:
            continue
        row = row / row_sum
        dist = jensenshannon(query_vec, row)
        if np.isnan(dist):
            continue
        scores.append((w, dist))

    scores.sort(key=lambda x: x[1])
    return scores[:top_n]


def clear_output_dir():
    if TIER1_4_PATH.exists():
        import shutil
        shutil.rmtree(TIER1_4_PATH)
    TIER1_4_PATH.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clear", action="store_true")
    p.add_argument("--doc-id", type=str, default=None,
                    help="Process exactly one specific document by id")
    p.add_argument("--limit", type=int, default=None,
                    help="Process at most N new documents (for smoke-testing on a "
                         "handful of docs before a full corpus run). Ignored if "
                         "--doc-id is set.")
    p.add_argument("--report-every", type=int, default=100)
    p.add_argument("--checkpoint-every", type=int, default=2000,
                    help="Checkpoint the accumulator every N documents")
    p.add_argument("--top-k", type=int, default=50,
                    help="Number of top vocabulary substitutes retained per occurrence")
    p.add_argument("--min-count", type=int, default=5,
                    help="Minimum occurrence count for a type to be exported/queried")
    p.add_argument("--year-min", type=int, default=1625,
                    help="Earliest pub_year to include (extraction query scope + "
                         "per-year accumulator range). Ignored if --doc-id is set.")
    p.add_argument("--year-max", type=int, default=1689,
                    help="Latest pub_year to include (extraction query scope + "
                         "per-year accumulator range). Ignored if --doc-id is set.")
    p.add_argument("--batch-size", type=int, default=EMBED_BATCH_SIZE)
    p.add_argument("--scales", type=str, default="medium",
                    help="Comma-separated window scales to use: local,medium,broad")
    p.add_argument("--export-only", action="store_true",
                    help="Skip extraction; just (re-)export the sparse matrix from the "
                         "existing checkpoint")
    p.add_argument("--query", type=str, default=None,
                    help="After export, print nearest substitutes for this wordform")
    return p.parse_args()


def main():
    args = parse_args()

    if args.clear:
        logger.info("Clearing Tier 1.4 output")
        clear_output_dir()

    n_forms = sum(len(c["forms"]) for c in CONCEPT_SETS.values())
    logger.info(f"Tier 1.4 target vocabulary: {len(CONCEPT_SETS)} concepts, "
                f"{n_forms} surface forms")

    store = SubstituteVectorStore(TIER1_4_PATH)

    scale_names = {s.strip() for s in args.scales.split(",")}
    configs = [c for c in WINDOW_CONFIGS if c["name"] in scale_names]
    if not configs:
        raise ValueError(f"No matching window scales for --scales={args.scales!r}")

    mac = load_macberth()
    tokenizer = mac.tokenizer
    model = mac.model
    device = mac.device

    if not args.export_only:
        conn = get_connection()
        pipeline = SubstitutePipeline(
            tokenizer, model, device,
            configs=configs, top_k=args.top_k, batch_size=args.batch_size,
        )
        proc = SubstituteCorpusProcessor(
            conn, pipeline, store,
            report_every=args.report_every, checkpoint_every=args.checkpoint_every,
            year_min=args.year_min, year_max=args.year_max,
        )
        proc.process(doc_id=args.doc_id, limit=args.limit)
        conn.close()
        logger.info("[Tier 1.4 extraction done]")

    matrix, types = store.export_sparse_matrix(vocab_size=tokenizer.vocab_size, min_count=args.min_count)
    store.export_year_table(args.year_min, args.year_max, min_count=args.min_count)
    store.export_vocab_strings(tokenizer)

    if args.query:
        results = query_nearest_substitutes(matrix, types, args.query.strip().lower())
        print(f"\nNearest substitutes for {args.query!r}:")
        for w, dist in results:
            print(f"  {w:20s}  JS divergence={dist:.4f}")


if __name__ == "__main__":
    main()
