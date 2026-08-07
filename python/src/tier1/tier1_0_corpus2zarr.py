#!/usr/bin/env python
"""
tier1_corpus2zarr.py - Tier 1: Multi-scale Contextual Embedding Construction

Construct the Tier 1 contextual observation store from the EEBO corpus.

Each content token is embedded under multiple contextual windows
(local, medium and broad), producing contextual observations for
downstream retrieval, clustering and semantic change analysis.

The default mode uses standard MacBERTh contextual embeddings, where
the original token remains visible to the model. This mode is intended
for rapid iteration, index construction and general corpus exploration.

An optional masked mode (``--mask``) replaces selected target tokens
with ``[MASK]`` before inference. The hidden state at the masked position
is then used as a context-driven semantic representation, reducing the
contribution of lexical identity. Because masked inference requires a
separate forward pass for each target occurrence, it is substantially
more expensive and is intended for focused semantic analysis rather than
routine rebuilding.

The output is written to the Tier 1 Zarr observation store, where each
row represents a single contextual observation and records:

    * stable event and concept identifiers
    * document and corpus token identifiers
    * aligned multi-scale contextual embeddings
    * window metadata (start position and token position)
    * token text and vector identifier

Architecture
------------

Input:

    PostgreSQL pamphlet_tokens
        to
    document buffering
        to
    token filtering (content words only)
        to
    multi-scale contextual embedding
        to
    aligned observation construction
        to
    Tier 1 Zarr observation store

The resulting store forms the canonical contextual observation layer
used by downstream indexing, neighbourhood search, clustering and
semantic change analysis.

Technical overview
------------------

The pipeline streams tokenised documents from PostgreSQL, filters
non-content tokens, and embeds remaining tokens using MacBERTh under
multiple contextual window configurations.

The unmasked pipeline embeds each context window once and extracts
token-level hidden states for all content tokens in the window.

The masked pipeline is an alternative analysis mode. For each selected
target occurrence, the token is replaced with ``[MASK]`` and MacBERTh is
run separately. The hidden state at the masked position captures the
model's contextual expectation of the missing token. This representation
is useful for semantic substitution and sense analysis, but is not used
for normal Tier 1 rebuilding due to its computational cost.

Window generation is shared conceptually between masked and unmasked
pipelines. Window metadata, token alignment and event identifiers are
kept explicit so that downstream processing can distinguish observations
created under different contextual scales.

Implementation notes
--------------------

The window-selection implementation uses a consistent masked-job
pipeline: ``_build_masked_window_jobs`` builds jobs,
``_run_masked_window_batch`` performs batched inference, and
``_events_from_masked_windows`` reconstructs observation records.

The masked and unmasked pipelines intentionally produce different
observation populations:

    unmasked:
        all content tokens

    masked:
        only configured semantic targets (currently CONCEPT_SETS)

This distinction should be preserved downstream. Masked embeddings are
not a replacement for the general Tier 1 store; they are a specialised
semantic analysis layer.
"""

from __future__ import annotations

import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Documents are large (windows of 256-512 tokens, batched), tokenization is not the bottleneck:
# the model forward pass is. Disabling this costs you essentially nothing in throughput and
# removes a lot of uncontrolled thread requests.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")        # harmless no-op on this build, but keep for portability
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")   # the one that actually matters here

# When/if using openblas rather than mkl
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")


import argparse
import shutil
import unicodedata
from dataclasses import dataclass
from collections import defaultdict
import time

import numpy as np
import torch
import xxhash

from lib.corpus_db import get_connection
from lib.corpus_logging import logger
from lib.corpus_config import ZARR_PATH, MASKED_ZARR_PATH, EMBED_BATCH_SIZE, CONCEPT_SETS
from lib.zarr_embedding_observation_store import ZarrEmbeddingObservationStore
from lib.DocBuffer import DocBuffer
from lib.stopwords_min import STOPWORDS

WINDOW_CONFIGS = [
    {"name": "local",  "size": 256,  "stride": 128},
    {"name": "medium", "size": 512,  "stride": 256},
    {"name": "broad",  "size": 512,  "stride": 384},
]


def stable_hash(key: str) -> np.int64:
    h = xxhash.xxh64(key, seed=0).intdigest()
    return np.int64(h & 0x7FFFFFFFFFFFFFFF)


def is_content_token(token: str) -> bool:
    stripped = token.strip().lower()
    if not stripped or stripped in STOPWORDS:
        return False
    if all(unicodedata.category(c).startswith(("P", "S", "Z")) for c in stripped):
        return False
    return True


def is_mask_target(token: str) -> bool:
    # return is_content_token(token);
    normalised = unicodedata.normalize("NFKC", token).lower()
    return any(
        normalised in concept["forms"]
        for concept in CONCEPT_SETS.values()
    )


@dataclass(slots=True)
class Event:
    event_id: np.int64
    concept_id: np.int64
    doc_id: str
    corpus: str
    corpus_token_idx: int
    window_start: int
    window_start_token_idx: int
    window_token_pos: int
    token: str
    vector_id: int
    vec: np.ndarray
    config_name: str


    @property
    def token_idx(self):
        return self.corpus_token_idx


    @staticmethod
    def make(
        corpus: str,
        doc_id: str,
        corpus_token_idx: int,
        window_start_token_idx: int,
        window_start: int,
        window_token_pos: int,
        token: str,
        vector_id: int,
        vec: np.ndarray,
        config_name: str = "medium"
    ) -> Event:
        # concept_id = stable_hash(f"{doc_id}:{corpus_token_idx}")
        # event_id = stable_hash( f"{doc_id}:{corpus_token_idx}:{window_start_token_idx}:{window_token_pos}:{config_name}" )
        concept_id = stable_hash(
            f"{corpus}:{doc_id}:{corpus_token_idx}"
        )
        event_id = stable_hash(
            f"{corpus}:{doc_id}:{corpus_token_idx}:"
            f"{window_start_token_idx}:{window_token_pos}:{config_name}"
        )
        return Event(
            event_id=event_id, concept_id=concept_id, doc_id=doc_id, corpus=corpus,
            corpus_token_idx=corpus_token_idx, window_start=window_start,
            window_start_token_idx=window_start_token_idx,
            window_token_pos=window_token_pos, token=token,
            vector_id=vector_id, vec=vec.astype(np.float32),
            config_name=config_name
        )


class EmbeddingPipeline:
    def __init__(self,
        mac,
        mask_targets: bool = True,
        pooling_scope: str = "mask_only",
        mask_only_position: bool = True,
        batch_size: int = EMBED_BATCH_SIZE
    ):
        self.macberth = mac
        self.tokenizer = mac.tokenizer
        self.hidden_size = mac.hidden_size
        self.device = mac.device
        self.mask_targets = mask_targets
        self.pooling_scope = pooling_scope
        self.mask_only_position = mask_only_position
        self.batch_size = batch_size
        self.configs = WINDOW_CONFIGS
        self._window_alignment_checked = False


    def embed_doc(self, buf: DocBuffer) -> list[Event]:
        if not self.mask_targets:
            return self._embed_doc_unmasked(buf)
        return self._embed_doc_masked(buf)


    # UNMASKED (the fast path)
    def _embed_doc_unmasked(self, buf: DocBuffer) -> list[Event]:
        input_ids, attention_mask, word_ids = self._encode(buf.tokens)

        # collect window specs across all configs first
        window_specs = []
        for config in self.configs:
            for window_start, ids, mask, wids in self._iter_windows_config(
                input_ids, attention_mask, word_ids, config["size"], config["stride"]
            ):
                window_specs.append({
                    "window_start": window_start,
                    "input_ids": ids,
                    "attention_mask": mask,
                    "word_ids": wids,
                    "config_name": config["name"],
                })

        all_events = []
        for i in range(0, len(window_specs), self.batch_size):
            chunk = window_specs[i:i + self.batch_size]
            hiddens = self._forward_window_batch(chunk)
            for item, hidden in zip(chunk, hiddens):
                all_events.extend(self._extract_events(buf, item, hidden, item["config_name"]))
        return all_events


    def _forward_window_batch(self, chunk):
        max_len = max(len(c["input_ids"]) for c in chunk)
        def pad(seq, value=0):
            return seq + [value] * (max_len - len(seq))

        input_ids_t = torch.tensor([pad(c["input_ids"]) for c in chunk], dtype=torch.long, device=self.device)
        attention_mask_t = torch.tensor([pad(c["attention_mask"]) for c in chunk], dtype=torch.long, device=self.device)

        with torch.inference_mode():
            out = self.macberth.encode(input_ids=input_ids_t, attention_mask=attention_mask_t, return_dict=True)

        hidden = out.last_hidden_state.cpu().numpy()
        # slice back to each window's real (unpadded) length
        return [hidden[j, :len(chunk[j]["input_ids"])] for j in range(len(chunk))]

    # MASKED
    def _embed_doc_masked(self, buf):
        input_ids, attention_mask, word_ids = self._encode(buf.tokens)
        all_events = []

        for config in self.configs:
            jobs = self._build_masked_window_jobs( buf, input_ids, attention_mask, word_ids, config )

            if not self._window_alignment_checked and config["name"] == "local":
                self._check_window_alignment_once(jobs)

            vectors = self._run_masked_window_batch(jobs)
            all_events.extend( self._events_from_masked_windows(buf, jobs, vectors, config["name"]) )

        logger.info("[tier1] Document %s: %d masked observations (target-only=%s)",
                    buf.doc_id, len(all_events), self.mask_only_position)
        return all_events


    def _check_window_alignment_once(self, jobs: list[dict]) -> None:
        """One-time regression guard for the window_token_pos overwrite bug.

        Confirms that a token appearing in multiple overlapping windows gets
        a distinct target_encoded_pos per window, so _flush's grouping key
        (corpus_token_idx, window_token_pos) won't silently collapse
        legitimate multi-window observations into one.

        Runs once per pipeline instance (first doc, 'local' config, where
        overlap is most likely) then disables itself regardless of outcome.
        """
        self._window_alignment_checked = True  # disable after this call, pass or fail

        if not jobs:
            logger.warning("[tier1] Window-alignment check skipped: no jobs in first doc/config")
            return

        positions_by_word = defaultdict(set)
        for job in jobs:
            positions_by_word[job["target_word_id"]].add(job["target_encoded_pos"])

        repeated = {wid: len(pos) for wid, pos in positions_by_word.items() if len(pos) > 1}

        if not repeated:
            logger.warning(
                "[tier1] Window-alignment check: no word appeared in >1 window for this doc "
                "(doc too short to exercise overlap) — check inconclusive, not a failure"
            )
            return

        sample_wid, n_positions = next(iter(repeated.items()))
        logger.info(
            "[tier1] Window-alignment check passed: word_id=%d appears in %d windows with distinct target_encoded_pos values (%s)",
            sample_wid, n_positions, sorted(positions_by_word[sample_wid])
        )

        assert all(n >= 2 for n in repeated.values()), (
            "[tier1] Window-alignment check failed: a word repeated across windows but target_encoded_pos did not vary: ",
            "window_token_pos will collapse distinct window observations in _flush's grouping key."
        )


    def _build_masked_window_jobs(self, buf: DocBuffer, input_ids, attention_mask, word_ids, config):
        """Mask only the target token per job (default)."""
        windows = self._compute_windows(word_ids, config["size"], config["stride"])
        if not windows:
            return []

        jobs = []

        for window in windows:
            start = window["encoded_start"]
            end = window["encoded_end"]

            window_ids = list(input_ids[start:end])
            window_mask = list(attention_mask[start:end])
            window_word_ids = word_ids[start:end]

            # Create one job per target token
            for encoded_pos, word_id in enumerate(window_word_ids):
                if word_id is None or word_id < 0:
                    continue
                if word_id >= len(buf.corpus_token_idxs):
                    continue

                if not is_mask_target(buf.tokens[word_id]):
                    continue

                target_window_ids = list(window_ids)
                target_window_ids[encoded_pos] = self.tokenizer.mask_token_id
                jobs.append({
                    "input_ids": target_window_ids,
                    "attention_mask": window_mask,
                    "target_encoded_pos": encoded_pos,
                    "target_word_id": word_id,
                    "window_start": window["start_word"],
                    "window_start_encoded": start,
                    "doc_id": buf.doc_id,
                    "config_name": config["name"],
                })

        return jobs


    def _run_masked_window_batch(self, jobs: list[dict]):
        """Run batched masked window inference.
        Returns a list aligned with jobs. Each item contains all token vectors
        extracted from that window.
        """
        if not jobs:
            return []

        results = []
        for i in range(0, len(jobs), self.batch_size):
            chunk = jobs[i:i + self.batch_size]
            results.extend(self._forward_masked_window_batch(chunk))
        return results

    def _forward_masked_window_batch(self, jobs: list[dict]):
        """Forward pass and extract vector ONLY at the target masked position."""
        if not jobs:
            return []

        max_len = max(len(j["input_ids"]) for j in jobs)

        def pad(seq, value=0):
            return seq + [value] * (max_len - len(seq))

        input_ids_t = torch.tensor([pad(j["input_ids"]) for j in jobs], dtype=torch.long, device=self.device)
        attention_mask_t = torch.tensor([pad(j["attention_mask"]) for j in jobs], dtype=torch.long, device=self.device)

        with torch.inference_mode():
            out = self.macberth.encode(input_ids=input_ids_t, attention_mask=attention_mask_t, return_dict=True)

        hidden = out.last_hidden_state.cpu().numpy()
        results = []

        for batch_idx, job in enumerate(jobs):
            pos = job["target_encoded_pos"]
            vec = hidden[batch_idx, pos].astype(np.float32)
            results.append(vec)

        return results


    def _events_from_masked_windows(self, buf: DocBuffer, jobs, vectors, config_name: str):
        """Convert masked target vectors into Events."""
        events = []

        for job, vec in zip(jobs, vectors):
            word_id = job["target_word_id"]

            if word_id >= len(buf.corpus_token_idxs):
                continue

            token = buf.tokens[word_id]

            if not is_mask_target(token):
                continue

            events.append( Event.make(
                    doc_id=buf.doc_id,
                    corpus=buf.corpus,
                    corpus_token_idx=buf.corpus_token_idxs[word_id],
                    window_start_token_idx=buf.corpus_token_idxs[
                        job.get("window_start", 0)
                    ],
                    window_start=job.get("window_start", 0),
                    window_token_pos=job["target_encoded_pos"],
                    token=token,
                    vector_id=buf.vector_ids[word_id],
                    vec=vec,
                    config_name=config_name,
            ))

        return events

    @staticmethod
    def _compute_windows(word_ids, window_size, stride):
        """Compute every window span once per (document, config).
        Each entry records the encoded [start, end) span plus the word-id
        range it covers, so per-token lookups become simple range checks
        instead of rescanning ``word_ids``.
        """
        first_encoded_idx: dict[int, int] = {}
        for i, wid in enumerate(word_ids):
            if wid is not None and wid >= 0 and wid not in first_encoded_idx:
                first_encoded_idx[wid] = i

        valid = [wid for wid in word_ids if wid is not None and wid >= 0]
        if not valid:
            return []
        n_words = max(valid) + 1
        n = len(word_ids)

        windows = []
        start_word = 0
        while start_word < n_words:
            encoded_start = first_encoded_idx.get(start_word)
            if encoded_start is None:
                break
            encoded_end = min(encoded_start + window_size, n)

            span_words = [
                wid for wid in word_ids[encoded_start:encoded_end]
                if wid is not None and wid >= 0
            ]
            if span_words:
                windows.append({
                    "encoded_start": encoded_start,
                    "encoded_end": encoded_end,
                    "start_word": start_word,
                    "min_word": span_words[0],
                    "max_word": span_words[-1],
                    "mid": (span_words[0] + span_words[-1]) / 2,
                })

            if encoded_end == n:
                break
            start_word += stride

        return windows


    @staticmethod
    def _best_window_for_token(windows: list[dict], word_id: int) -> dict | None:
        """
        Given the windows for a (document, config) pair, as produced by
        ``_compute_windows``, return the single window that gives
        ``word_id`` (a word-level position, not an encoded/subword index)
        the most *centered* placement.

        A token can fall inside several overlapping windows (that's the
        point of striding). Callers that want exactly one representative
        window per occurrence — rather than one observation per window,
        as Tier 1's own masked-job builder does — use this to pick
        whichever window's covered span best centers the target, on the
        theory that a token nearer the middle of its window has the most
        balanced left/right context available to the model, rather than
        being truncated near an edge.

        Ties (equal distance to two windows' midpoints) resolve to
        whichever window is encountered first in ``windows`` (i.e. the
        earlier / smaller ``start_word``), which is deterministic given
        ``_compute_windows`` always returns windows in start-word order.

        Returns ``None`` if no window's [min_word, max_word] span covers
        ``word_id`` at all (can happen right at a document boundary).
        """
        best = None
        best_dist = None
        for window in windows:
            if window["min_word"] <= word_id <= window["max_word"]:
                dist = abs(window["mid"] - word_id)
                if best is None or dist < best_dist:
                    best = window
                    best_dist = dist
        return best


    # Shared helper methods
    def _encode(self, tokens):
        enc = self.tokenizer(tokens, is_split_into_words=True, truncation=False, return_tensors="pt")
        word_ids = enc.word_ids() or [None] * len(enc["input_ids"][0])
        return enc["input_ids"][0].tolist(), enc["attention_mask"][0].tolist(), word_ids


    def _forward_single_window(self, ids, mask):
        input_ids = torch.tensor([ids], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([mask], dtype=torch.long).to(self.device)
        with torch.inference_mode():
            out = self.macberth.encode(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        return out.last_hidden_state[0].cpu().numpy()


    @staticmethod
    def _iter_windows_config(input_ids, attention_mask, word_ids, window_size, stride):
        n = len(input_ids)
        if n == 0:
            return
        valid_word_ids = [wid for wid in word_ids if wid is not None and wid >= 0]
        n_words = max(valid_word_ids) + 1 if valid_word_ids else 0

        start_word = 0
        while start_word < n_words:
            try:
                encoded_start = next(i for i, wid in enumerate(word_ids) if wid == start_word)
            except StopIteration:
                break
            encoded_end = min(encoded_start + window_size, n)

            yield (start_word, input_ids[encoded_start:encoded_end],
                   attention_mask[encoded_start:encoded_end],
                   word_ids[encoded_start:encoded_end])

            if encoded_end == n:
                break
            start_word += stride


    @staticmethod
    def _extract_events(buf: DocBuffer, item: dict, hidden: np.ndarray, config_name: str):
        window_start = item["window_start"]
        if window_start >= len(buf.corpus_token_idxs):
            return []

        window_start_token_idx = buf.corpus_token_idxs[window_start]
        events = []

        for i, wid in enumerate(item["word_ids"]):
            if wid is None or wid < 0 or wid >= len(buf.corpus_token_idxs):
                continue
            if not is_content_token(buf.tokens[wid]):
                continue
            events.append(Event.make(
                doc_id=buf.doc_id,
                corpus=buf.corpus,
                corpus_token_idx=buf.corpus_token_idxs[wid],
                window_start_token_idx=window_start_token_idx,
                window_start=window_start,
                window_token_pos=i,
                token=buf.tokens[wid],
                vector_id=buf.vector_ids[wid],
                vec=hidden[i],
                config_name=config_name
            ))
        return events


class CorpusProcessor:
    def __init__(
        self, conn,
        zarr_path,
        pipeline,
        report_every=100,
        shard=None,
        num_shards=1
    ):
        self.zarr_path = zarr_path
        self.conn = conn
        self.pipeline = pipeline
        self.report_every = report_every
        self.shard = shard
        self.num_shards = num_shards

    def _shard_clause(self):
        if self.shard is None or self.num_shards <= 1:
            return "", []
        # abs() because hashtext can be negative; %% escapes the literal % for psycopg2
        return " AND abs(hashtext(corpus || ':' || doc_id)) %% %s = %s", [self.num_shards, self.shard]

    def process(self, doc_id=None):
        store = ZarrEmbeddingObservationStore(
            path=str(self.zarr_path),
            dim=self.pipeline.hidden_size
        )

        # already_processed = set(store.get_doc_ids())
        already_processed = set(store.get_doc_keys())
        completed_docs = len(already_processed)

        logger.info("[tier1] Already processed: %d", completed_docs)

        shard_sql, shard_params = self._shard_clause()

        # Count total documents in this run's scope
        count_cur = self.conn.cursor()
        if doc_id:
            count_cur.execute(
                f"SELECT COUNT(DISTINCT (corpus, doc_id)) FROM pamphlet_tokens WHERE doc_id = %s{shard_sql}",
                [doc_id] + shard_params
            )
        else:
            count_cur.execute(
                f"SELECT COUNT(DISTINCT (corpus, doc_id)) FROM pamphlet_tokens WHERE 1=1{shard_sql}",
                shard_params
            )
        total_docs = count_cur.fetchone()[0]
        count_cur.close()
        logger.info("[tier1] Documents in scope: %d", total_docs)

        cur = self.conn.cursor(name="tier1_cursor")
        cur.itersize = 10000

        if doc_id:
            cur.execute(
                f"""SELECT corpus, doc_id, token_idx, vector_id, token, pub_year
                    FROM pamphlet_tokens WHERE doc_id = %s{shard_sql}
                    ORDER BY token_idx""",
                [doc_id] + shard_params
            )
        else:
            cur.execute(
                f"""SELECT corpus, doc_id, token_idx, vector_id, token, pub_year
                    FROM pamphlet_tokens WHERE 1=1{shard_sql}
                    ORDER BY corpus, doc_id, token_idx""",
                shard_params
            )

        logger.info("[tier1] Query executed for doc_id=%s", doc_id)
        buf = None

        for row_corpus, row_doc_id, token_idx, vid, token, pub_year in cur:
            doc_key = (row_corpus, row_doc_id)

            if doc_key in already_processed:
                continue

            if buf is None or doc_key != buf.key:
                if buf:
                    self._flush(buf, store)
                    completed_docs += 1

                    if completed_docs % self.report_every == 0:
                        pct = (
                            completed_docs / total_docs * 100
                            if total_docs
                            else 0
                        )
                        logger.info( "[tier1] Processed %d/%d documents (%.1f%%)", completed_docs, total_docs, pct )

                buf = DocBuffer(
                    corpus=row_corpus,
                    doc_id=row_doc_id,
                    pub_year=pub_year
                )

            buf.append(token, vid, token_idx)

        # Final document
        if buf and buf.key not in already_processed:
            self._flush(buf, store)
            completed_docs += 1

            pct = (
                completed_docs / total_docs * 100
                if total_docs
                else 0
            )
            logger.info( "[tier1] Processed %d/%d documents (%.1f%%)", completed_docs, total_docs, pct )

    def _flush(self, buf, store):
        start = time.perf_counter()

        raw_events = self.pipeline.embed_doc(buf)

        logger.info( "[tier1] Embedding %s took %.2fs", buf.doc_id, time.perf_counter() - start )

        if not raw_events:
            return

        # Group by (corpus_token_idx, window_token_pos) and align scales
        events_by_token = defaultdict(dict)

        for e in raw_events:
            # key = (e.corpus_token_idx, e.window_token_pos)
            key = (
                e.corpus,
                e.doc_id,
                e.corpus_token_idx,
                e.window_token_pos
            )
            events_by_token[key][e.config_name] = e

        logger.info( "[tier1] Aligned observations: %d", len(events_by_token) )

        # Build aligned arrays (same schema as before)
        event_ids, concept_ids = [], []
        emb_local, emb_medium, emb_broad = [], [], []
        vector_ids, doc_ids, token_idxs, tokens = [], [], [], []
        window_ids, window_token_poss = [], []
        corpora = []

        for key, config_dict in events_by_token.items():
            if len(config_dict) < 2:
                continue

            canonical = config_dict.get("medium") or list(config_dict.values())[0]

            event_ids.append(canonical.event_id)
            concept_ids.append(canonical.concept_id)

            emb_local.append(config_dict.get("local", canonical).vec)
            emb_medium.append(config_dict.get("medium", canonical).vec)
            emb_broad.append(config_dict.get("broad", canonical).vec)

            vector_ids.append(canonical.vector_id)
            corpora.append(canonical.corpus)
            doc_ids.append(canonical.doc_id)
            token_idxs.append(canonical.corpus_token_idx)
            tokens.append(canonical.token)
            window_ids.append(canonical.window_start)
            window_token_poss.append(canonical.window_token_pos)

        if not event_ids:
            return

        logger.info(f"[tier1] Doc {buf.doc_id}: {len(event_ids):,} tokens with multi-scale embeddings "
                   f"({'masked' if self.pipeline.mask_targets else 'unmasked'})")

        store.append_events(
            event_id            = np.asarray(event_ids, dtype=np.int64),
            concept_id          = np.asarray(concept_ids, dtype=np.int64),
            emb_local           = np.stack(emb_local),
            emb_medium          = np.stack(emb_medium),
            emb_broad           = np.stack(emb_broad),
            vector_id           = np.asarray(vector_ids, dtype=np.int64),
            doc_id              = np.asarray(doc_ids, dtype="U64"),
            corpus              = np.asarray(corpora, dtype="U32"),
            pub_year            = np.full(len(event_ids), buf.pub_year, dtype=np.int16),
            token_idx           = np.asarray(token_idxs, dtype=np.int64),
            token               = np.asarray(tokens, dtype=object),
            window_id           = np.asarray(window_ids, dtype=np.int64),
            window_token_pos    = np.asarray(window_token_poss, dtype=np.int32),
        )


def clear_output_dir(zarr_path):
    zarr_path.mkdir(parents=True, exist_ok=True)   # ensure it exists, never remove the inode itself
    for child in zarr_path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clear", action="store_true", help="Wipe the store, start from scratch")
    p.add_argument("--doc-id", type=str, default=None, help="doc_id of a document to process")
    p.add_argument("--report-every", type=int, default=100)
    p.add_argument("--mask", action="store_true", help="Use masking")
    p.add_argument("--pooling-scope", choices=["mask_only", "context"], default="mask_only")
    p.add_argument("--batch-size", type=int, default=EMBED_BATCH_SIZE)
    p.add_argument("--mask-only-position", action="store_true", default=True, help="Mask only the target token (recommended for semantics)")
    p.add_argument("--shard", type=int, default=None, help="This process's shard index (0-based)")
    p.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    p.add_argument("--backend", choices=["onnx", "pytorch"], default="onnx", help="Inference backend for embedding (default: onnx/DirectML)")
    return p.parse_args()


def main():
    args = parse_args()

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 2)))  # match OMP/MKL env vars — intra-op parallelism
    torch.set_num_interop_threads(1)                                  # not running parallel independent ops, so keep this low
    logger.info("[tier1] Thread config: OMP_NUM_THREADS=%s, torch.get_num_threads()=%d", os.environ.get("OMP_NUM_THREADS"), torch.get_num_threads())

    if args.backend == "pytorch":
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 2)))
        torch.set_num_interop_threads(1)
        logger.info("[tier1] Thread config: OMP_NUM_THREADS=%s, torch.get_num_threads()=%d", os.environ.get("OMP_NUM_THREADS"), torch.get_num_threads())

    if args.mask:
        base_path = MASKED_ZARR_PATH
    else:
        base_path = ZARR_PATH

    if args.num_shards > 1:
        from numcodecs import blosc # Used by Zarr
        blosc.set_nthreads(1)
        zarr_path = base_path.parent / f"{base_path.name}_shard{args.shard}"
    else:
        zarr_path = base_path

    if args.clear:
        logger.info("[tier1] Clearing Tier 1 output")
        clear_output_dir(zarr_path)

    conn = get_connection()

    if args.backend == "onnx":
        from lib.macberth import load_macberth_onnx
        mac = load_macberth_onnx()
    else:
        from lib.macberth import load_macberth
        mac = load_macberth(use_qint8=False)


    pipeline = EmbeddingPipeline(
        mac,
        mask_targets        = args.mask,
        mask_only_position  = args.mask_only_position,
        pooling_scope       = args.pooling_scope,
        batch_size          = args.batch_size
    )

    proc = CorpusProcessor(
        conn, zarr_path, pipeline,
        report_every = args.report_every,
        shard        = args.shard,
        num_shards   = args.num_shards
    )
    proc.process(doc_id=args.doc_id)

    # A shard is mergeable only after the complete document stream has
    # processed successfully. Presence of the Zarr directory alone is not
    # sufficient because interrupted writes leave valid-looking stores.
    if args.num_shards > 1:
        (zarr_path / "_COMPLETE").touch()

    conn.close()
    logger.info(f"[Tier 1 done] mode={'masked' if args.mask else 'unmasked'}")


if __name__ == "__main__":
    main()
