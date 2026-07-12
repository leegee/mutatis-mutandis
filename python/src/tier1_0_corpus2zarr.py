#!/usr/bin/env python
"""
tier1_corpus2zarr.py - Tier 1: Multi-scale Contextual Embedding Construction

Construct the Tier 1 contextual observation store from the EEBO corpus.

Each content token is embedded under multiple contextual windows
(local, medium and broad), producing one contextual observation per
(token, window) pair. By default the target token is replaced with
[MASK] before inference so that the resulting embedding primarily
captures contextual semantics rather than lexical identity. The
original unmasked behaviour can be restored with ``--no-mask``.

The output is written to the Tier 1 Zarr observation store, where each
row represents a single contextual observation and records:

    • stable event and concept identifiers
    • document and corpus token identifiers
    • aligned multi-scale contextual embeddings
    • window metadata (start position and token position)
    • token text and vector identifier

Architecture
------------

Input:

    PostgreSQL pamphlet_tokens
        ↓
    document buffering
        ↓
    token filtering (content words only)
        ↓
    multi-scale contextual embedding
        ↓
    aligned observation construction
        ↓
    Tier 1 Zarr observation store

The resulting store forms the canonical contextual observation layer
used by downstream indexing, neighbourhood search, clustering and
semantic change analysis.

Technical overview
------------------

The pipeline streams tokenised documents from PostgreSQL, filters
non-content tokens, and embeds each remaining token under multiple
contextual windows using MacBERTh. Each contextual observation is
assigned stable event and concept identifiers before aligned local,
medium and broad embeddings are written to the Tier 1 Zarr store.

Masked-target embeddings are the default. For each observation the
target token is replaced with ``[MASK]`` and the hidden state at the
masked position (or surrounding context) is pooled, encouraging the
embedding to encode contextual meaning rather than lexical identity.
The original unmasked embedding behaviour remains available via
``--no-mask``.

Implementation notes
--------------------

The window-selection implementation now uses a single, consistent
masked-job pipeline: ``_build_masked_jobs_for_config`` builds jobs,
``_run_masked_batch`` runs them in integer-indexed batches, and
``_make_event_from_job`` converts results back into ``Event`` objects.
An earlier duplicate set of these methods (keyed by string
``event_key`` instead of integer batch index) has been removed — it
silently shadowed the working implementation and caused
``jobs[job_idx]`` to receive a string instead of an int.

The intended long-term design is for window generation to become a
single, well-defined abstraction shared by both masked and unmasked
pipelines, with explicit semantics for window selection, positioning
and metadata generation.

"""

from __future__ import annotations


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"        # limit to avoid overload
os.environ["MKL_NUM_THREADS"] = "4"



import argparse
import shutil
import unicodedata
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import torch
import xxhash

from lib.eebo_db import get_connection
from lib.eebo_logging import logger
from lib.eebo_config import ZARR_PATH, MASKED_ZARR_PATH, EMBED_BATCH_SIZE
from lib.zarr_embedding_observation_store import ZarrEmbeddingObservationStore
from lib.macberth import load_macberth


WINDOW_CONFIGS = [
    {"name": "local",  "size": 256,  "stride": 128},
    {"name": "medium", "size": 512,  "stride": 256},
    {"name": "broad",  "size": 512,  "stride": 384},
]



STOPWORDS = {
    "the", "a", "an", "i", "he", "she", "it", "we", "they", "who", "which", "that",
    "his", "her", "its", "our", "their", "my", "thy", "your", "hast", "art", "is",
    "thine", "mine", "him", "them", "us", "me", "and", "or", "but", "of", "in",
    "to", "for", "with", "by", "at", "from", "as", "on", "into", "upon", "unto",
    "not", "nor", "yet", "so", "if", "be", "are", "was", "were", "shall", "will",
    "may", "should", "would", "could", "than", "then", "when", "this", "these",
    "those", "all", "no", "any", "such", "many", "some",
}


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


@dataclass(slots=True)
class Event:
    event_id: np.int64
    concept_id: np.int64
    doc_id: str
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
    def make(doc_id: str, corpus_token_idx: int, window_start_token_idx: int,
             window_start: int, window_token_pos: int, token: str, vector_id: int,
             vec: np.ndarray, config_name: str = "medium") -> Event:
        concept_id = stable_hash(f"{doc_id}:{corpus_token_idx}")
        event_id = stable_hash(
            f"{doc_id}:{corpus_token_idx}:{window_start_token_idx}:{window_token_pos}:{config_name}"
        )
        return Event(
            event_id=event_id, concept_id=concept_id, doc_id=doc_id,
            corpus_token_idx=corpus_token_idx, window_start=window_start,
            window_start_token_idx=window_start_token_idx,
            window_token_pos=window_token_pos, token=token,
            vector_id=vector_id, vec=vec.astype(np.float32),
            config_name=config_name
        )


@dataclass
class DocBuffer:
    doc_id: str

    def __post_init__(self):
        self.tokens = []
        self.vector_ids = []
        self.corpus_token_idxs = []

    def append(self, token, vector_id, token_idx):
        self.tokens.append(token)
        self.vector_ids.append(vector_id)
        self.corpus_token_idxs.append(token_idx)

    def __bool__(self):
        return bool(self.tokens)


class EmbeddingPipeline:
    def __init__(self, tokenizer, model, device, mask_targets: bool = True,
                 pooling_scope: str = "mask_only", batch_size: int = EMBED_BATCH_SIZE):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.mask_targets = mask_targets
        self.pooling_scope = pooling_scope
        self.batch_size = batch_size
        self.configs = WINDOW_CONFIGS

    def embed_doc(self, buf: DocBuffer) -> list[Event]:
        if not self.mask_targets:
            return self._embed_doc_unmasked(buf)
        return self._embed_doc_masked(buf)

    # UNMASKED (original fast path)
    def _embed_doc_unmasked(self, buf: DocBuffer) -> list[Event]:
        input_ids, attention_mask, word_ids = self._encode(buf.tokens)
        all_events = []

        for config in self.configs:
            for window_start, ids, mask, wids in self._iter_windows_config(
                input_ids, attention_mask, word_ids, config["size"], config["stride"]
            ):
                hidden = self._forward_single_window(ids, mask)
                events = self._extract_events(buf, {"window_start": window_start, "word_ids": wids},
                                              hidden, config["name"])
                all_events.extend(events)
        return all_events

    # ==================== MASKED (default) ====================
    def _embed_doc_masked(self, buf):
        input_ids, attention_mask, word_ids = self._encode(buf.tokens)

        all_events = []

        for config in self.configs:
            jobs = self._build_masked_window_jobs(
                buf,
                input_ids,
                attention_mask,
                word_ids,
                config,
            )

            vectors = self._run_masked_window_batch(jobs)

            all_events.extend(
                self._events_from_masked_windows(
                    buf,
                    jobs,
                    vectors,
                    config["name"]
                )
            )

        logger.info(
            "Document %s: %d masked observations",
            buf.doc_id,
            len(all_events),
        )

        return all_events

    def _build_masked_window_jobs(
        self,
        buf: DocBuffer,
        input_ids,
        attention_mask,
        word_ids,
        config,
    ):
        """Create one masked inference job per contextual window.

        The previous implementation created one job per target token. This
        version creates one job per transformer input window and extracts all
        masked token representations afterwards.
        """
        windows = self._compute_windows(
            word_ids,
            config["size"],
            config["stride"],
        )

        if not windows:
            return []

        jobs = []

        for window in windows:
            start = window["encoded_start"]
            end = window["encoded_end"]

            window_ids = list(input_ids[start:end])
            window_mask = list(attention_mask[start:end])
            window_word_ids = word_ids[start:end]

            mask_positions = {}

            for encoded_pos, word_id in enumerate(window_word_ids):
                if word_id is None or word_id < 0:
                    continue

                if word_id >= len(buf.corpus_token_idxs):
                    continue

                # Ignore punctuation/special tokens. buf already contains only
                # content tokens, so every valid word id is a target candidate.
                mask_positions[encoded_pos] = word_id

                window_ids[encoded_pos] = self.tokenizer.mask_token_id

            if not mask_positions:
                continue

            jobs.append(
                {
                    "input_ids": window_ids,
                    "attention_mask": window_mask,
                    "mask_positions": mask_positions,
                    "window_start": window["start_word"],
                    "window_start_encoded": start,
                    "doc_id": buf.doc_id,
                }
            )

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

            results.extend(
                self._forward_masked_window_batch(chunk)
            )

        return results


    def _forward_masked_window_batch(self, jobs: list[dict]):
        """Forward a batch of masked windows and extract all masked positions."""

        logger.info( "MacBERTh forward batch: %d windows", len(jobs) )

        max_len = max(len(j["input_ids"]) for j in jobs)

        def pad(seq, value=0):
            return seq + [value] * (max_len - len(seq))

        input_ids_t = torch.tensor(
            [pad(j["input_ids"]) for j in jobs],
            dtype=torch.long,
            device=self.device,
        )

        attention_mask_t = torch.tensor(
            [pad(j["attention_mask"]) for j in jobs],
            dtype=torch.long,
            device=self.device,
        )

        with torch.inference_mode():
            out = self.model(
                input_ids=input_ids_t,
                attention_mask=attention_mask_t,
                return_dict=True,
            )

        hidden = out.last_hidden_state.cpu().numpy()

        results = []

        for batch_idx, job in enumerate(jobs):
            vectors = {}

            for position, word_id in job["mask_positions"].items():
                vectors[word_id] = hidden[batch_idx, position].astype(np.float32)

            results.append(vectors)

        return results


    def _events_from_masked_windows(
        self,
        buf: DocBuffer,
        jobs,
        vectors,
        config_name: str,
    ):
        """Convert window outputs into Tier 1 Events."""

        events = []

        for job, window_vectors in zip(jobs, vectors):

            window_start = job["window_start"]

            if window_start >= len(buf.corpus_token_idxs):
                continue

            window_start_token_idx = buf.corpus_token_idxs[window_start]

            for word_id, vec in window_vectors.items():

                if word_id >= len(buf.corpus_token_idxs):
                    continue

                if not is_content_token(buf.tokens[word_id]):
                    continue

                events.append(
                    Event.make(
                        doc_id=buf.doc_id,
                        corpus_token_idx=buf.corpus_token_idxs[word_id],
                        window_start_token_idx=window_start_token_idx,
                        window_start=window_start,
                        window_token_pos=word_id,
                        token=buf.tokens[word_id],
                        vector_id=buf.vector_ids[word_id],
                        vec=vec,
                        config_name=config_name,
                    )
                )

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


    def _make_event_from_job(self, buf: DocBuffer, job: dict, vec: np.ndarray, config_name: str) -> Event:
        """Convert a processed job back into an Event with proper window metadata."""
        window_start = job["window_start_word"]
        window_start_token_idx = (
            buf.corpus_token_idxs[window_start]
            if window_start < len(buf.corpus_token_idxs)
            else 0
        )

        return Event.make(
            doc_id=job["doc_id"],
            corpus_token_idx=job["target_idx"],
            window_start_token_idx=window_start_token_idx,
            window_start=window_start,
            window_token_pos=job["window_token_pos"],
            token=job["token"],
            vector_id=job["vector_id"],
            vec=vec,
            config_name=config_name
        )

    # Shared helper methods
    def _encode(self, tokens):
        enc = self.tokenizer(tokens, is_split_into_words=True, truncation=False, return_tensors="pt")
        word_ids = enc.word_ids() or [None] * len(enc["input_ids"][0])
        return enc["input_ids"][0].tolist(), enc["attention_mask"][0].tolist(), word_ids

    def _forward_single_window(self, ids, mask):
        input_ids = torch.tensor([ids], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([mask], dtype=torch.long).to(self.device)

        with torch.inference_mode():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
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
    def __init__(self,  conn, zarr_path, pipeline, report_every: int = 100):
        self.zarr_path = zarr_path
        self.conn = conn
        self.pipeline = pipeline
        self.report_every = report_every

    def process(self, doc_id=None):
        store = ZarrEmbeddingObservationStore(path=str(self.zarr_path), dim=self.pipeline.model.config.hidden_size)
        already_processed = set(store.get_doc_ids())
        logger.info("already processed: %d", len(already_processed))

        cur = self.conn.cursor(name="tier1_cursor")
        cur.itersize = 10000

        if doc_id:
            cur.execute("SELECT doc_id, token_idx, vector_id, token FROM pamphlet_tokens WHERE doc_id = %s ORDER BY token_idx", (doc_id,))
        else:
            cur.execute("SELECT doc_id, token_idx, vector_id, token FROM pamphlet_tokens ORDER BY doc_id, token_idx")

        logger.info("query executed for doc_id=%s", doc_id)

        buf = None
        docs_processed = 0

        for row_doc_id, token_idx, vid, token in cur:
            if row_doc_id in already_processed:
                continue

            if buf is None or row_doc_id != buf.doc_id:
                if buf:
                    self._flush(buf, store)
                    docs_processed += 1
                    if docs_processed % self.report_every == 0:
                        logger.info(f"Processed {docs_processed} documents")
                buf = DocBuffer(doc_id=row_doc_id)

            buf.append(token, vid, token_idx)

        if buf and buf.doc_id not in already_processed:
            self._flush(buf, store)

    def _flush(self, buf, store):

        import time
        start = time.perf_counter()

        raw_events = self.pipeline.embed_doc(buf)

        logger.info( "Embedding %s took %.2fs", buf.doc_id, time.perf_counter() - start )

        if not raw_events:
            return

        # Group by (corpus_token_idx, window_token_pos) and align scales
        events_by_token = defaultdict(dict)
        for e in raw_events:
            key = (e.corpus_token_idx, e.window_token_pos)
            events_by_token[key][e.config_name] = e

        logger.info( "aligned observations: %d", len(events_by_token) )

        # Build aligned arrays (same schema as before)
        event_ids, concept_ids = [], []
        emb_local, emb_medium, emb_broad = [], [], []
        vector_ids, doc_ids, token_idxs, tokens = [], [], [], []
        window_ids, window_token_poss = [], []

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
            doc_ids.append(canonical.doc_id)
            token_idxs.append(canonical.corpus_token_idx)
            tokens.append(canonical.token)
            window_ids.append(canonical.window_start)
            window_token_poss.append(canonical.window_token_pos)

        if not event_ids:
            return

        logger.info(f"Doc {buf.doc_id}: {len(event_ids):,} tokens with multi-scale embeddings "
                   f"({'masked' if self.pipeline.mask_targets else 'unmasked'})")

        store.append_events(
            event_id=np.asarray(event_ids, dtype=np.int64),
            concept_id=np.asarray(concept_ids, dtype=np.int64),
            emb_local=np.stack(emb_local),
            emb_medium=np.stack(emb_medium),
            emb_broad=np.stack(emb_broad),
            vector_id=np.asarray(vector_ids, dtype=np.int64),
            doc_id=np.asarray(doc_ids, dtype="U32"),
            token_idx=np.asarray(token_idxs, dtype=np.int64),
            token=np.asarray(tokens, dtype=object),
            window_id=np.asarray(window_ids, dtype=np.int64),
            window_token_pos=np.asarray(window_token_poss, dtype=np.int32),
        )


def clear_output_dir(zarr_path):
    if zarr_path.exists():
        shutil.rmtree(zarr_path)
    zarr_path.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clear", action="store_true")
    p.add_argument("--doc-id", type=str, default=None)
    p.add_argument("--report-every", type=int, default=100)
    p.add_argument("--no-mask", action="store_true", help="Disable masking (original unmasked behavior)")
    p.add_argument("--pooling-scope", choices=["mask_only", "context"], default="mask_only")
    p.add_argument("--batch-size", type=int, default=EMBED_BATCH_SIZE)
    return p.parse_args()


def main():
    args = parse_args()

    if args.no_mask:
        zarr_path = ZARR_PATH
    else:
        zarr_path = MASKED_ZARR_PATH

    if args.clear:
        logger.info("Clearing Tier 1 output")
        clear_output_dir(zarr_path)

    conn = get_connection()
    mac = load_macberth()

    pipeline = EmbeddingPipeline(
        mac.tokenizer, mac.model, mac.device,
        mask_targets=not args.no_mask,
        pooling_scope=args.pooling_scope,
        batch_size=args.batch_size
    )

    proc = CorpusProcessor(conn, zarr_path, pipeline, report_every=args.report_every)
    proc.process(doc_id=args.doc_id)

    conn.close()
    logger.info(f"[Tier 1 done] mode={'masked' if not args.no_mask else 'unmasked'}")


if __name__ == "__main__":
    main()
