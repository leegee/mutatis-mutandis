#!/usr/bin/env python
"""
tier1_corpus2parquet.py - Tier 1: Multi-scale Contextual Embedding Construction

Construct the Tier 1 contextual observation store from the EEBO corpus.

Each selected token occurrence is embedded under multiple contextual windows
(local, medium and broad), producing three aligned contextual embeddings for
downstream retrieval, clustering and semantic change analysis.

The output is written to the Tier 1 observation store using the Parquet
storage backend. Each row represents one token occurrence
with its aligned multi-scale embeddings and scale-specific window provenance.
"""

from __future__ import annotations

import os
import psutil


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
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

from pathlib import Path

from lib.corpus_db import get_connection
from lib.corpus_logging import logger
import lib.corpus_config as config
from lib.DocBuffer import DocBuffer
from lib.stopwords_min import STOPWORDS

from tier0.tier0_clmet_extreme_whiteness import SEPARATOR

from tier1.observation_store_api import (
    resolve_store_path,
    open_observation_writer,
)

import lib.parquet_observation_backend

WINDOW_CONFIGS = [
    {"name": "local", "size": 256, "stride": 128},
    {"name": "medium", "size": 512, "stride": 256},
    {"name": "broad", "size": 512, "stride": 384},
]

_PROCESS = psutil.Process(os.getpid())



def rss_gb():
    return _PROCESS.memory_info().rss / (1024 ** 3)


def stable_hash(key: str) -> np.int64:
    h = xxhash.xxh64(key, seed=0).intdigest()
    return np.int64(h & 0x7FFFFFFFFFFFFFFF)


def is_separator_token(token: str) -> bool:
    return token == SEPARATOR


def is_content_token(token: str) -> bool:
    stripped = token.strip().lower()

    if not stripped or stripped in STOPWORDS:
        return False

    if is_separator_token(token):
        return False

    if all(
        unicodedata.category(c).startswith(("P", "S", "Z"))
        for c in stripped
    ):
        return False

    return True


def is_mask_target(token: str) -> bool:
    normalised = unicodedata.normalize("NFKC", token).lower()

    return any(
        normalised in concept["forms"]
        for concept in config.CONCEPT_SETS.values()
    )


@dataclass(slots=True)
class Event:
    """
    Intermediate scale-specific representation of one corpus occurrence.

    A single corpus occurrence may produce several Event instances because
    overlapping windows are processed independently. These are aligned later
    into one stored observation containing one embedding per scale.
    """

    event_id: np.int64
    doc_id: str
    corpus: str
    corpus_token_idx: int
    window_start: int
    window_start_token_idx: int
    window_token_pos: int
    token: str
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
        vec: np.ndarray,
        config_name: str = "medium",
    ) -> Event:
        # The intermediate Event identity includes scale because the same
        # corpus occurrence is represented separately while each scale is
        # being constructed.
        event_id = stable_hash(
            f"{corpus}:{doc_id}:{corpus_token_idx}:{config_name}"
        )

        return Event(
            event_id=event_id,
            doc_id=doc_id,
            corpus=corpus,
            corpus_token_idx=corpus_token_idx,
            window_start=window_start,
            window_start_token_idx=window_start_token_idx,
            window_token_pos=window_token_pos,
            token=token,
            vec=vec.astype(np.float32),
            config_name=config_name,
        )


class EmbeddingPipeline:
    def __init__(
        self,
        mac,
        mask_targets: bool = True,
        pooling_scope: str = "mask_only",
        mask_only_position: bool = True,
        batch_size: int = config.EMBED_BATCH_SIZE,
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

        events_by_token = defaultdict(dict)
        batch = []

        def flush_batch():
            if not batch:
                return

            hiddens = self._forward_window_batch(batch)

            for item, hidden in zip(batch, hiddens):
                for event in self._extract_events(
                    buf,
                    item,
                    hidden,
                    item["config_name"],
                ):
                    key = (
                        event.corpus,
                        event.doc_id,
                        event.corpus_token_idx,
                    )

                    # Multiple overlapping windows produce the same
                    # token/scale observation. Retain only the latest one.
                    events_by_token[key][event.config_name] = event

            batch.clear()

        for config in self.configs:
            for (
                window_start,
                ids,
                mask,
                wids,
            ) in self._iter_windows_config(
                input_ids,
                attention_mask,
                word_ids,
                config["size"],
                config["stride"],
            ):
                batch.append(
                    {
                        "window_start": window_start,
                        "input_ids": ids,
                        "attention_mask": mask,
                        "word_ids": wids,
                        "config_name": config["name"],
                    }
                )

                if len(batch) >= self.batch_size:
                    flush_batch()

        flush_batch()

        return [
            event
            for config_dict in events_by_token.values()
            for event in config_dict.values()
        ]


    def _forward_window_batch(self, chunk):
        max_len = max(len(c["input_ids"]) for c in chunk)

        def pad(seq, value=0):
            return seq + [value] * (max_len - len(seq))

        input_ids_t = torch.tensor(
            [pad(c["input_ids"]) for c in chunk],
            dtype=torch.long,
            device=self.device,
        )

        attention_mask_t = torch.tensor(
            [pad(c["attention_mask"]) for c in chunk],
            dtype=torch.long,
            device=self.device,
        )

        with torch.inference_mode():
            out = self.macberth.encode(
                input_ids=input_ids_t,
                attention_mask=attention_mask_t,
                return_dict=True,
            )

        hidden = out.last_hidden_state.cpu().numpy()

        return [
            hidden[j, : len(chunk[j]["input_ids"])]
            for j in range(len(chunk))
        ]

    # MASKED
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

            if (
                not self._window_alignment_checked
                and config["name"] == "local"
            ):
                self._check_window_alignment_once(jobs)

            vectors = self._run_masked_window_batch(jobs)

            all_events.extend(
                self._events_from_masked_windows(
                    buf,
                    jobs,
                    vectors,
                    config["name"],
                )
            )

        logger.info(
            "[tier1] Document %s: %d masked observations (target-only=%s)",
            buf.doc_id,
            len(all_events),
            self.mask_only_position,
        )

        return all_events

    def _check_window_alignment_once(self, jobs: list[dict]) -> None:
        """Guard against collapsing overlapping window observations."""

        self._window_alignment_checked = True

        if not jobs:
            logger.warning(
                "[tier1] Window-alignment check skipped: "
                "no jobs in first doc/config"
            )
            return

        positions_by_word = defaultdict(set)

        for job in jobs:
            positions_by_word[job["target_word_id"]].add(
                job["target_encoded_pos"]
            )

        repeated = {
            wid: len(pos)
            for wid, pos in positions_by_word.items()
            if len(pos) > 1
        }

        if not repeated:
            logger.warning(
                "[tier1] Window-alignment check: no word appeared in "
                ">1 window for this doc (doc too short to exercise "
                "overlap) — check inconclusive"
            )
            return

        sample_wid, n_positions = next(iter(repeated.items()))

        logger.info(
            "[tier1] Window-alignment check passed: "
            "word_id=%d appears in %d windows with distinct "
            "target_encoded_pos values (%s)",
            sample_wid,
            n_positions,
            sorted(positions_by_word[sample_wid]),
        )

        assert all(n >= 2 for n in repeated.values()), (
            "[tier1] Window-alignment check failed: a word repeated "
            "across windows but target_encoded_pos did not vary"
        )

    def _build_masked_window_jobs(
        self,
        buf: DocBuffer,
        input_ids,
        attention_mask,
        word_ids,
        config,
    ):
        """Mask only the target token per job."""

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

            for encoded_pos, word_id in enumerate(window_word_ids):
                if word_id is None or word_id < 0:
                    continue

                if word_id >= len(buf.corpus_token_idxs):
                    continue

                if not is_mask_target(buf.tokens[word_id]):
                    continue

                target_window_ids = list(window_ids)
                target_window_ids[encoded_pos] = (
                    self.tokenizer.mask_token_id
                )

                jobs.append(
                    {
                        "input_ids": target_window_ids,
                        "attention_mask": window_mask,
                        "target_encoded_pos": encoded_pos,
                        "target_word_id": word_id,
                        "window_start": window["start_word"],
                        "window_start_encoded": start,
                        "doc_id": buf.doc_id,
                        "config_name": config["name"],
                    }
                )

        return jobs

    def _run_masked_window_batch(self, jobs: list[dict]):
        if not jobs:
            return []

        results = []

        for i in range(0, len(jobs), self.batch_size):
            chunk = jobs[i : i + self.batch_size]
            results.extend(
                self._forward_masked_window_batch(chunk)
            )

        return results

    def _forward_masked_window_batch(self, jobs: list[dict]):
        """Extract the hidden state only at the target position."""

        if not jobs:
            return []

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
            out = self.macberth.encode(
                input_ids=input_ids_t,
                attention_mask=attention_mask_t,
                return_dict=True,
            )

        hidden = out.last_hidden_state.cpu().numpy()
        results = []

        for batch_idx, job in enumerate(jobs):
            pos = job["target_encoded_pos"]
            vec = hidden[batch_idx, pos].astype(np.float32)
            results.append(vec)

        return results

    def _events_from_masked_windows(
        self,
        buf,
        jobs,
        vectors,
        config_name: str,
    ):
        """Convert masked target vectors into scale-specific Events."""

        events = []

        for job, vec in zip(jobs, vectors):
            word_id = job["target_word_id"]

            if word_id >= len(buf.corpus_token_idxs):
                continue

            token = buf.tokens[word_id]

            if not is_mask_target(token):
                continue

            events.append(
                Event.make(
                    doc_id=buf.doc_id,
                    corpus=buf.corpus,
                    corpus_token_idx=buf.corpus_token_idxs[word_id],
                    window_start_token_idx=buf.corpus_token_idxs[
                        job.get("window_start", 0)
                    ],
                    window_start=job.get("window_start", 0),
                    window_token_pos=job["target_encoded_pos"],
                    token=token,
                    vec=vec,
                    config_name=config_name,
                )
            )

        return events

    @staticmethod
    def _compute_windows(word_ids, window_size, stride):
        """Compute encoded spans once per document/config."""

        first_encoded_idx: dict[int, int] = {}

        for i, wid in enumerate(word_ids):
            if (
                wid is not None
                and wid >= 0
                and wid not in first_encoded_idx
            ):
                first_encoded_idx[wid] = i

        valid = [
            wid for wid in word_ids
            if wid is not None and wid >= 0
        ]

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

            encoded_end = min(
                encoded_start + window_size,
                n,
            )

            span_words = [
                wid
                for wid in word_ids[encoded_start:encoded_end]
                if wid is not None and wid >= 0
            ]

            if span_words:
                windows.append(
                    {
                        "encoded_start": encoded_start,
                        "encoded_end": encoded_end,
                        "start_word": start_word,
                        "min_word": span_words[0],
                        "max_word": span_words[-1],
                        "mid": (
                            span_words[0] + span_words[-1]
                        ) / 2,
                    }
                )

            if encoded_end == n:
                break

            start_word += stride

        return windows

    @staticmethod
    def _best_window_for_token(
        windows: list[dict],
        word_id: int,
    ) -> dict | None:
        """
        Return the window giving a token the most centred placement.
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

    def _encode(self, tokens):
        """
        Encode corpus tokens while treating <SEP> as a structural boundary.

        <SEP> is retained in the corpus/token store, but is not passed to
        MacBERTh literally because it is not part of the pretrained
        vocabulary.

        We substitute a normal punctuation token for model encoding so that
        the separator still occupies exactly one word-level position and
        therefore does not disturb word_ids alignment.

        The original corpus token list remains unchanged.
        """

        model_tokens = [
            "." if is_separator_token(token) else token
            for token in tokens
        ]

        enc = self.tokenizer(
            model_tokens,
            is_split_into_words=True,
            truncation=False,
            return_tensors="pt",
        )

        word_ids = enc.word_ids() or [
            None
        ] * len(enc["input_ids"][0])

        return (
            enc["input_ids"][0].tolist(),
            enc["attention_mask"][0].tolist(),
            word_ids,
        )


    def _forward_single_window(self, ids, mask):
        input_ids = torch.tensor(
            [ids],
            dtype=torch.long,
        ).to(self.device)

        attention_mask = torch.tensor(
            [mask],
            dtype=torch.long,
        ).to(self.device)

        with torch.inference_mode():
            out = self.macberth.encode(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

        return out.last_hidden_state[0].cpu().numpy()

    @staticmethod
    def _iter_windows_config(
        input_ids,
        attention_mask,
        word_ids,
        window_size,
        stride,
    ):
        n = len(input_ids)

        if n == 0:
            return

        valid_word_ids = [
            wid
            for wid in word_ids
            if wid is not None and wid >= 0
        ]

        n_words = (
            max(valid_word_ids) + 1
            if valid_word_ids
            else 0
        )

        start_word = 0

        while start_word < n_words:
            try:
                encoded_start = next(
                    i
                    for i, wid in enumerate(word_ids)
                    if wid == start_word
                )
            except StopIteration:
                break

            encoded_end = min(
                encoded_start + window_size,
                n,
            )

            yield (
                start_word,
                input_ids[encoded_start:encoded_end],
                attention_mask[encoded_start:encoded_end],
                word_ids[encoded_start:encoded_end],
            )

            if encoded_end == n:
                break

            start_word += stride

    @staticmethod
    def _extract_events(
        buf: DocBuffer,
        item: dict,
        hidden: np.ndarray,
        config_name: str,
    ):
        window_start = item["window_start"]

        if window_start >= len(buf.corpus_token_idxs):
            return []

        window_start_token_idx = buf.corpus_token_idxs[
            window_start
        ]

        events = []

        for i, wid in enumerate(item["word_ids"]):
            if (
                wid is None
                or wid < 0
                or wid >= len(buf.corpus_token_idxs)
            ):
                continue

            if not is_content_token(buf.tokens[wid]):
                continue

            events.append(
                Event.make(
                    doc_id=buf.doc_id,
                    corpus=buf.corpus,
                    corpus_token_idx=buf.corpus_token_idxs[wid],
                    window_start_token_idx=window_start_token_idx,
                    window_start=window_start,
                    window_token_pos=i,
                    token=buf.tokens[wid],
                    vec=hidden[i],
                    config_name=config_name,
                )
            )

        return events


class CorpusProcessor:
    def __init__(
        self,
        conn,
        store_path,
        pipeline,
        *,
        shard=None,
        num_shards=1,
        parquet_min_rows=None,
        parquet_min_bytes=None,
    ):
        self.store_path = Path(store_path)
        self.store_backend = 'parquet'
        self.conn = conn
        self.pipeline = pipeline
        self.shard = shard
        self.num_shards = num_shards
        self.parquet_min_rows = parquet_min_rows
        self.parquet_min_bytes = parquet_min_bytes

        # Back-compat alias used by older callers/logging.
        self.EVENTSTORE_T1_PATH = self.store_path

    def _shard_clause(self):
        if self.shard is None or self.num_shards <= 1:
            return "", []

        return (
            " AND abs(hashtext(corpus || ':' || doc_id)) "
            "%% %s = %s",
            [self.num_shards, self.shard],
        )

    def process(self, doc_id=None):
        writer_kwargs = {}

        if self.parquet_min_rows is not None:
            writer_kwargs["min_rows"] = self.parquet_min_rows

        if self.parquet_min_bytes is not None:
            writer_kwargs["min_bytes"] = self.parquet_min_bytes

        store = open_observation_writer(
            self.store_backend,
            self.store_path,
            dim=self.pipeline.hidden_size,
            **writer_kwargs,
        )

        already_processed = set(store.get_doc_keys())
        completed_docs = len(already_processed)

        logger.info(
            "[tier1] store_backend=%s path=%s already_processed=%d",
            self.store_backend,
            self.store_path,
            completed_docs,
        )

        shard_sql, shard_params = self._shard_clause()

        count_cur = self.conn.cursor()

        if doc_id:
            count_cur.execute(
                f"""
                SELECT COUNT(DISTINCT (corpus, doc_id))
                FROM pamphlet_tokens
                WHERE doc_id = %s{shard_sql}
                """,
                [doc_id] + shard_params,
            )
        else:
            count_cur.execute(
                f"""
                SELECT COUNT(DISTINCT (corpus, doc_id))
                FROM pamphlet_tokens
                WHERE 1=1{shard_sql}
                """,
                shard_params,
            )

        total_docs = count_cur.fetchone()[0]
        count_cur.close()

        logger.info( "[tier1] Documents in scope: %d", total_docs, )

        cur = self.conn.cursor(name="tier1_cursor")
        cur.itersize = 10000

        if doc_id:
            cur.execute(
                f"""
                SELECT corpus, doc_id, token_idx, token, pub_year
                FROM pamphlet_tokens
                WHERE doc_id = %s{shard_sql}
                ORDER BY token_idx
                """,
                [doc_id] + shard_params,
            )
        else:
            cur.execute(
                f"""
                SELECT corpus, doc_id, token_idx, token, pub_year
                FROM pamphlet_tokens
                WHERE 1=1{shard_sql}
                ORDER BY corpus, doc_id, token_idx
                """,
                shard_params,
            )

        logger.info( "[tier1] Query executed for doc_id=%s", doc_id, )

        buf = None

        for (
            row_corpus,
            row_doc_id,
            token_idx,
            token,
            pub_year,
        ) in cur:
            doc_key = (row_corpus, row_doc_id)

            if doc_key in already_processed:
                continue

            if buf is None or doc_key != buf.key:
                if buf:
                    self._flush(buf, store)
                    already_processed.add(buf.key)
                    completed_docs += 1

                    pct = (
                        completed_docs / total_docs * 100
                        if total_docs
                        else 0
                    )

                    logger.info(
                        "[tier1] Processed %d/%d documents (%.1f%%)",
                        completed_docs,
                        total_docs,
                        pct,
                    )

                buf = DocBuffer(
                    corpus=row_corpus,
                    doc_id=row_doc_id,
                    pub_year=pub_year,
                )

            buf.append(token, token_idx)

        if buf and buf.key not in already_processed:
            self._flush(buf, store)
            already_processed.add(buf.key)
            completed_docs += 1

            pct = (
                completed_docs / total_docs * 100
                if total_docs
                else 0
            )

            logger.info( "[tier1] Processed %d/%d documents (%.1f%%)", completed_docs, total_docs, pct, )

        if hasattr(store, "flush"):
            store.flush()

        if hasattr(store, "close"):
            store.close()

    def _flush(self, buf, store):
        start = time.perf_counter()

        logger.debug( "[tier1] %s RSS before embed: %.2f GB", buf.doc_id, rss_gb(), )
        raw_events = self.pipeline.embed_doc(buf)

        logger.info( "[tier1] %s RSS after embed: %.2f GB; raw_events=%d", buf.doc_id, rss_gb(), len(raw_events), )
        logger.debug( "[tier1] Embedding %s took %.2fs", buf.doc_id, time.perf_counter() - start, )

        if not raw_events:
            return

        # A raw Event is scale-specific. Alignment collapses the overlapping
        # window computations for the same corpus occurrence into one stored
        # observation with exactly one representation per configured scale.
        events_by_token = defaultdict(dict)

        for e in raw_events:
            key = (
                e.corpus,
                e.doc_id,
                e.corpus_token_idx,
            )

            events_by_token[key][e.config_name] = e

        logger.info( "[tier1] Aligned observations: %d", len(events_by_token), )

        event_ids = []
        emb_local = []
        emb_medium = []
        emb_broad = []
        doc_ids = []
        token_idxs = []
        tokens = []
        local_window_ids = []
        local_window_token_poss = []
        medium_window_ids = []
        medium_window_token_poss = []
        broad_window_ids = []
        broad_window_token_poss = []
        corpora = []

        for key, config_dict in events_by_token.items():
            # A stored observation is required to have exactly one embedding
            # at every configured scale. Partial scale sets are not observations.
            if set(config_dict) != {"local", "medium", "broad"}:
                continue

            local = config_dict["local"]
            medium = config_dict["medium"]
            broad = config_dict["broad"]

            # The scale-specific Event IDs are intermediate identities. The
            # stored observation identity is stable across its three scales,
            # so use the medium event ID as the canonical observation ID.
            event_ids.append(medium.event_id)
            emb_local.append(local.vec)
            emb_medium.append(medium.vec)
            emb_broad.append(broad.vec)
            corpora.append(medium.corpus)
            doc_ids.append(medium.doc_id)
            token_idxs.append(medium.corpus_token_idx)
            tokens.append(medium.token)
            local_window_ids.append(local.window_start)
            local_window_token_poss.append(local.window_token_pos)
            medium_window_ids.append(medium.window_start)
            medium_window_token_poss.append(medium.window_token_pos)
            broad_window_ids.append(broad.window_start)
            broad_window_token_poss.append(broad.window_token_pos)

        if not event_ids:
            return

        logger.info(
            "[tier1] Doc %s: %d tokens with multi-scale embeddings (%s)",
            buf.doc_id, len(event_ids),
            "masked" if self.pipeline.mask_targets else "unmasked",
        )

        logger.debug( "[tier1] %s RSS before parquet append: %.2f GB; rows=%d", buf.doc_id, rss_gb(), len(event_ids), )

        store.append_events(
            event_id=np.asarray(
                event_ids,
                dtype=np.int64,
            ),
            emb_local=np.stack(emb_local),
            emb_medium=np.stack(emb_medium),
            emb_broad=np.stack(emb_broad),
            doc_id=np.asarray(
                doc_ids,
                dtype="U64",
            ),
            corpus=np.asarray(
                corpora,
                dtype="U32",
            ),
            pub_year=np.full(
                len(event_ids),
                buf.pub_year,
                dtype=np.int16,
            ),
            token_idx=np.asarray(
                token_idxs,
                dtype=np.int64,
            ),
            token=np.asarray(
                tokens,
                dtype=object,
            ),
            local_window_id=np.asarray(
                local_window_ids,
                dtype=np.int64,
            ),
            local_window_token_pos=np.asarray(
                local_window_token_poss,
                dtype=np.int32,
            ),
            medium_window_id=np.asarray(
                medium_window_ids,
                dtype=np.int64,
            ),
            medium_window_token_pos=np.asarray(
                medium_window_token_poss,
                dtype=np.int32,
            ),
            broad_window_id=np.asarray(
                broad_window_ids,
                dtype=np.int64,
            ),
            broad_window_token_pos=np.asarray(
                broad_window_token_poss,
                dtype=np.int32,
            ),
        )
        logger.debug( "[tier1] %s RSS after parquet append: %.2f GB", buf.doc_id, rss_gb(), )


def clear_output_dir(store_path: Path):
    """
    Wipe store contents in place without removing the directory inode.

    Works for Hive-partitioned Parquet trees.
    """

    store_path = Path(store_path)
    store_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    for child in store_path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def parse_args():
    p = argparse.ArgumentParser(
        description="Tier 1: multi-scale contextual embedding construction"
    )

    p.add_argument( "--clear", action="store_true", help="Wipe the store, start from scratch", )
    p.add_argument( "--doc-id", type=str, default=None, help="doc_id of a document to process", )
    p.add_argument( "--batch-size", type=int, default=config.EMBED_BATCH_SIZE, )

    p.add_argument( "--mask", action="store_true", help="Use masking", )
    p.add_argument( "--pooling-scope", choices=["mask_only", "context"], default="mask_only", )
    p.add_argument( "--mask-only-position", action="store_true", default=True,
        help="Mask only the target token (recommended for semantics)",
    )

    p.add_argument( "--shard", type=int, default=None, help="This process's shard index (0-based)", )
    p.add_argument( "--num-shards", type=int, default=1, help="Total number of shards", )

    p.add_argument( "--backend", choices=["onnx", "pytorch"], default="onnx", help="Inference backend for embedding", )
    p.add_argument( "--onnx-provider", choices=["cpu", "dml"], default="cpu", help="ONNX Runtime provider", )

    p.add_argument( "--store", type=Path, default=config.EVENTSTORE_T1_PATH, help="Override observation store root path", )
    p.add_argument( "--parquet-min-rows", type=int, default=None,
        help="Parquet writer: flush after this many buffered rows",
    )
    p.add_argument( "--parquet-min-bytes", type=int, default=None,
        help="Parquet writer: flush after approximately this many bytes",
    )

    return p.parse_args()


def main():
    args = parse_args()

    torch.set_num_threads(
        int(os.environ.get("OMP_NUM_THREADS", 2))
    )
    torch.set_num_interop_threads(1)

    logger.info(
        "[tier1] Thread config: OMP_NUM_THREADS=%s, "
        "torch.get_num_threads()=%d",
        os.environ.get("OMP_NUM_THREADS"),
        torch.get_num_threads(),
    )

    if args.backend == "pytorch":
        torch.set_num_threads(
            int(os.environ.get("OMP_NUM_THREADS", 2))
        )
        torch.set_num_interop_threads(1)

        logger.info(
            "[tier1] Thread config: OMP_NUM_THREADS=%s, "
            "torch.get_num_threads()=%d",
            os.environ.get("OMP_NUM_THREADS"),
            torch.get_num_threads(),
        )

    store_path = resolve_store_path(
        store_backend='parquet',
        masked=args.mask,
        store=args.store,
        shard=args.shard,
        num_shards=args.num_shards,
    )

    if args.clear:
        logger.info( "[tier1] Clearing Tier 1 output at %s", store_path )
        clear_output_dir(store_path)

    if args.backend == "onnx":
        from lib.macberth import load_macberth_onnx
        providers = (
            ["DmlExecutionProvider", "CPUExecutionProvider"]
            if args.onnx_provider == "dml"
            else ["CPUExecutionProvider"]
        )
        mac = load_macberth_onnx( providers=providers, )

    else:
        from lib.macberth import load_macberth
        mac = load_macberth( use_qint8=False, )

    pipeline = EmbeddingPipeline(
        mac,
        mask_targets=args.mask,
        mask_only_position=args.mask_only_position,
        pooling_scope=args.pooling_scope,
        batch_size=args.batch_size,
    )

    conn = get_connection()

    proc = CorpusProcessor(
        conn,
        store_path,
        pipeline,
        shard=args.shard,
        num_shards=args.num_shards,
        parquet_min_rows=args.parquet_min_rows,
        parquet_min_bytes=args.parquet_min_bytes,
    )

    proc.process( doc_id=args.doc_id )

    # A shard is mergeable only after the complete document stream has
    # processed successfully. Presence of the store directory alone is
    # not sufficient because interrupted writes leave valid-looking stores.
    if args.num_shards > 1:
        (store_path / "_COMPLETE").touch()

    conn.close()

    logger.info(
        "[Tier 1 done] mode=%s store_backend=%s path=%s",
        "masked" if args.mask else "unmasked",
        'parquet',
        store_path,
    )


if __name__ == "__main__":
    main()
