#!/usr/bin/env python
"""
tier1_corpus2zarr.py

Contextual Observation Log (Tier 1)

Core invariant
--------------
- Postgres defines corpus identity
- concept_id defines stable lexical occurrence identity
- event_id defines contextual embedding observation identity
- Later, FAISS indexes event_id space ONLY

Each document is embedded once per WindowStrategy in WINDOW_STRATEGIES.
Each strategy writes to its own shard path resolved by ShardResolver:

    <ZARR_ROOT>/<corpus_id>/<period>/<model>/<strategy_tag>/

Check the minimal STOPWORDS list.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass, field

import numpy as np
import torch
import xxhash
import unicodedata

from lib.eebo_db import get_connection
from lib.eebo_logging import logger
from lib.eebo_config import ZARR_ROOT, EMBED_BATCH_SIZE
from lib.zarr_embedding_observation_store import ZarrEmbeddingObservationStore
from lib.macberth import load_macberth
from lib.shard_resolver import ShardResolver
from lib.window_strategy import WindowStrategy, WINDOW_STRATEGIES


# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

STOPWORDS = {
    "the", "a", "an",
    "i", "he", "she", "it", "we", "they", "who", "which", "that",
    "his", "her", "its", "our", "their", "my", "thy", "your", "hast",
    "art", "is", "who", "thine", "mine", "his", "her", "its",
    "him", "them", "us", "me", "thee", "thy", "thou", "thoust",
    "and", "or", "but", "of", "in", "to", "for", "with", "by",
    "at", "from", "as", "on", "into", "upon", "unto", "not",
    "nor", "yet", "so", "if", "be", "is", "are", "was", "were", "where",
    "have", "hath", "hath", "do", "doth",
    "shall", "will", "may", "should", "would", "could",
    "than", "then", "when",
    "this", "these", "those", "all", "no", "any", "such", "many", "some",
    "been", "s", "you", "had", "v", "what",
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def stable_hash(key: str) -> np.int64:
    h = xxhash.xxh64(key, seed=0).intdigest()
    return np.int64(h & 0x7FFFFFFFFFFFFFFF)


def is_content_token(token: str) -> bool:
    """Reject punctuation, whitespace, stopwords, and bare symbols.
    Early modern texts include many non-lexical characters."""
    stripped = token.strip().lower()
    if not stripped or stripped in STOPWORDS:
        return False
    if all(unicodedata.category(c).startswith(("P", "S", "Z")) for c in stripped):
        return False
    return True


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Event:
    event_id:         np.int64
    concept_id:       np.int64
    doc_id:           str
    token_idx:        int
    window_start:     int
    window_token_pos: int
    token:            str
    vector_id:        int
    vec:              np.ndarray

    @staticmethod
    def make(
        doc_id:           str,
        corpus_token_idx: int,
        window_start:     int,
        window_token_pos: int,
        token:            str,
        vector_id:        int,
        vec:              np.ndarray,
    ) -> "Event":
        concept_id = stable_hash(f"{doc_id}:{corpus_token_idx}")
        event_id   = stable_hash(
            f"{doc_id}:{corpus_token_idx}:{window_start}:{window_token_pos}"
        )
        return Event(
            event_id         = event_id,
            concept_id       = concept_id,
            doc_id           = doc_id,
            token_idx        = corpus_token_idx,
            window_start     = window_start,
            window_token_pos = window_token_pos,
            token            = token,
            vector_id        = vector_id,
            vec              = vec.astype(np.float32),
        )


# ---------------------------------------------------------------------------
# DocBuffer
# Now carries corpus_id and pub_year so _flush can route to the correct shard.
# unit_id will carry paragraph/sentence boundary ids once TEI parsing is ready.
# ---------------------------------------------------------------------------

@dataclass
class DocBuffer:
    """Accumulates content tokens for one document before embedding."""
    doc_id:     str
    corpus_id:  str
    pub_year:   int | None

    tokens:     list = field(default_factory=list)
    vector_ids: list = field(default_factory=list)
    token_idxs: list = field(default_factory=list)

    # Future: populated from TEI structure once para/sentence parsing is ready.
    # Each entry is a unit_id (int) identifying which paragraph/sentence the
    # token belongs to.  None = not yet available.
    unit_ids:   list | None = None

    def append(self, token: str, vector_id: int, token_idx: int) -> None:
        self.tokens.append(token)
        self.vector_ids.append(vector_id)
        self.token_idxs.append(token_idx)

    def __bool__(self) -> bool:
        return bool(self.tokens)


# ---------------------------------------------------------------------------
# EmbeddingPipeline
# ---------------------------------------------------------------------------

class EmbeddingPipeline:
    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model     = model
        self.device    = device

    # ------------------------------------------------------------------
    # Public dispatch — routes to the correct strategy implementation
    # ------------------------------------------------------------------

    def embed_doc(self, buf: DocBuffer, strategy: WindowStrategy) -> list[Event]:
        """Embed a document buffer according to the given WindowStrategy."""
        if strategy.name == "sliding":
            return self._embed_sliding(buf, strategy.size, strategy.stride)
        elif strategy.name == "doc":
            return self._embed_doc_level(buf)
        elif strategy.name == "paragraph":
            return self._embed_paragraph_level(buf)
        elif strategy.name == "sentence":
            return self._embed_sentence_level(buf)
        else:
            raise ValueError(f"Unknown strategy: {strategy.name!r}")

    # ------------------------------------------------------------------
    # Sliding window (original logic, now accepts size/stride explicitly)
    # ------------------------------------------------------------------

    def _embed_sliding(
        self,
        buf:    DocBuffer,
        size:   int,
        stride: int,
    ) -> list[Event]:
        input_ids, attention_mask, word_ids = self._encode(buf.tokens)

        events = []
        batch  = []

        for window_start, ids, mask, wids in self._iter_windows(
            input_ids, attention_mask, word_ids, size, stride
        ):
            batch.append({
                "input_ids":    ids,
                "mask":         mask,
                "word_ids":     wids,
                "window_start": window_start,
            })
            if len(batch) >= EMBED_BATCH_SIZE:
                events.extend(self._flush_batch(buf, batch))
                batch.clear()

        if batch:
            events.extend(self._flush_batch(buf, batch))

        return events

    # ------------------------------------------------------------------
    # Doc-level: one embedding for the entire document (truncated to
    # the model's max sequence length if necessary).
    # ------------------------------------------------------------------

    def _embed_doc_level(self, buf: DocBuffer) -> list[Event]:
        input_ids, attention_mask, word_ids = self._encode(buf.tokens)

        # Truncate to model max — doc embedding is a single forward pass
        max_len       = self.model.config.max_position_embeddings
        input_ids     = input_ids[:max_len]
        attention_mask = attention_mask[:max_len]
        word_ids      = word_ids[:max_len]

        batch = [{
            "input_ids":    input_ids,
            "mask":         attention_mask,
            "word_ids":     word_ids,
            "window_start": 0,
        }]
        return self._flush_batch(buf, batch)

    # ------------------------------------------------------------------
    # Paragraph-level: one embedding per paragraph.
    # Requires buf.unit_ids to be populated with paragraph boundary ids.
    # Falls back gracefully to doc-level if unit_ids are absent.
    # ------------------------------------------------------------------

    def _embed_paragraph_level(self, buf: DocBuffer) -> list[Event]:
        if not buf.unit_ids:
            logger.warning(
                f"[PARA] {buf.doc_id}: unit_ids not set — "
                "falling back to doc-level embedding"
            )
            return self._embed_doc_level(buf)

        return self._embed_by_unit(buf)

    # ------------------------------------------------------------------
    # Sentence-level: one embedding per sentence.
    # Requires buf.unit_ids to be populated with sentence boundary ids.
    # Falls back gracefully to doc-level if unit_ids are absent.
    # ------------------------------------------------------------------

    def _embed_sentence_level(self, buf: DocBuffer) -> list[Event]:
        if not buf.unit_ids:
            logger.warning(
                f"[SENT] {buf.doc_id}: unit_ids not set — "
                "falling back to doc-level embedding"
            )
            return self._embed_doc_level(buf)

        return self._embed_by_unit(buf)

    # ------------------------------------------------------------------
    # Shared unit embedding logic (para + sentence)
    # ------------------------------------------------------------------

    def _embed_by_unit(self, buf: DocBuffer) -> list[Event]:
        """
        Group tokens by unit_id, encode each group independently,
        and produce one forward pass per unit.  Batched in groups of
        EMBED_BATCH_SIZE units.
        """
        # Group token indices by unit_id, preserving order
        from itertools import groupby

        unit_groups: dict[int, list[int]] = {}
        for pos, uid in enumerate(buf.unit_ids):
            unit_groups.setdefault(uid, []).append(pos)

        events = []
        batch  = []

        for unit_id, positions in unit_groups.items():
            unit_tokens = [buf.tokens[p]     for p in positions]
            unit_vidxs  = [buf.vector_ids[p] for p in positions]
            unit_tidxs  = [buf.token_idxs[p] for p in positions]

            input_ids, attention_mask, word_ids = self._encode(unit_tokens)

            # window_start = corpus token index of the first token in this unit
            window_start = unit_tidxs[0] if unit_tidxs else 0

            batch.append({
                "input_ids":    input_ids,
                "mask":         attention_mask,
                "word_ids":     word_ids,
                "window_start": window_start,
                # Carry unit-specific token metadata for event construction
                "_token_idxs":  unit_tidxs,
                "_vector_ids":  unit_vidxs,
                "_tokens":      unit_tokens,
            })

            if len(batch) >= EMBED_BATCH_SIZE:
                events.extend(self._flush_unit_batch(buf, batch))
                batch.clear()

        if batch:
            events.extend(self._flush_unit_batch(buf, batch))

        return events

    # ------------------------------------------------------------------
    # Encoding and forward pass
    # ------------------------------------------------------------------

    def _encode(self, tokens: list[str]):
        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=False,
            return_tensors="pt",
        )
        word_ids = enc.word_ids() or [None] * len(enc["input_ids"][0])
        return (
            enc["input_ids"][0].tolist(),
            enc["attention_mask"][0].tolist(),
            word_ids,
        )

    @staticmethod
    def _iter_windows(input_ids, attention_mask, word_ids, size, stride):
        n, start = len(input_ids), 0
        while start < n:
            end = min(start + size, n)
            yield (
                start,
                input_ids[start:end],
                attention_mask[start:end],
                word_ids[start:end],
            )
            if end == n:
                break
            start += stride

    def _flush_batch(self, buf: DocBuffer, batch: list) -> list[Event]:
        hidden_states = self._forward(batch)
        events = []
        for item, hidden in zip(batch, hidden_states):
            events.extend(self._extract_events(buf, item, hidden))
        return events

    def _flush_unit_batch(self, buf: DocBuffer, batch: list) -> list[Event]:
        """Forward pass for unit batches (para/sentence) which carry their
        own token metadata rather than indexing into buf directly."""
        hidden_states = self._forward(batch)
        events = []
        for item, hidden in zip(batch, hidden_states):
            events.extend(self._extract_unit_events(item, hidden))
        return events

    def _forward(self, batch: list) -> np.ndarray:
        max_len = max(len(x["input_ids"]) for x in batch)

        def pad(seq):
            return seq + [0] * (max_len - len(seq))

        input_ids = torch.tensor(
            [pad(x["input_ids"]) for x in batch], dtype=torch.long
        ).to(self.device)

        mask = torch.tensor(
            [pad(x["mask"]) for x in batch], dtype=torch.long
        ).to(self.device)

        with torch.inference_mode():
            out = self.model(
                input_ids=input_ids,
                attention_mask=mask,
                return_dict=True,
            )

        return out.last_hidden_state.cpu().numpy()

    @staticmethod
    def _extract_events(
        buf:    DocBuffer,
        item:   dict,
        hidden: np.ndarray,
    ) -> list[Event]:
        window_start = item["window_start"]
        return [
            Event.make(
                doc_id           = buf.doc_id,
                corpus_token_idx = buf.token_idxs[wid],
                window_start     = window_start,
                window_token_pos = i,
                token            = buf.tokens[wid],
                vector_id        = buf.vector_ids[wid],
                vec              = hidden[i],
            )
            for i, wid in enumerate(item["word_ids"])
            if wid is not None and wid >= 0
        ]

    @staticmethod
    def _extract_unit_events(item: dict, hidden: np.ndarray) -> list[Event]:
        """Extract events for a unit batch item, using per-item token metadata."""
        window_start = item["window_start"]
        doc_id       = item.get("_doc_id", "")
        token_idxs   = item["_token_idxs"]
        vector_ids   = item["_vector_ids"]
        tokens       = item["_tokens"]

        return [
            Event.make(
                doc_id           = doc_id,
                corpus_token_idx = token_idxs[wid],
                window_start     = window_start,
                window_token_pos = i,
                token            = tokens[wid],
                vector_id        = vector_ids[wid],
                vec              = hidden[i],
            )
            for i, wid in enumerate(item["word_ids"])
            if wid is not None and wid >= 0
        ]


# ---------------------------------------------------------------------------
# CorpusProcessor
# ---------------------------------------------------------------------------

class CorpusProcessor:
    def __init__(
        self,
        conn,
        pipeline:     EmbeddingPipeline,
        resolver:     ShardResolver,
        report_every: int = 100,
    ):
        self.conn         = conn
        self.pipeline     = pipeline
        self.resolver     = resolver
        self.report_every = report_every

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def process(self, doc_id: str | None = None):
        # Build a combined set of already-processed doc_ids across all
        # existing shards so we don't re-embed docs for any strategy.
        already_processed = self._get_all_processed_doc_ids()

        if doc_id is not None:
            if doc_id in already_processed:
                logger.info(f"[SKIP] {doc_id} already in store")
                return
            self._process_query(already_processed, doc_id=doc_id)
        else:
            self._process_query(already_processed)

    # ------------------------------------------------------------------
    # Query and row iteration
    # ------------------------------------------------------------------

    def _process_query(
        self,
        already_processed: set[str],
        doc_id: str | None = None,
    ):
        label = doc_id or "full corpus"
        logger.info(f"[START] {label}")

        cur = self.conn.cursor(name="tier1_cursor")
        cur.itersize = 10000

        if doc_id is not None:
            cur.execute("""
                SELECT pt.doc_id,
                       pt.token_idx,
                       pt.vector_id,
                       pt.token,
                       pd.corpus_id,
                       pd.pub_year
                FROM   pamphlet_tokens pt
                JOIN   pamphlet_docs   pd USING (doc_id)
                WHERE  pt.doc_id = %s
                ORDER  BY pt.token_idx
            """, (doc_id,))
        else:
            cur.execute("""
                SELECT pt.doc_id,
                       pt.token_idx,
                       pt.vector_id,
                       pt.token,
                       pd.corpus_id,
                       pd.pub_year
                FROM   pamphlet_tokens pt
                JOIN   pamphlet_docs   pd USING (doc_id)
                ORDER  BY pt.doc_id, pt.token_idx
            """)

        buf:           DocBuffer | None = None
        docs_processed = 0

        for row_doc_id, token_idx, vid, token, corpus_id, pub_year in cur:
            if row_doc_id in already_processed:
                continue

            if buf is None or row_doc_id != buf.doc_id:
                if buf:
                    self._flush(buf)
                    already_processed.add(buf.doc_id)
                    docs_processed += 1
                    if docs_processed % self.report_every == 0:
                        logger.info(
                            f"[PROGRESS] {docs_processed} documents processed"
                        )
                buf = DocBuffer(
                    doc_id    = row_doc_id,
                    corpus_id = corpus_id,
                    pub_year  = pub_year,
                )

            if is_content_token(token):
                buf.append(token, vid, token_idx)

        if buf and buf.doc_id not in already_processed:
            self._flush(buf)
            docs_processed += 1

        logger.info(
            f"[COMPLETE] {label} — {docs_processed} documents processed"
        )

    # ------------------------------------------------------------------
    # Flush: embed once per strategy and write to the correct shard
    # ------------------------------------------------------------------

    def _flush(self, buf: DocBuffer) -> None:
        if not buf:
            return

        for strategy in WINDOW_STRATEGIES:
            events = self.pipeline.embed_doc(buf, strategy)
            if not events:
                logger.warning(
                    f"[EMPTY] {buf.doc_id} produced no events "
                    f"for strategy {strategy.tag}"
                )
                continue

            store = self._get_store(buf.corpus_id, buf.pub_year, strategy)
            self._write_events(store, events)

    def _write_events(
        self,
        store:  ZarrEmbeddingObservationStore,
        events: list[Event],
    ) -> None:
        (
            event_ids, concept_ids, doc_ids, token_idxs,
            window_starts, window_token_pos, tokens, vector_ids, vecs,
        ) = zip(*[
            (
                e.event_id, e.concept_id, e.doc_id, e.token_idx,
                e.window_start, e.window_token_pos, e.token,
                e.vector_id, e.vec,
            )
            for e in events
        ])

        store.append_events(
            event_id         = np.asarray(event_ids,        dtype=np.int64),
            concept_id       = np.asarray(concept_ids,      dtype=np.int64),
            emb_raw          = np.stack(vecs),
            vector_id        = np.asarray(vector_ids,       dtype=np.int64),
            doc_id           = np.asarray(doc_ids,          dtype="U32"),
            token_idx        = np.asarray(token_idxs,       dtype=np.int32),
            window_id        = np.asarray(window_starts,    dtype=np.int32),
            window_token_pos = np.asarray(window_token_pos, dtype=np.int32),
            token            = np.asarray(tokens,           dtype=object),
        )

    # ------------------------------------------------------------------
    # Store / shard helpers
    # ------------------------------------------------------------------

    def _get_store(
        self,
        corpus_id: str,
        pub_year:  int | None,
        strategy:  WindowStrategy,
    ) -> ZarrEmbeddingObservationStore:
        path = self.resolver.resolve(corpus_id, pub_year, strategy)
        return ZarrEmbeddingObservationStore(
            path = str(path),
            dim  = self.pipeline.model.config.hidden_size,
        )

    def _get_all_processed_doc_ids(self) -> set[str]:
        """
        Union of doc_ids already written across all existing shards.
        Prevents re-embedding a document for any strategy if it was
        previously completed.
        """
        processed: set[str] = set()
        for shard_path in self.resolver.all_shards():
            try:
                store = ZarrEmbeddingObservationStore(
                    path = str(shard_path),
                    dim  = self.pipeline.model.config.hidden_size,
                )
                processed |= store.get_doc_ids()
            except Exception as exc:
                logger.warning(
                    f"[WARN] Could not read doc_ids from {shard_path}: {exc}"
                )
        return processed


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def clear_output_dir():
    path = ZARR_ROOT / "tier1"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--clear",
        action="store_true",
        help="Wipe entire ZARR_ROOT before processing",
    )
    p.add_argument(
        "--doc-id",
        type=str,
        default=None,
        help="Embed a single document by doc_id and append to store",
    )
    p.add_argument(
        "--report-every",
        type=int,
        default=100,
        help="Log progress every N documents (default: 100)",
    )
    p.add_argument(
        "--model",
        type=str,
        default="MacBERTh",
        help="Model name used as directory layer in shard path (default: MacBERTh)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.doc_id and args.clear:
        logger.warning(
            "--clear with --doc-id will wipe the entire store "
            "before adding one document"
        )
    elif args.clear:
        logger.info("Clearing Zarr root")
        if ZARR_ROOT.exists():
            shutil.rmtree(ZARR_ROOT)
        ZARR_ROOT.mkdir(parents=True, exist_ok=True)

    conn     = get_connection()
    mac      = load_macberth()
    pipeline = EmbeddingPipeline(mac.tokenizer, mac.model, mac.device)
    resolver = ShardResolver(model_name=args.model)
    proc     = CorpusProcessor(
        conn,
        pipeline,
        resolver,
        report_every=args.report_every,
    )

    proc.process(doc_id=args.doc_id)
    conn.close()


if __name__ == "__main__":
    main()
