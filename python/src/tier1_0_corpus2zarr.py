#!/usr/bin/env python

"""
tier1_corpus2zarr.py

Tier 1: Contextual Embedding Observation Construction

This module constructs the foundational event-space representation of a corpus.
It transforms a tokenised document stream into a set of contextual embedding
observations, each anchored to a stable corpus index and a sliding window
context.

Core abstraction
-----------------
The system defines a single atomic unit of analysis:

    Event = (token occurrence, windowed context, embedding vector)

Each event is uniquely identified and fully traceable to the source corpus
position.

Indexing invariants
-------------------
1. Postgres defines corpus truth
   - Each token has a stable (doc_id, token_idx) identity.

2. Token identity is immutable
   - token_idx always refers to the original corpus ordering and is never
     recomputed or inferred from window position.

3. Windowing is contextual only
   - Sliding windows provide transformer context but do not redefine corpus
     structure.

4. Event identity is fully grounded
   - event_id is a stable hash of:
         (doc_id, token_idx, window_start_token_idx, window_token_pos)

5. Window anchors are corpus-aligned
   - window_start_token_idx preserves the true corpus position of the window
     origin, decoupling embedding geometry from filtered buffer indices.

Pipeline behaviour
------------------
For each document:

1. Tokens are filtered (stopword and non-content removal)
2. Remaining tokens are accumulated in a document buffer
3. Overlapping sliding windows are constructed over the buffer
4. A transformer model produces contextual embeddings per window
5. Word-aligned embeddings are emitted as atomic events
6. Events are written to a Zarr-backed observation store

Outputs
-------
The resulting store contains:

- event_id: globally unique embedding observation identifier
- concept_id: stable lexical occurrence identity
- token_idx: corpus-aligned token position
- window_start / window_start_token_idx: window alignment in both buffer
  and corpus coordinate spaces
- window_token_pos: position of token within window
- emb_raw: contextual embedding vector

Design intent
-------------
Tier 1 does not perform semantic aggregation, clustering, or neighbourhood
analysis. It exists solely to produce a stable, fully-referencable event
space that higher tiers (e.g. FAISS neighbourhood search, UMAP projection,
semantic drift analysis) can operate over without ambiguity.

This separation ensures that all downstream geometric or statistical
behaviour can be traced back to a deterministic corpus coordinate system,
avoiding conflation of windowing artefacts with semantic structure.

Risks
-----
Subword tokenization alignment: The code relies on word_ids from the tokenizer correctly mapping back to the filtered buffer indices. MacBERTh (BERT-based) should be stable here, but if you ever change tokenizers or preprocessing, this is a breakage point.

Overlapping windows + same token: Each contextual occurrence gets its own event_id (correct), but concept_id stays the same (also correct). Confirm downstream tiers (Tier 2+) expect this duality.

Updates WIP
-----------
This module now generates multiple contextual embeddings per token occurrence
using different window sizes to capture semantic meaning at multiple scales:

- local (384 tokens):  fine-grained syntactic and immediate context
- medium (512 tokens): standard paragraph-level context (original default)
- broad (1024 tokens):  larger discourse / argumentative context

Each event stores three separate embedding vectors. Downstream tiers can
use them individually or via an ensemble (weighted average or concatenation).

This design significantly improves representation of abstract historical
concepts (liberty, fanaticism, sedition, etc.) whose meaning often depends
on broader rhetorical context.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import torch
import xxhash
import unicodedata

from lib.eebo_db import get_connection
from lib.eebo_logging import logger
from lib.eebo_config import ZARR_PATH, EMBED_BATCH_SIZE
from lib.zarr_embedding_observation_store import ZarrEmbeddingObservationStore
from lib.macberth import load_macberth


WINDOW_CONFIGS = [
    {"name": "local",  "size": 256,  "stride": 128},   # finer grain
    {"name": "medium", "size": 512,  "stride": 256},   # original
    {"name": "broad",  "size": 512,  "stride": 256},   # same size as medium but different stride for more overlap
]


STOPWORDS = {
    "the", "a", "an",
    "i", "he", "she", "it", "we", "they", "who", "which", "that",
    "his", "her", "its", "our", "their", "my", "thy", "your", "hast",
    "art", "is", "who", "thine", "mine", "him", "them", "us", "me",
    "and", "or", "but", "of", "in", "to", "for", "with", "by",
    "at", "from", "as", "on", "into", "upon", "unto", "not",
    "nor", "yet", "so", "if", "be", "are", "was", "were",
    "shall", "will", "may", "should", "would", "could",
    "than", "then", "when",
    "this", "these", "those", "all", "no", "any", "such", "many", "some",
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
    """
    Represents one contextual embedding observation at a specific scale.

    Each token occurrence now generates multiple events (one per window config).
    """
    event_id: np.int64
    concept_id: np.int64
    doc_id: str
    corpus_token_idx: int          # primary corpus position
    window_start: int
    window_start_token_idx: int
    window_token_pos: int
    token: str
    vector_id: int
    vec: np.ndarray
    config_name: str               # "local", "medium", or "broad"

    @property
    def token_idx(self):
        """Legacy compatibility"""
        return self.corpus_token_idx

    @staticmethod
    def make(
        doc_id: str,
        corpus_token_idx: int,
        window_start_token_idx: int,
        window_start: int,
        window_token_pos: int,
        token: str,
        vector_id: int,
        vec: np.ndarray,
        config_name: str = "medium",   # NEW
    ) -> Event:
        """
        Factory method to create an Event with proper hashing.
        """
        concept_id = stable_hash(f"{doc_id}:{corpus_token_idx}")

        event_id = stable_hash(
            f"{doc_id}:{corpus_token_idx}:{window_start_token_idx}:{window_token_pos}:{config_name}"
        )

        return Event(
            event_id=event_id,
            concept_id=concept_id,
            doc_id=doc_id,
            corpus_token_idx=corpus_token_idx,
            window_start=window_start,
            window_start_token_idx=window_start_token_idx,
            window_token_pos=window_token_pos,
            token=token,
            vector_id=vector_id,
            vec=vec.astype(np.float32),
            config_name=config_name,
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
    """
    Produces multi-scale contextual embeddings using MacBERTh.
    """
    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.configs = WINDOW_CONFIGS

    # def embed_doc(self, buf: DocBuffer) -> list[Event]:
    #     input_ids, attention_mask, word_ids = self._encode(buf.tokens)

    #     events = []
    #     batch = []

    #     for window_start, ids, mask, wids in self._iter_windows(
    #         input_ids, attention_mask, word_ids
    #     ):
    #         batch.append({
    #             "input_ids": ids,
    #             "mask": mask,
    #             "word_ids": wids,
    #             "window_start": window_start,
    #         })

    #         if len(batch) >= EMBED_BATCH_SIZE:
    #             events.extend(self._flush_batch(buf, batch))
    #             batch.clear()

    #     if batch:
    #         events.extend(self._flush_batch(buf, batch))
    #     return events

    def embed_doc(self, buf: DocBuffer) -> list[Event]:
        """
        Generate multi-scale contextual embeddings for the document.
        Runs separate windowing passes for each context size.
        """
        input_ids, attention_mask, word_ids = self._encode(buf.tokens)
        all_events = []

        for config in self.configs:
            logger.debug(f"Processing {config['name']} windows (size={config['size']}) for doc {buf.doc_id}")

            for window_start, ids, mask, wids in self._iter_windows_config(
                input_ids, attention_mask, word_ids, config["size"], config["stride"]
            ):
                hidden = self._forward_single_window(ids, mask)

                events = self._extract_events(
                    buf,
                    {"window_start": window_start, "word_ids": wids},  # item dict
                    hidden,
                    config["name"]
                )
                all_events.extend(events)

        return all_events

    @staticmethod
    def _iter_windows_config(input_ids, attention_mask, word_ids, window_size, stride):
        """Yield windows for a specific size/stride combination."""
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

            yield (
                start_word,
                input_ids[encoded_start:encoded_end],
                attention_mask[encoded_start:encoded_end],
                word_ids[encoded_start:encoded_end]
            )

            if encoded_end == n:
                break
            start_word += stride


    def _forward_single_window(self, ids, mask):
        """Forward pass for one window."""
        input_ids = torch.tensor([ids], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([mask], dtype=torch.long).to(self.device)

        with torch.inference_mode():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )

        return out.last_hidden_state[0].cpu().numpy()


    @staticmethod
    def _extract_events(
        buf: DocBuffer,
        item: dict,
        hidden: np.ndarray,
        config_name: str
    ) -> list[Event]:
        """
        Extract Event objects for all tokens in this window for a given config scale.

        Args:
            buf: Document buffer containing tokens and metadata
            item: Window item with window_start, word_ids, etc.
            hidden: Hidden states from the model for this window (numpy array)
            config_name: Which context scale was used ("local", "medium", "broad")
        """
        window_start = item["window_start"]

        # Robust guard
        if window_start >= len(buf.corpus_token_idxs):
            logger.warning(
                f"Window start {window_start} out of range "
                f"(buffer has {len(buf.corpus_token_idxs)} tokens) for doc {buf.doc_id}"
            )
            return []

        window_start_token_idx = buf.corpus_token_idxs[window_start]

        events = []
        for i, wid in enumerate(item["word_ids"]):
            if wid is None or wid < 0 or wid >= len(buf.corpus_token_idxs):
                continue

            events.append(
                Event.make(
                    doc_id=buf.doc_id,
                    corpus_token_idx=buf.corpus_token_idxs[wid],
                    window_start_token_idx=window_start_token_idx,
                    window_start=window_start,
                    window_token_pos=i,
                    token=buf.tokens[wid],
                    vector_id=buf.vector_ids[wid],
                    vec=hidden[i],
                    config_name=config_name,          # ← Critical
                )
            )
        return events

    def _encode(self, tokens):
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
    def _iter_windows(input_ids, attention_mask, word_ids):
        """Yield windows aligned to original word tokens (buffer indices)."""
        n = len(input_ids)
        if n == 0:
            return

        # Number of original words in this document
        valid_word_ids = [wid for wid in word_ids if wid is not None and wid >= 0]
        if not valid_word_ids:
            return
        n_words = max(valid_word_ids) + 1

        start_word = 0
        while start_word < n_words:
            # Find the first subtoken position for this word
            try:
                encoded_start = next(
                    i for i, wid in enumerate(word_ids)
                    if wid == start_word
                )
            except StopIteration:
                break

            # End in encoded (subtoken) space
            encoded_end = min(encoded_start + WINDOW_SIZE, n)

            yield (
                start_word,                                   # window_start in word space
                input_ids[encoded_start:encoded_end],
                attention_mask[encoded_start:encoded_end],
                word_ids[encoded_start:encoded_end]
            )

            if encoded_end == n:
                break
            start_word += STRIDE


    def _forward(self, batch):
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
            out = self.model(input_ids=input_ids, attention_mask=mask, return_dict=True)

        return out.last_hidden_state.cpu().numpy()


    def _flush_batch(self, buf, batch):
        hidden_states = self._forward(batch)

        events = []
        for item, hidden in zip(batch, hidden_states):
            events.extend(self._extract_events(buf, item, hidden))
        return events


class CorpusProcessor:
    def __init__(self, conn, pipeline, report_every: int = 100):
        self.conn = conn
        self.pipeline = pipeline
        self.report_every = report_every

    def process(self, doc_id=None):
        store = ZarrEmbeddingObservationStore(
            path = str(ZARR_PATH),
            dim  = self.pipeline.model.config.hidden_size,
        )

        already_processed = store.get_doc_ids()

        cur = self.conn.cursor(name="tier1_cursor")
        cur.itersize = 10000

        if doc_id:
            cur.execute(
                """
                SELECT doc_id, token_idx, vector_id, token
                FROM pamphlet_tokens
                WHERE doc_id = %s
                ORDER BY token_idx
                """,
                (doc_id,),
            )
        else:
            cur.execute(
                """
                SELECT doc_id, token_idx, vector_id, token
                FROM pamphlet_tokens
                ORDER BY doc_id, token_idx
                """
            )

        buf = None
        docs_processed = 0

        for row_doc_id, token_idx, vid, token in cur:
            if row_doc_id in already_processed:
                continue

            if buf is None or row_doc_id != buf.doc_id:
                if buf:
                    self._flush(buf, store)
                    docs_processed += 1

                buf = DocBuffer(doc_id=row_doc_id)

            if is_content_token(token):
                buf.append(token, vid, token_idx)

        if buf and buf.doc_id not in already_processed:
            self._flush(buf, store)

    #
    def _flush(self, buf, store):
        """Generate multi-scale embeddings and append to store."""
        raw_events = self.pipeline.embed_doc(buf)
        if not raw_events:
            return

        from collections import defaultdict
        events_by_token = defaultdict(dict)   # key -> config_name -> event

        for e in raw_events:
            key = (e.corpus_token_idx, e.window_token_pos)
            events_by_token[key][e.config_name] = e

        # Build aligned arrays
        event_ids = []
        concept_ids = []
        emb_local = []
        emb_medium = []
        emb_broad = []
        vector_ids = []
        doc_ids = []
        token_idxs = []
        tokens = []
        window_ids = []
        window_token_poss = []

        for key, config_dict in events_by_token.items():
            if len(config_dict) < 2:   # at least 2 scales
                continue

            # Use medium if available, else any
            canonical = config_dict.get("medium") or list(config_dict.values())[0]

            event_ids.append(canonical.event_id)
            concept_ids.append(canonical.concept_id)

            # Get embeddings with fallback to medium if missing
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
            logger.warning(f"No valid multi-scale events for doc {buf.doc_id}")
            return

        logger.info(f"Doc {buf.doc_id}: {len(event_ids):,} tokens with multi-scale embeddings")

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



def clear_output_dir():
    path = ZARR_PATH
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clear",        action="store_true",
                   help="Wipe existing store before processing")
    p.add_argument("--doc-id",       type=str, default=None,
                   help="Embed a single document by doc_id and append to store")
    p.add_argument("--report-every", type=int, default=100,
                   help="Log progress every N documents (default: 100)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.doc_id:
        if args.clear:
            logger.warning("--clear with --doc-id will wipe the store before adding one document")
    elif args.clear:
        logger.info("Clearing Tier 1 output")
        clear_output_dir()

    conn     = get_connection()
    mac      = load_macberth()
    pipeline = EmbeddingPipeline(mac.tokenizer, mac.model, mac.device)
    proc     = CorpusProcessor(conn, pipeline, report_every=args.report_every)

    proc.process(doc_id=args.doc_id)

    conn.close()
    logger.info("[Tier1 done]")


if __name__ == "__main__":
    main()
