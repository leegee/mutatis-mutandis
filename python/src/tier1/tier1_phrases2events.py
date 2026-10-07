# tier1/tier1_phrases2events.py

from __future__ import annotations

import argparse
import os
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch

from lib.corpus_config import CONCEPT_SETS, PHRASE_SETS, EMBED_BATCH_SIZE, LANCE_INDEXES_DIR
from lib.corpus_db import get_connection
from lib.corpus_logging import logger
from lib.macberth import load_macberth
from lib.stopwords_min import STOPWORDS
from tier1.db_observation_backend import allocate_event_ids, insert_events, create_events_table

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")


WINDOW_CONFIGS = (
    {"name": "local", "size": 256, "stride": 128},
    {"name": "medium", "size": 512, "stride": 256},
    {"name": "broad", "size": 512, "stride": 384},
)
SCALE_NAMES = tuple(c["name"] for c in WINDOW_CONFIGS)

ACTIVE_SCALES = ("local", "medium")

LANCE_MODEL_NAME = "macberth"
LANCE_BUCKET_SIZE = 50


def normalise_token(token: str) -> str:
    return unicodedata.normalize("NFKC", token).strip().lower()


def is_punctuation(token: str) -> bool:
    value = normalise_token(token)
    return bool(value) and all(
        unicodedata.category(char).startswith("P")
        for char in value
    )


def is_stopword(token: str) -> bool:
    return normalise_token(token) in STOPWORDS


def is_storable_event(token: str) -> bool:
    return not is_stopword(token) and not is_punctuation(token)


def seed_forms() -> set[str]:
    forms: set[str] = set()
    for rule in CONCEPT_SETS.values():
        forms.update(normalise_token(form) for form in rule["forms"])
    return forms


def phrase_forms() -> list[tuple[str, ...]]:
    forms = []
    for rule in PHRASE_SETS.values():
        for seq in rule["forms"]:
            forms.append(tuple(normalise_token(t) for t in seq))
    return forms


def false_positive_forms() -> set[str]:
    forms: set[str] = set()
    for rule in CONCEPT_SETS.values():
        forms.update(normalise_token(form) for form in rule["false_positives"])
    return forms


SEED_FORMS = seed_forms()
FALSE_POSITIVE_FORMS = false_positive_forms()
PHRASE_FORMS = phrase_forms()


def is_seed(token: str) -> bool:
    value = normalise_token(token)
    return value in SEED_FORMS and value not in FALSE_POSITIVE_FORMS


@dataclass(slots=True)
class TokenRow:
    corpus: str
    doc_id: str
    token_idx: int
    token: str
    pub_year: int | None


@dataclass(slots=True)
class EmbeddedVector:
    vector: np.ndarray
    window_id: int
    window_token_pos: int


@dataclass(slots=True)
class Observation:
    event_id: int | None
    corpus: str
    doc_id: str
    token: str
    token_idx: int
    pub_year: int | None
    local_window_id: int | None
    local_window_token_pos: int | None
    medium_window_id: int | None
    medium_window_token_pos: int | None
    broad_window_id: int | None
    broad_window_token_pos: int | None


@dataclass(slots=True)
class SpanObservation:
    """One multi-token phrase occurrence."""
    event_id: int | None
    corpus: str
    doc_id: str
    token: str                  # canonical form of the whole phrase
    token_idx: int              # start index
    span_end_idx: int           # inclusive end index
    pub_year: int | None
    local_window_id: int | None
    local_window_token_pos: int | None
    medium_window_id: int | None
    medium_window_token_pos: int | None
    broad_window_id: int | None
    broad_window_token_pos: int | None


@dataclass(slots=True)
class EmbeddedObservation:
    observation: Observation
    vectors: dict[str, np.ndarray]


@dataclass(slots=True)
class EmbeddedSpanObservation:
    observation: SpanObservation
    vectors: dict[str, np.ndarray]


AnyEmbedded = Union[EmbeddedObservation, EmbeddedSpanObservation]


@dataclass(slots=True)
class DocBuffer:
    corpus: str
    doc_id: str
    pub_year: int | None
    rows: list[TokenRow]

    @property
    def tokens(self) -> list[str]:
        return [row.token for row in self.rows]

    def __bool__(self) -> bool:
        return bool(self.rows)


def existing_event_documents(conn, *, corpus: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT e.doc_id
            FROM events AS e
            WHERE e.corpus = %s
            """,
            (corpus,),
        )
        return {row[0] for row in cur.fetchall()}


class MacBERThPipeline:
    def __init__(
        self,
        mac,
        *,
        batch_size: int = EMBED_BATCH_SIZE,
        mask_targets: bool = False,
    ) -> None:
        self.mac = mac
        self.tokenizer = mac.tokenizer
        self.model = mac.model
        self.device = mac.device
        self.batch_size = batch_size
        self.mask_targets = mask_targets

    def embed(
        self,
        document: DocBuffer,
        target_positions: set[int],
        scales: tuple[str, ...] = ACTIVE_SCALES,
    ) -> dict[int, dict[str, EmbeddedVector]]:
        if not target_positions:
            return {}

        results: dict[int, dict[str, EmbeddedVector]] = {
            position: {} for position in target_positions
        }

        encoded = self.tokenizer(
            document.tokens,
            is_split_into_words=True,
            truncation=False,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"][0].tolist()
        attention_mask = encoded["attention_mask"][0].tolist()
        word_ids = encoded.word_ids()

        if word_ids is None:
            raise RuntimeError(
                "MacBERTh tokenizer did not return word_ids; "
                "cannot align token occurrences to hidden states."
            )

        for config in WINDOW_CONFIGS:
            if config["name"] not in scales:
                continue

            jobs = self._make_jobs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                word_ids=word_ids,
                target_positions=target_positions,
                window_size=config["size"],
                stride=config["stride"],
            )

            for offset in range(0, len(jobs), self.batch_size):
                batch = jobs[offset : offset + self.batch_size]
                hidden = self._forward(batch)

                for job, vectors in zip(batch, hidden):
                    for target, vector in zip(job["targets"], vectors):
                        word_position = target["word_position"]
                        if word_position not in results:
                            raise RuntimeError(
                                "Embedding returned a target position "
                                f"that was not requested: {word_position}"
                            )
                        results[word_position][config["name"]] = EmbeddedVector(
                            vector=vector,
                            window_id=job["window_id"],
                            window_token_pos=target["encoded_position"],
                        )

        missing = []
        for position in sorted(target_positions):
            missing_scales = [s for s in scales if s not in results[position]]
            if missing_scales:
                missing.append(
                    (position, document.rows[position].token, missing_scales)
                )

        if missing:
            logger.error("[tier1] incomplete embeddings: %d observations", len(missing))
            for position, token, scales in missing[:20]:
                logger.error(
                    "[tier1] position=%d token=%r missing=%s",
                    position, token, scales,
                )
            raise RuntimeError(
                f"{len(missing)} observations did not receive all active embeddings."
            )

        return results

    def embed_span(
        self,
        document: DocBuffer,
        start: int,
        end: int,          # inclusive
        scales: tuple[str, ...] = ACTIVE_SCALES,
    ) -> dict[str, EmbeddedVector]:
        """Embed a contiguous span by mean-pooling the token vectors inside it."""
        target_positions = set(range(start, end + 1))
        raw = self.embed(document, target_positions, scales=scales)

        pooled: dict[str, EmbeddedVector] = {}
        for scale in scales:
            vecs = [raw[p][scale].vector for p in range(start, end + 1)]
            mean_vec = np.mean(vecs, axis=0).astype(np.float32)
            first = raw[start][scale]
            pooled[scale] = EmbeddedVector(
                vector=mean_vec,
                window_id=first.window_id,
                window_token_pos=first.window_token_pos,
            )
        return pooled

    def _make_jobs(
        self,
        *,
        input_ids: list[int],
        attention_mask: list[int],
        word_ids: list[int | None],
        target_positions: set[int],
        window_size: int,
        stride: int,
    ) -> list[dict]:
        word_count = (
            max(word_id for word_id in word_ids if word_id is not None) + 1
        )

        word_spans: list[tuple[int, int]] = []
        current_word = None
        current_start = None

        for encoded_position, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id != current_word:
                if current_word is not None:
                    word_spans.append((current_start, encoded_position))
                current_word = word_id
                current_start = encoded_position

        if current_word is not None:
            word_spans.append((current_start, len(word_ids)))

        if len(word_spans) != word_count:
            raise RuntimeError(
                f"MacBERTh word alignment is incomplete: expected {word_count} "
                f"corpus tokens, got {len(word_spans)} encoded spans."
            )

        jobs: list[dict] = []
        covered_targets: set[int] = set()
        start_word = 0

        while start_word < word_count:
            end_word = min(word_count, start_word + window_size)
            candidate_targets = sorted(
                p for p in target_positions if start_word <= p < end_word
            )

            if candidate_targets:
                group: list[int] = []
                for target in candidate_targets:
                    if not group:
                        group.append(target)
                        continue
                    group_start = word_spans[group[0]][0]
                    group_end = word_spans[target][1]
                    if group_end - group_start <= 512:
                        group.append(target)
                    else:
                        self._append_job(
                            jobs=jobs,
                            covered_targets=covered_targets,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            word_ids=word_ids,
                            word_spans=word_spans,
                            target_positions=group,
                            context_start_word=start_word,
                            context_end_word=end_word,
                        )
                        group = [target]
                if group:
                    self._append_job(
                        jobs=jobs,
                        covered_targets=covered_targets,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        word_ids=word_ids,
                        word_spans=word_spans,
                        target_positions=group,
                        context_start_word=start_word,
                        context_end_word=end_word,
                    )

            if start_word + stride >= word_count:
                break
            start_word += stride

        missing_targets = target_positions - covered_targets
        if missing_targets:
            raise RuntimeError(
                "Some target observations were not assigned to a MacBERTh job: "
                f"{sorted(missing_targets)[:20]}"
            )
        return jobs

    def _append_job(
        self,
        *,
        jobs: list[dict],
        covered_targets: set[int],
        input_ids: list[int],
        attention_mask: list[int],
        word_ids: list[int | None],
        word_spans: list[tuple[int, int]],
        target_positions: list[int],
        context_start_word: int,
        context_end_word: int,
    ) -> None:
        target_start_word = target_positions[0]
        target_end_word = target_positions[-1] + 1

        context_start = word_spans[context_start_word][0]
        context_end = word_spans[context_end_word - 1][1]
        target_start = word_spans[target_start_word][0]
        target_end = word_spans[target_end_word - 1][1]
        target_span = target_end - target_start

        if target_span > 512:
            raise RuntimeError(
                "A target group exceeds MacBERTh's 512-position limit: "
                f"targets={target_start_word}:{target_end_word}, "
                f"encoded_length={target_span}"
            )

        available_length = context_end - context_start
        if available_length > 512:
            desired_start = target_start - (512 - target_span) // 2
            encoded_start = max(context_start, desired_start)
            encoded_end = min(context_end, encoded_start + 512)
            if encoded_end - encoded_start < 512:
                encoded_start = max(context_start, encoded_end - 512)
        else:
            encoded_start = context_start
            encoded_end = context_end

        if not (encoded_start <= target_start and target_end <= encoded_end):
            raise RuntimeError(
                "Constructed MacBERTh context does not contain all targets: "
                f"targets={target_start_word}:{target_end_word}, "
                f"context={context_start_word}:{context_end_word}"
            )

        relative_word_ids = word_ids[encoded_start:encoded_end]
        window_ids = input_ids[encoded_start:encoded_end].copy()
        window_mask = attention_mask[encoded_start:encoded_end]

        target_positions_in_window = []
        for word_position in target_positions:
            try:
                relative = relative_word_ids.index(word_position)
            except ValueError as exc:
                raise RuntimeError(
                    f"Target disappeared from its MacBERTh window: "
                    f"word_position={word_position}"
                ) from exc

            target_positions_in_window.append(
                {"word_position": word_position, "encoded_position": relative}
            )

            if self.mask_targets:
                mask_token_id = self.tokenizer.mask_token_id
                if mask_token_id is None:
                    raise RuntimeError("MacBERTh tokenizer has no mask token.")
                for i, wid in enumerate(relative_word_ids):
                    if wid == word_position:
                        window_ids[i] = mask_token_id

        jobs.append(
            {
                "input_ids": window_ids,
                "attention_mask": window_mask,
                "window_id": context_start_word,
                "targets": target_positions_in_window,
            }
        )
        covered_targets.update(target_positions)

    def _forward(self, jobs: list[dict]) -> list[list[np.ndarray]]:
        if not jobs:
            return []

        max_length = max(len(job["input_ids"]) for job in jobs)
        if max_length > 512:
            raise RuntimeError(f"Prepared MacBERTh batch exceeds 512 tokens: {max_length}")

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise RuntimeError("MacBERTh tokenizer has no pad token.")

        input_ids = []
        attention_masks = []
        for job in jobs:
            padding = max_length - len(job["input_ids"])
            input_ids.append(job["input_ids"] + [pad_token_id] * padding)
            attention_masks.append(job["attention_mask"] + [0] * padding)

        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        attention_tensor = torch.tensor(attention_masks, dtype=torch.long, device=self.device)

        with torch.inference_mode():
            output = self.mac.encode(
                input_ids=input_tensor,
                attention_mask=attention_tensor,
                return_dict=True,
            )

        hidden = output.last_hidden_state.cpu().numpy()
        return [
            [
                hidden[batch_index, target["encoded_position"]].astype(np.float32, copy=False)
                for target in job["targets"]
            ]
            for batch_index, job in enumerate(jobs)
        ]


class EventWriter:
    """
    Owns Postgres event provenance and delegates all vector I/O to one
    VectorWriter per active scale.  Accepts both token and span observations.
    """

    def __init__(self, conn, lance_root: Path) -> None:
        self.conn = conn
        self.lance_root = lance_root
        self.vector_writers: dict[str, VectorWriter] = {
            scale: VectorWriter(
                lance_root,
                scale=scale,
                model_name=LANCE_MODEL_NAME,
                bucket_size=LANCE_BUCKET_SIZE,
            )
            for scale in ACTIVE_SCALES
        }

    def purge_orphans(self, conn, *, year_range=None, apply=False):
        return {
            s: w.purge_orphans(conn, year_range=year_range, apply=apply)
            for s, w in self.vector_writers.items()
        }

    def write(self, observations: list[AnyEmbedded]) -> int:
        if not observations:
            return 0

        event_ids = allocate_event_ids(self.conn, len(observations))
        for embedded, event_id in zip(observations, event_ids):
            embedded.observation.event_id = event_id

        self._write_postgres(observations)
        self._write_lance(observations)
        return len(observations)

    def repair_lance(self, observations: list[AnyEmbedded]) -> int:
        if not observations:
            return 0
        self._resolve_repair_event_ids(observations)
        return self._write_lance(observations)

    def index_existing_tables(self) -> None:
        for writer in self.vector_writers.values():
            writer.index_existing_tables()

    def build_indexes(self) -> None:
        for writer in self.vector_writers.values():
            writer.build_indexes()

    def _resolve_repair_event_ids(self, observations: list[AnyEmbedded]) -> None:
        if not observations:
            return

        for embedded in observations:
            observation = embedded.observation
            clauses = [
                "corpus = %s",
                "doc_id = %s",
                "token_idx = %s",
                "span_end_idx IS NOT DISTINCT FROM %s",
                "local_window_id IS NOT DISTINCT FROM %s",
                "local_window_token_pos IS NOT DISTINCT FROM %s",
                "medium_window_id IS NOT DISTINCT FROM %s",
                "medium_window_token_pos IS NOT DISTINCT FROM %s",
                "broad_window_id IS NOT DISTINCT FROM %s",
                "broad_window_token_pos IS NOT DISTINCT FROM %s",
            ]
            params = (
                observation.corpus,
                observation.doc_id,
                observation.token_idx,
                getattr(observation, "span_end_idx", None),
                observation.local_window_id,
                observation.local_window_token_pos,
                observation.medium_window_id,
                observation.medium_window_token_pos,
                observation.broad_window_id,
                observation.broad_window_token_pos,
            )

            with self.conn.cursor() as cur:
                cur.execute(
                    f"SELECT event_id FROM events WHERE {' AND '.join(clauses)}",
                    params,
                )
                rows = cur.fetchall()

            if not rows:
                raise RuntimeError(
                    "Lance repair could not find an existing PostgreSQL event for "
                    f"{observation.corpus}/{observation.doc_id}/{observation.token_idx}"
                )
            if len(rows) > 1:
                raise RuntimeError(
                    "Lance repair found multiple PostgreSQL events for the same "
                    f"provenance: {observation.corpus}/{observation.doc_id}/{observation.token_idx}"
                )
            observation.event_id = int(rows[0][0])

    def _write_postgres(self, observations: list[AnyEmbedded]) -> None:
        if not observations:
            return

        if any(e.observation.event_id is None for e in observations):
            raise RuntimeError("Cannot persist observations before event IDs are allocated.")

        insert__events(
            self.conn,
            event_id=[e.observation.event_id for e in observations],
            corpus=[e.observation.corpus for e in observations],
            doc_id=[e.observation.doc_id for e in observations],
            token=[e.observation.token for e in observations],
            token_idx=[e.observation.token_idx for e in observations],
            span_end_idx=[
                getattr(e.observation, "span_end_idx", None)
                for e in observations
            ],
            pub_year=[e.observation.pub_year for e in observations],
            local_window_id=[e.observation.local_window_id for e in observations],
            local_window_token_pos=[e.observation.local_window_token_pos for e in observations],
            medium_window_id=[e.observation.medium_window_id for e in observations],
            medium_window_token_pos=[e.observation.medium_window_token_pos for e in observations],
            broad_window_id=[e.observation.broad_window_id for e in observations],
            broad_window_token_pos=[e.observation.broad_window_token_pos for e in observations],
        )
        self.conn.commit()

    def _write_lance(self, observations: list[AnyEmbedded]) -> int:
        batches: dict[tuple[str, int], list[AnyEmbedded]] = defaultdict(list)

        for embedded in observations:
            observation = embedded.observation
            if observation.event_id is None:
                raise RuntimeError("Cannot write to Lance without an event ID.")
            if observation.pub_year is None:
                raise ValueError(
                    f"Observation {observation.event_id} has no publication year."
                )
            for scale in ACTIVE_SCALES:
                batches[(scale, observation.pub_year)].append(embedded)

        total_written = 0
        for (scale, pub_year), batch in batches.items():
            writer = self.vector_writers[scale]
            event_ids = [e.observation.event_id for e in batch]
            vectors = np.stack([e.vectors[scale] for e in batch])
            total_written += writer.write(
                event_ids=event_ids,
                pub_year=pub_year,
                vectors=vectors,
            )
        return total_written


class CorpusProcessor:
    def __init__(
        self,
        conn,
        pipeline: MacBERThPipeline,
        writer: EventWriter,
        *,
        neighbour_radius: int = 256,
        report_every: int = 25,
    ) -> None:
        self.conn = conn
        self.pipeline = pipeline
        self.writer = writer
        self.neighbour_radius = neighbour_radius
        self.report_every = report_every

    # ------------------------------------------------------------------
    # Token path (unchanged behaviour)
    # ------------------------------------------------------------------

    def process(
        self,
        *,
        corpus: str | None = None,
        doc_id: str | None = None,
    ) -> None:
        documents = self._find_seed_documents(corpus=corpus, doc_id=doc_id)
        logger.info("[tier1] seed documents: %d", len(documents))
        if documents:
            logger.info("[tier1] first seed documents: %s", documents[:5])

        completed_docs_by_corpus: dict[str, set[str]] = {}
        for document_corpus, _ in documents:
            if document_corpus not in completed_docs_by_corpus:
                completed_docs_by_corpus[document_corpus] = existing_event_documents(
                    self.writer.conn, corpus=document_corpus
                )

        for number, (document_corpus, document_id) in enumerate(documents, start=1):
            completed_docs = completed_docs_by_corpus[document_corpus]
            if document_id in completed_docs:
                logger.info(
                    "[tier1] skipping completed document %s/%s",
                    document_corpus, document_id,
                )
                continue

            started = time.perf_counter()
            document = self._load_document(document_corpus, document_id)
            if document is None:
                continue

            seed_positions = {
                p for p, row in enumerate(document.rows) if is_seed(row.token)
            }
            if not seed_positions:
                continue

            target_positions = self._select_neighbours(seed_positions, document)
            embeddings = self.pipeline.embed(document, target_positions)
            observations = self._build_observations(document, target_positions, embeddings)
            written = self.writer.write(observations)

            elapsed = time.perf_counter() - started
            logger.info(
                "[tier1] %3d/%-3d %-4s %-15s seeds=%3d observations=%5d "
                "written=%5d elapsed=%7.2fs",
                number, len(documents), document_corpus, document_id,
                len(seed_positions), len(observations), written, elapsed,
            )
            completed_docs.add(document_id)

            if number % self.report_every == 0:
                logger.info("[tier1] processed %d documents", number)

    # ------------------------------------------------------------------
    # Phrase / span path
    # ------------------------------------------------------------------

    def process_phrases(
        self,
        *,
        corpus: str | None = None,
        doc_id: str | None = None,
    ) -> None:
        documents = self._find_phrase_documents(corpus=corpus, doc_id=doc_id)
        logger.info("[tier1-phrases] candidate documents: %d", len(documents))

        completed_docs_by_corpus: dict[str, set[str]] = {}
        for document_corpus, _ in documents:
            if document_corpus not in completed_docs_by_corpus:
                completed_docs_by_corpus[document_corpus] = existing_event_documents(
                    self.writer.conn, corpus=document_corpus
                )

        for number, (document_corpus, document_id) in enumerate(documents, start=1):
            completed_docs = completed_docs_by_corpus[document_corpus]
            # Note: we reuse the same completed-docs set.  If you want phrases
            # to be processed even when the token path already ran on a doc,
            # remove this check or use a separate marker.
            if document_id in completed_docs:
                logger.info(
                    "[tier1-phrases] skipping completed document %s/%s",
                    document_corpus, document_id,
                )
                continue

            started = time.perf_counter()
            document = self._load_document(document_corpus, document_id)
            if document is None:
                continue

            spans = self._find_phrase_spans(document)
            if not spans:
                continue

            observations: list[EmbeddedSpanObservation] = []
            for start, end, canonical in spans:
                embeddings = self.pipeline.embed_span(document, start, end)
                local = embeddings.get("local")
                medium = embeddings.get("medium")
                broad = embeddings.get("broad")

                obs = SpanObservation(
                    event_id=None,
                    corpus=document.corpus,
                    doc_id=document.doc_id,
                    token=canonical,
                    token_idx=start,
                    span_end_idx=end,
                    pub_year=document.pub_year,
                    local_window_id=local.window_id if local else None,
                    local_window_token_pos=local.window_token_pos if local else None,
                    medium_window_id=medium.window_id if medium else None,
                    medium_window_token_pos=medium.window_token_pos if medium else None,
                    broad_window_id=broad.window_id if broad else None,
                    broad_window_token_pos=broad.window_token_pos if broad else None,
                )
                observations.append(
                    EmbeddedSpanObservation(
                        observation=obs,
                        vectors={s: e.vector for s, e in embeddings.items()},
                    )
                )

            written = self.writer.write(observations)
            elapsed = time.perf_counter() - started
            logger.info(
                "[tier1-phrases] %3d/%-3d %-4s %-15s spans=%3d written=%5d elapsed=%7.2fs",
                number, len(documents), document_corpus, document_id,
                len(spans), written, elapsed,
            )
            completed_docs.add(document_id)

    def _find_phrase_spans(
        self, document: DocBuffer
    ) -> list[tuple[int, int, str]]:
        """Return list of (start_idx, end_idx inclusive, canonical_form)."""
        tokens = [normalise_token(r.token) for r in document.rows]
        spans = []
        for form in PHRASE_FORMS:
            n = len(form)
            for i in range(len(tokens) - n + 1):
                if tuple(tokens[i : i + n]) == form:
                    canonical = " ".join(form)
                    spans.append((i, i + n - 1, canonical))
        return spans

    def _find_phrase_documents(
        self, *, corpus: str | None = None, doc_id: str | None = None
    ) -> list[tuple[str, str]]:
        all_phrase_tokens = {t for seq in PHRASE_FORMS for t in seq}
        clauses = ["lower(t.token) = ANY(%s)"]
        params: list[object] = [sorted(all_phrase_tokens)]

        if corpus is not None:
            clauses.append("t.corpus = %s")
            params.append(corpus)
        if doc_id is not None:
            clauses.append("t.doc_id = %s")
            params.append(doc_id)

        sql = f"""
            SELECT DISTINCT t.corpus, t.doc_id
            FROM pamphlet_tokens t
            WHERE {" AND ".join(clauses)}
            ORDER BY t.corpus, t.doc_id
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _build_observations(
        self,
        document: DocBuffer,
        target_positions: set[int],
        embeddings: dict[int, dict[str, EmbeddedVector]],
    ) -> list[EmbeddedObservation]:
        observations = []
        for position in sorted(target_positions):
            row = document.rows[position]
            vectors: dict[str, np.ndarray] = {}
            provenance: dict[str, EmbeddedVector] = {}

            for scale in ACTIVE_SCALES:
                embedded = embeddings[position].get(scale)
                if embedded is None:
                    raise RuntimeError(
                        f"Missing embedding for {document.corpus}/{document.doc_id}/"
                        f"{row.token_idx}, scale={scale}"
                    )
                vectors[scale] = embedded.vector
                provenance[scale] = embedded

            local = provenance.get("local")
            medium = provenance.get("medium")
            broad = provenance.get("broad")

            observation = Observation(
                event_id=None,
                corpus=row.corpus,
                doc_id=row.doc_id,
                token=row.token,
                token_idx=row.token_idx,
                pub_year=row.pub_year,
                local_window_id=local.window_id if local else None,
                local_window_token_pos=local.window_token_pos if local else None,
                medium_window_id=medium.window_id if medium else None,
                medium_window_token_pos=medium.window_token_pos if medium else None,
                broad_window_id=broad.window_id if broad else None,
                broad_window_token_pos=broad.window_token_pos if broad else None,
            )
            observations.append(EmbeddedObservation(observation=observation, vectors=vectors))
        return observations

    def _find_seed_documents(
        self, *, corpus: str | None, doc_id: str | None
    ) -> list[tuple[str, str]]:
        clauses = ["lower(t.token) = ANY(%s)"]
        params: list[object] = [sorted(SEED_FORMS - FALSE_POSITIVE_FORMS)]
        if corpus is not None:
            clauses.append("t.corpus = %s")
            params.append(corpus)
        if doc_id is not None:
            clauses.append("t.doc_id = %s")
            params.append(doc_id)

        sql = f"""
            SELECT DISTINCT t.corpus, t.doc_id
            FROM pamphlet_tokens AS t
            WHERE {" AND ".join(clauses)}
            ORDER BY t.corpus, t.doc_id
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def _load_document(self, corpus: str, doc_id: str) -> DocBuffer | None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT t.corpus, t.doc_id, t.token_idx, t.token, d.pub_year
                FROM pamphlet_tokens AS t
                JOIN pamphlet_corpus AS d
                  ON d.corpus = t.corpus AND d.doc_id = t.doc_id
                WHERE t.corpus = %s AND t.doc_id = %s
                ORDER BY t.token_idx
                """,
                (corpus, doc_id),
            )
            rows = cur.fetchall()

        if not rows:
            return None

        return DocBuffer(
            corpus=corpus,
            doc_id=doc_id,
            pub_year=rows[0][4],
            rows=[
                TokenRow(
                    corpus=row[0],
                    doc_id=row[1],
                    token_idx=row[2],
                    token=row[3],
                    pub_year=row[4],
                )
                for row in rows
            ],
        )

    def _select_neighbours(
        self, seed_positions: set[int], document: DocBuffer
    ) -> set[int]:
        positions: set[int] = set()
        for seed_position in seed_positions:
            start = max(0, seed_position - self.neighbour_radius)
            end = min(len(document.rows), seed_position + self.neighbour_radius + 1)
            positions.update(
                p
                for p in range(start, end)
                if p in seed_positions or is_storable_event(document.rows[p].token)
            )
        return positions

    # (backfill / repair helpers left unchanged – they only touch the token path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Tier 1 token or phrase observations via MacBERTh → Lance."
    )
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--neighbour-radius", type=int, default=256)
    parser.add_argument("--lance-root", type=Path, default=Path(LANCE_INDEXES_DIR))
    parser.add_argument("--batch-size", type=int, default=EMBED_BATCH_SIZE)
    parser.add_argument("--report-every", type=int, default=1)
    parser.add_argument(
        "--phrases",
        action="store_true",
        help="Run the phrase/span path instead of the token seed path.",
    )
    parser.add_argument("--skip-indexing", action="store_true")
    # (repair / add-scale flags omitted for brevity – keep them if you still need them)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    torch.set_num_interop_threads(1)

    conn = get_connection(application_name="tier1-phrases2events")
    create_events_table(conn)

    try:
        mac = load_macberth()
        pipeline = MacBERThPipeline(mac, batch_size=args.batch_size)
        writer = EventWriter(conn, args.lance_root)
        processor = CorpusProcessor(
            conn,
            pipeline,
            writer,
            neighbour_radius=args.neighbour_radius,
            report_every=args.report_every,
        )

        if args.phrases:
            processor.process_phrases(corpus=args.corpus, doc_id=args.doc_id)
        else:
            processor.process(corpus=args.corpus, doc_id=args.doc_id)

        if not args.skip_indexing:
            writer.build_indexes()
        else:
            logger.info("[tier1] --skip-indexing set; leaving index rebuild for later")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
