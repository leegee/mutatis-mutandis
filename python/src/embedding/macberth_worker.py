# embedding/macberth_worker.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from lib.corpus_logging import logger


WINDOW_CONFIGS = (
    {"name": "local", "size": 256, "stride": 128},
    {"name": "medium", "size": 512, "stride": 256},
    {"name": "broad", "size": 512, "stride": 384},
)


@dataclass(slots=True)
class TokenRow:
    corpus: str
    doc_id: str
    token_idx: int
    token: str
    pub_year: int | None


@dataclass(slots=True)
class DocBuffer:
    corpus: str
    doc_id: str
    pub_year: int | None
    rows: list[TokenRow]

    @property
    def tokens(self) -> list[str]:
        return [row.token for row in self.rows]


@dataclass(slots=True)
class EmbeddedVector:
    vector: np.ndarray
    window_id: int
    window_token_pos: int


class MacBERThEventEmbedder:
    """
    Embed authoritative PostgreSQL corpus events with MacBERTh.

    This class is the reusable embedding engine for the Mutatis
    embedding worker.

    The input identity is an existing PostgreSQL events.event_id.
    Event IDs are resolved to their authoritative corpus/document/token
    coordinates, after which the original Tier 1 MacBERTh windowing
    algorithm is applied.

    The window construction is intentionally equivalent to the former
    MacBERThPipeline implementation in tier1_corpus2events.py:

        * corpus tokens are loaded in token_idx order;
        * Hugging Face tokenization uses is_split_into_words=True;
        * corpus-token positions are aligned to MacBERTh subword spans;
        * windows use the configured word-level size and stride;
        * each target remains inside its selected <=512-position
          MacBERTh input window;
        * multiple nearby targets may share a MacBERTh forward pass;
        * target hidden states are taken from the base encoder;
        * --mask behaviour is preserved by replacing the target
          subword position with the tokenizer mask token.

    The class deliberately does not know about:

        * work leases;
        * embedding inventory;
        * batch files;
        * LanceDB;
        * PostgreSQL transactions around work completion.

    Those belong to the embedding worker/importer layers.

    Args:
        conn:
            PostgreSQL connection used to resolve event IDs and load
            authoritative corpus tokens.
        mac:
            Loaded MacBERTh model returned by load_macberth().
        batch_size:
            Number of MacBERTh windows processed in one forward pass.
        scale:
            Observation scale to embed: "local", "medium", or "broad".
        mask_targets:
            If true, replace each target with MacBERTh's mask token
            before the forward pass.
    """

    def __init__(
        self,
        conn,
        mac,
        *,
        batch_size: int = 64,
        scale: str = "local",
        mask_targets: bool = False,
    ) -> None:
        self.conn = conn
        self.mac = mac
        self.tokenizer = mac.tokenizer
        self.device = mac.device
        self.batch_size = batch_size
        self.scale = scale
        self.mask_targets = mask_targets

        configs = {
            config["name"]: config
            for config in WINDOW_CONFIGS
        }

        if scale not in configs:
            raise ValueError(
                f"Unknown embedding scale: {scale!r}. "
                f"Expected one of {sorted(configs)}."
            )

        self.window_size = configs[scale]["size"]
        self.stride = configs[scale]["stride"]

    def embed_events(
        self,
        event_ids: Sequence[int],
    ) -> dict[int, np.ndarray]:
        """
        Embed exactly the supplied PostgreSQL event IDs.

        Returns:
            A mapping of event_id -> float32 embedding.

        Raises:
            RuntimeError:
                If an event does not exist, its document/token cannot
                be resolved, or an event fails to receive an embedding.
        """
        if not event_ids:
            return {}

        requested_ids = [
            int(event_id)
            for event_id in event_ids
        ]

        if len(set(requested_ids)) != len(requested_ids):
            raise ValueError(
                "embed_events() received duplicate event IDs."
            )

        targets_by_document = self._load_targets(
            requested_ids,
        )

        results: dict[int, np.ndarray] = {}

        for (
            document,
            target_positions,
            event_id_by_position,
        ) in targets_by_document:
            embedded = self.embed_document_targets(
                document=document,
                target_positions=target_positions,
            )

            for position, value in embedded.items():
                event_id = event_id_by_position[position]

                results[event_id] = value.vector

        missing = set(requested_ids) - results.keys()

        if missing:
            raise RuntimeError(
                "Some requested events did not receive embeddings: "
                f"{sorted(missing)[:20]}"
            )

        return results

    def _load_targets(
        self,
        event_ids: Sequence[int],
    ) -> list[
        tuple[
            DocBuffer,
            set[int],
            dict[int, int],
        ]
    ]:
        """
        Resolve event IDs to document-relative token positions.

        Events are grouped by corpus/document so each affected document
        is loaded exactly once.

        Returns:
            Tuples containing:

                document
                target_positions
                event_id_by_position
        """
        unique_ids = sorted(
            set(int(event_id) for event_id in event_ids)
        )

        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    e.event_id,
                    e.corpus,
                    e.doc_id,
                    e.token_idx
                FROM events AS e
                WHERE e.event_id = ANY(%s)
                ORDER BY
                    e.corpus,
                    e.doc_id,
                    e.token_idx
                """,
                (unique_ids,),
            )

            rows = cur.fetchall()

        found_ids = {
            int(row[0])
            for row in rows
        }

        missing = set(unique_ids) - found_ids

        if missing:
            raise RuntimeError(
                "Requested event IDs do not exist in PostgreSQL: "
                f"{sorted(missing)[:20]}"
            )

        grouped: dict[
            tuple[str, str],
            list[tuple[int, int]],
        ] = {}

        for event_id, corpus, doc_id, token_idx in rows:
            grouped.setdefault(
                (corpus, doc_id),
                [],
            ).append(
                (
                    int(event_id),
                    int(token_idx),
                )
            )

        result = []

        for (
            corpus,
            doc_id,
        ), event_rows in grouped.items():
            document = self._load_document(
                corpus,
                doc_id,
            )

            if document is None:
                raise RuntimeError(
                    f"Document not found: {corpus}/{doc_id}"
                )

            position_by_token_idx = {
                row.token_idx: position
                for position, row in enumerate(document.rows)
            }

            target_positions: set[int] = set()
            event_id_by_position: dict[int, int] = {}

            for event_id, token_idx in event_rows:
                try:
                    position = position_by_token_idx[token_idx]
                except KeyError as exc:
                    raise RuntimeError(
                        "Event refers to a token that is not present "
                        f"in the authoritative document: "
                        f"event_id={event_id}, "
                        f"{corpus}/{doc_id}/{token_idx}"
                    ) from exc

                if position in event_id_by_position:
                    raise RuntimeError(
                        "Multiple event IDs refer to the same "
                        "document token position: "
                        f"{corpus}/{doc_id}/{token_idx}"
                    )

                target_positions.add(position)
                event_id_by_position[position] = event_id

            result.append(
                (
                    document,
                    target_positions,
                    event_id_by_position,
                )
            )

        return result

    def _load_document(
        self,
        corpus: str,
        doc_id: str,
    ) -> DocBuffer | None:
        """
        Load the complete authoritative token sequence for one document.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    t.corpus,
                    t.doc_id,
                    t.token_idx,
                    t.token,
                    d.pub_year
                FROM tokens AS t
                JOIN documents AS d
                  ON d.corpus = t.corpus
                 AND d.doc_id = t.doc_id
                WHERE t.corpus = %s
                  AND t.doc_id = %s
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

    def embed_document_targets(
        self,
        *,
        document: DocBuffer,
        target_positions: set[int],
    ) -> dict[int, EmbeddedVector]:
        """
        Apply the existing Tier 1 MacBERTh windowing algorithm to
        selected corpus-token positions.

        Returns:
            document token position -> EmbeddedVector
        """
        if not target_positions:
            return {}

        results: dict[int, EmbeddedVector] = {}

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

        jobs = self._make_jobs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_ids=word_ids,
            target_positions=target_positions,
            window_size=self.window_size,
            stride=self.stride,
        )

        for offset in range(0, len(jobs), self.batch_size):
            batch = jobs[
                offset : offset + self.batch_size
            ]

            hidden = self._forward(batch)

            for job, vectors in zip(batch, hidden):
                for target, vector in zip(
                    job["targets"],
                    vectors,
                ):
                    word_position = target["word_position"]

                    if word_position not in target_positions:
                        raise RuntimeError(
                            "Embedding returned a target position "
                            f"that was not requested: {word_position}"
                        )

                    results[word_position] = EmbeddedVector(
                        vector=vector,
                        window_id=job["window_id"],
                        window_token_pos=target[
                            "encoded_position"
                        ],
                    )

        missing = []

        for position in sorted(target_positions):
            if position not in results:
                missing.append(
                    (
                        position,
                        document.rows[position].token,
                    )
                )

        if missing:
            logger.error(
                "[embedding] incomplete embeddings: %d observations",
                len(missing),
            )

            for position, token in missing[:20]:
                logger.error(
                    "[embedding] position=%d token=%r",
                    position,
                    token,
                )

            raise RuntimeError(
                f"{len(missing)} observations did not receive "
                f"a {self.scale} embedding."
            )

        return results

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
        """
        Construct MacBERTh jobs using corpus-token windows.

        This is the original Tier 1 window construction algorithm.
        """
        word_count = (
            max(
                word_id
                for word_id in word_ids
                if word_id is not None
            )
            + 1
        )

        word_spans: list[tuple[int, int]] = []
        current_word = None
        current_start = None

        for encoded_position, word_id in enumerate(word_ids):
            if word_id is None:
                continue

            if word_id != current_word:
                if current_word is not None:
                    word_spans.append(
                        (
                            current_start,
                            encoded_position,
                        )
                    )

                current_word = word_id
                current_start = encoded_position

        if current_word is not None:
            word_spans.append(
                (
                    current_start,
                    len(word_ids),
                )
            )

        if len(word_spans) != word_count:
            raise RuntimeError(
                "MacBERTh word alignment is incomplete: "
                f"expected {word_count} corpus tokens, "
                f"got {len(word_spans)} encoded spans."
            )

        jobs: list[dict] = []
        covered_targets: set[int] = set()

        start_word = 0

        while start_word < word_count:
            end_word = min(
                word_count,
                start_word + window_size,
            )

            candidate_targets = sorted(
                position
                for position in target_positions
                if start_word <= position < end_word
            )

            if candidate_targets:
                group: list[int] = []

                for target in candidate_targets:
                    if not group:
                        group.append(target)
                        continue

                    group_start = word_spans[
                        group[0]
                    ][0]

                    group_end = word_spans[
                        target
                    ][1]

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

        missing_targets = (
            target_positions - covered_targets
        )

        if missing_targets:
            raise RuntimeError(
                "Some target observations were not assigned to "
                f"a MacBERTh job: "
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
        """
        Construct one <=512-position MacBERTh input window.

        This is intentionally the original Tier 1 implementation.
        """
        target_start_word = target_positions[0]
        target_end_word = target_positions[-1] + 1

        context_start = word_spans[
            context_start_word
        ][0]

        context_end = word_spans[
            context_end_word - 1
        ][1]

        target_start = word_spans[
            target_start_word
        ][0]

        target_end = word_spans[
            target_end_word - 1
        ][1]

        target_span = target_end - target_start

        if target_span > 512:
            raise RuntimeError(
                "A target group exceeds MacBERTh's 512-position "
                f"limit: targets={target_start_word}:"
                f"{target_end_word}, "
                f"encoded_length={target_span}"
            )

        available_length = (
            context_end - context_start
        )

        if available_length > 512:
            desired_start = target_start - (
                512 - target_span
            ) // 2

            encoded_start = max(
                context_start,
                desired_start,
            )

            encoded_end = min(
                context_end,
                encoded_start + 512,
            )

            if encoded_end - encoded_start < 512:
                encoded_start = max(
                    context_start,
                    encoded_end - 512,
                )
        else:
            encoded_start = context_start
            encoded_end = context_end

        if not (
            encoded_start <= target_start
            and target_end <= encoded_end
        ):
            raise RuntimeError(
                "Constructed MacBERTh context does not contain "
                f"all targets: targets={target_start_word}:"
                f"{target_end_word}, "
                f"context={context_start_word}:"
                f"{context_end_word}"
            )

        relative_word_ids = word_ids[
            encoded_start:encoded_end
        ]

        window_ids = input_ids[
            encoded_start:encoded_end
        ].copy()

        window_mask = attention_mask[
            encoded_start:encoded_end
        ]

        target_positions_in_window = []

        for word_position in target_positions:
            try:
                relative = relative_word_ids.index(
                    word_position
                )
            except ValueError as exc:
                raise RuntimeError(
                    "Target disappeared from its MacBERTh "
                    f"window: word_position={word_position}"
                ) from exc

            target_positions_in_window.append(
                {
                    "word_position": word_position,
                    "encoded_position": relative,
                }
            )

            if self.mask_targets:
                mask_token_id = (
                    self.tokenizer.mask_token_id
                )

                if mask_token_id is None:
                    raise RuntimeError(
                        "MacBERTh tokenizer has no mask token."
                    )

                window_ids[relative] = mask_token_id

        jobs.append(
            {
                "input_ids": window_ids,
                "attention_mask": window_mask,
                "window_id": context_start_word,
                "targets": target_positions_in_window,
            }
        )

        covered_targets.update(target_positions)

    def _forward(
        self,
        jobs: list[dict],
    ) -> list[list[np.ndarray]]:
        """
        Run the prepared windows through MacBERTh's base encoder.
        """
        if not jobs:
            return []

        max_length = max(
            len(job["input_ids"])
            for job in jobs
        )

        if max_length > 512:
            raise RuntimeError(
                "Prepared MacBERTh batch exceeds 512 tokens: "
                f"{max_length}"
            )

        pad_token_id = self.tokenizer.pad_token_id

        if pad_token_id is None:
            raise RuntimeError(
                "MacBERTh tokenizer has no pad token."
            )

        input_ids = []
        attention_masks = []

        for job in jobs:
            padding = (
                max_length
                - len(job["input_ids"])
            )

            input_ids.append(
                job["input_ids"]
                + [pad_token_id] * padding
            )

            attention_masks.append(
                job["attention_mask"]
                + [0] * padding
            )

        input_tensor = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=self.device,
        )

        attention_tensor = torch.tensor(
            attention_masks,
            dtype=torch.long,
            device=self.device,
        )

        with torch.inference_mode():
            output = self.mac.encode(
                input_ids=input_tensor,
                attention_mask=attention_tensor,
                return_dict=True,
            )

        hidden = (
            output.last_hidden_state
            .cpu()
            .numpy()
        )

        return [
            [
                hidden[
                    batch_index,
                    target["encoded_position"],
                ].astype(
                    np.float32,
                    copy=False,
                )
                for target in job["targets"]
            ]
            for batch_index, job
            in enumerate(jobs)
        ]
