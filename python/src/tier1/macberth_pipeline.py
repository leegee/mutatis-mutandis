import numpy as np

from lib.corpus_config import (
    EMBED_BATCH_SIZE, ACTIVE_SCALES
 )

from tier1.models import *

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


    def _make_window_jobs(
        self,
        *,
        document: DocBuffer,
        target_positions: set[int],
        window_size: int,
        stride: int,
    ) -> list[dict]:
        """
        Produce the same logical windows as the old _make_jobs, but never
        materialise a full-document encoding.
        """
        word_count = len(document.rows)
        jobs: list[dict] = []
        covered: set[int] = set()

        start_word = 0
        while start_word < word_count:
            end_word = min(word_count, start_word + window_size)

            candidate_targets = sorted(
                p for p in target_positions if start_word <= p < end_word
            )

            if candidate_targets:
                # Group targets greedily according to their actual encoded
                # subword span.  Source-word count is not a reliable proxy
                # for MacBERTh token count, especially for historical text.
                group: list[int] = []

                for target in candidate_targets:
                    if not group:
                        group = [target]
                        continue

                    trial = group + [target]

                    span_start = trial[0]
                    span_end = trial[-1] + 1

                    span_tokens = document.tokens[span_start:span_end]
                    encoded = self.tokenizer(
                        span_tokens,
                        is_split_into_words=True,
                        truncation=False,
                        return_tensors=None,
                    )

                    encoded_len = len(encoded["input_ids"])

                    if encoded_len <= 512:
                        group = trial
                    else:
                        jobs.append(
                            self._build_one_window(
                                document=document,
                                target_positions=group,
                                context_start_word=start_word,
                                context_end_word=end_word,
                            )
                        )
                        covered.update(group)
                        group = [target]

                if group:
                    jobs.append(
                        self._build_one_window(
                            document=document,
                            target_positions=group,
                            context_start_word=start_word,
                            context_end_word=end_word,
                        )
                    )
                    covered.update(group)

            if start_word + stride >= word_count:
                break

            start_word += stride

        missing = target_positions - covered
        if missing:
            raise RuntimeError( f"Some target observations were not assigned to a MacBERTh job: {sorted(missing)[:20]}" )

        return jobs

    def _build_one_window(
        self,
        *,
        document: DocBuffer,
        target_positions: list[int],
        context_start_word: int,
        context_end_word: int,
    ) -> dict:
        """
        Tokenize only the concrete window that will be fed to the model.
        Recentres around the targets exactly as the old _append_job did.
        """
        # 1. Decide the final token slice that will be encoded. Start from the logical context window, then shrink/recenter so the encoded length stays ≤ 512.
        target_start_word = target_positions[0]
        target_end_word = target_positions[-1] + 1

        # First try the full context window.
        slice_start = context_start_word
        slice_end = context_end_word

        # Tokenize a trial to learn the real subword length.
        trial_tokens = document.tokens[slice_start:slice_end]
        trial = self.tokenizer(
            trial_tokens,
            is_split_into_words=True,
            truncation=False,
            return_tensors=None,          # plain lists
        )
        encoded_len = len(trial["input_ids"])

        if encoded_len > 512:
            # The model limit is in subword tokens, not source words. Shrink the word-space window around the target span until the actual tokenized length fits within 512.
            # Always preserve the complete target span.
            while encoded_len > 512:
                left_available = target_start_word - slice_start
                right_available = slice_end - target_end_word

                removable = left_available + right_available
                if removable <= 0:
                    raise RuntimeError(
                        f"Target span itself exceeds 512 encoded tokens: "
                        f"words={target_start_word}:{target_end_word}, "
                        f"encoded={encoded_len}"
                    )

                # Estimate how many words need to be removed.  Use the current token/word ratio, with a small safety margin.
                word_count = slice_end - slice_start
                excess = encoded_len - 512
                tokens_per_word = encoded_len / max(1, word_count)

                remove_words = max(
                    1,
                    int(np.ceil((excess / tokens_per_word) * 1.25)),
                )
                remove_words = min(remove_words, removable)

                # Remove proportionally from the two sides, preferring the side with more available context.
                if removable:
                    remove_left = min(
                        left_available,
                        int(round(remove_words * left_available / removable)),
                    )
                    remove_right = remove_words - remove_left

                    # If rounding pushed the right side beyond what is available, give the remainder to the left.
                    if remove_right > right_available:
                        overflow = remove_right - right_available
                        remove_right = right_available
                        remove_left = min( left_available, remove_left + overflow, )

                    slice_start += remove_left
                    slice_end -= remove_right

                trial_tokens = document.tokens[slice_start:slice_end]
                trial = self.tokenizer(
                    trial_tokens,
                    is_split_into_words=True,
                    truncation=False,
                    return_tensors=None,
                )
                encoded_len = len(trial["input_ids"])
            # At this point the actual tokenized window is guaranteed to fit the model limit.

        input_ids = trial["input_ids"]
        attention_mask = trial["attention_mask"]
        word_ids = trial.word_ids()          # local offsets 0..len(slice)-1

        # 2. Map local word_ids back to global word positions. word_ids[i] is None for special tokens, otherwise an offset relative to the slice.
        targets_in_window = []
        for global_pos in target_positions:
            local_word = global_pos - slice_start
            try:
                # first subword of this word
                encoded_position = word_ids.index(local_word)
            except ValueError as exc:
                raise RuntimeError(
                    f"Target disappeared from its MacBERTh window: global={global_pos}, slice={slice_start}:{slice_end}"
                ) from exc

            targets_in_window.append(
                {
                    "word_position": global_pos,       # still the global index
                    "encoded_position": encoded_position,
                }
            )

            if self.mask_targets:
                mask_id = self.tokenizer.mask_token_id
                if mask_id is None:
                    raise RuntimeError("MacBERTh tokenizer has no mask token.")
                for i, wid in enumerate(word_ids):
                    if wid == local_word:
                        input_ids[i] = mask_id

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "window_id": slice_start,                 # global start word
            "targets": targets_in_window,
        }


    def _forward_windows(self, jobs: list[dict]) -> list[list[np.ndarray]]:
        """Identical to the old _forward, just renamed for clarity."""
        if not jobs:
            return []

        max_length = max(len(job["input_ids"]) for job in jobs)
        if max_length > 512:
            raise RuntimeError(f"Prepared MacBERTh batch exceeds 512 tokens: {max_length}")

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise RuntimeError("MacBERTh tokenizer has no pad token.")

        input_ids = []
        attention_masks = []
        for job in jobs:
            pad = max_length - len(job["input_ids"])
            input_ids.append(job["input_ids"] + [pad_id] * pad)
            attention_masks.append(job["attention_mask"] + [0] * pad)

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

        word_count = len(document.rows)

        for config in WINDOW_CONFIGS:
            if config["name"] not in scales:
                continue

            jobs = self._make_window_jobs(
                document=document,
                target_positions=target_positions,
                window_size=config["size"],
                stride=config["stride"],
            )

            for offset in range(0, len(jobs), self.batch_size):
                batch = jobs[offset : offset + self.batch_size]
                hidden = self._forward_windows(batch)   # see below

                for job, vectors in zip(batch, hidden):
                    for target, vector in zip(job["targets"], vectors):
                        word_position = target["word_position"]
                        results[word_position][config["name"]] = EmbeddedVector(
                            vector=vector,
                            window_id=job["window_id"],          # global start word
                            window_token_pos=target["encoded_position"],
                        )

        # identical completeness check as today
        missing = []
        for position in sorted(target_positions):
            missing_scales = [
                scale for scale in scales if scale not in results[position]
            ]
            if missing_scales:
                missing.append( (position, document.rows[position].token, missing_scales) )

        if missing:
            logger.error("[tier1] incomplete embeddings: %d observations", len(missing))
            for position, token, missing_scales in missing[:20]:
                logger.error( "[tier1] position=%d token=%r missing=%s", position, token, missing_scales )
            raise RuntimeError(
                f"{len(missing)} observations did not receive all requested embeddings."
            )

        return results

    def embed_span(
        self,
        document: DocBuffer,
        start_position: int,
        end_position: int,
        scales: tuple[str, ...] = ACTIVE_SCALES,
    ) -> dict[str, EmbeddedVector]:
        """
        Embed a contiguous phrase span.

        The phrase vector is the mean of the contextual token vectors for
        the complete span. Provenance is anchored to the first token in
        the span's generated MacBERTh window.

        `end_position` is exclusive.
        """

        if not ( 0 <= start_position < end_position <= len(document.rows) ):
            raise ValueError( f"Invalid span {start_position}:{end_position}" )

        positions = set( range(start_position, end_position) )

        embedded = self.embed(
            document,
            positions,
            scales=scales,
        )

        results: dict[str, EmbeddedVector] = {}

        for scale in scales:
            vectors = [
                embedded[position][scale].vector
                for position in range(
                    start_position,
                    end_position,
                )
            ]

            if not vectors:
                raise RuntimeError( f"No vectors generated for span {start_position}:{end_position}" )

            vector = np.mean(
                np.stack(vectors),
                axis=0,
            ).astype(np.float32, copy=False)

            first = embedded[start_position][scale]

            results[scale] = EmbeddedVector(
                vector=vector,
                window_id=first.window_id,
                window_token_pos=first.window_token_pos,
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
            raise RuntimeError( f"MacBERTh word alignment is incomplete: expected {word_count} corpus tokens, got {len(word_spans)} encoded spans." )

        jobs: list[dict] = []
        covered_targets: set[int] = set()
        start_word = 0

        while start_word < word_count:
            end_word = min( word_count, start_word + window_size, )

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
            raise RuntimeError( f"Some target observations were not assigned to a MacBERTh job: {sorted(missing_targets)[:20]}" )

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

        context_start = word_spans[ context_start_word ][0]
        context_end = word_spans[ context_end_word - 1 ][1]
        target_start = word_spans[ target_start_word ][0]
        target_end = word_spans[ target_end_word - 1 ][1]
        target_span = target_end - target_start

        if target_span > 512:
            raise RuntimeError( f"A target group exceeds MacBERTh's 512-position limit: targets={target_start_word}: {target_end_word}, encoded_length={target_span}" )

        available_length = context_end - context_start

        if available_length > 512:
            desired_start = target_start - (
                512 - target_span
            ) // 2

            encoded_start = max( context_start, desired_start, )
            encoded_end = min( context_end, encoded_start + 512, )

            if encoded_end - encoded_start < 512:
                encoded_start = max( context_start, encoded_end - 512, )
        else:
            encoded_start = context_start
            encoded_end = context_end

        if not (
            encoded_start <= target_start
            and target_end <= encoded_end
        ):
            raise RuntimeError( "Constructed MacBERTh context does not contain all targets: targets={target_start_word}: {target_end_word}, context={context_start_word}: {context_end_word}" )

        relative_word_ids = word_ids[ encoded_start:encoded_end ]
        window_ids = input_ids[ encoded_start:encoded_end ].copy()

        window_mask = attention_mask[ encoded_start:encoded_end ]

        target_positions_in_window = []

        for word_position in target_positions:
            try:
                relative = relative_word_ids.index(
                    word_position
                )
            except ValueError as exc:
                raise RuntimeError( f"Target disappeared from its MacBERTh window: word_position={word_position}" ) from exc

            target_positions_in_window.append(
                {
                    "word_position": word_position,
                    "encoded_position": relative,
                }
            )

            if self.mask_targets:
                mask_token_id = self.tokenizer.mask_token_id

                if mask_token_id is None:
                    raise RuntimeError( "MacBERTh tokenizer has no mask token." )

                # Mask every wordpiece belonging to the target.
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

    def _forward(
        self,
        jobs: list[dict],
    ) -> list[list[np.ndarray]]:

        if not jobs:
            return []

        max_length = max(
            len(job["input_ids"])
            for job in jobs
        )

        if max_length > 512:
            raise RuntimeError(
                f"Prepared MacBERTh batch exceeds 512 tokens: "
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
            padding = max_length - len(job["input_ids"])

            input_ids.append( job["input_ids"] + [pad_token_id] * padding )
            attention_masks.append( job["attention_mask"] + [0] * padding )

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

        hidden = output.last_hidden_state.cpu().numpy()

        return [
            [
                hidden[
                    batch_index,
                    target["encoded_position"],
                ].astype(np.float32, copy=False)
                for target in job["targets"]
            ]
            for batch_index, job in enumerate(jobs)
        ]

