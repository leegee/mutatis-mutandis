from collections.abc import Iterator
from typing import List, Tuple
from psycopg import Connection
from transformers import PreTrainedTokenizerBase
from lib.eebo_logging import logger

import re

# invariant: sentence boundaries must be detected from token content only
SENTENCE_END_RE = re.compile(r'[.!?…]$')

# invariant: abbreviations must not trigger sentence boundaries
ABBREV = {"mr.", "dr.", "sr.", "jr.", "etc.", "&c.", "etc."}


def is_sentence_end(token: str) -> bool:
    t = token.lower()
    if t in ABBREV:
        return False
    return bool(SENTENCE_END_RE.search(token))


def stream_sentences_within_model_limit(
    conn: Connection,
    slice_range: tuple[int, int],
    tokenizer: PreTrainedTokenizerBase,
    max_model_tokens: int,
    safety_margin: int = 64
) -> Iterator[Tuple[str, str, List[int]]]:

    # invariant: never rely on tokenizer.model_max_length
    max_tokens = max_model_tokens - safety_margin
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)

    slice_start, slice_end = slice_range

    query = """
    SELECT doc_id, token, token_occurrence_id
    FROM pamphlet_tokens
    WHERE slice_start = %(slice_start)s
      AND slice_end   = %(slice_end)s
    ORDER BY doc_id, token_idx;
    """

    token_len_cache: dict[str, int] = {}

    def token_subword_len(tok: str) -> int:
        if tok in token_len_cache:
            return token_len_cache[tok]
        length = len(tokenizer(tok, add_special_tokens=False, return_attention_mask=False)["input_ids"])
        token_len_cache[tok] = length
        return length

    with conn.cursor(name="slice_sentence_cursor") as cur:
        cur.itersize = 2000
        cur.execute(query, {"slice_start": slice_start, "slice_end": slice_end})

        current_doc = None

        buffer_tokens: List[str] = []
        buffer_ids: List[int] = []
        buffer_len = 0

        sentence_tokens: List[str] = []
        sentence_ids: List[int] = []
        sentence_len = 0

        def flush_buffer():
            nonlocal buffer_tokens, buffer_ids, buffer_len
            if not buffer_tokens:
                return None
            text = " ".join(buffer_tokens)
            ids = buffer_ids.copy()
            buffer_tokens.clear()
            buffer_ids.clear()
            buffer_len = 0
            return current_doc, text, ids

        def flush_sentence_into_buffer():
            nonlocal sentence_tokens, sentence_ids, sentence_len
            nonlocal buffer_tokens, buffer_ids, buffer_len

            if not sentence_tokens:
                return

            # pathological sentence → degrade to token-level packing
            if sentence_len + special_tokens > max_tokens:
                for tok, occ in zip(sentence_tokens, sentence_ids):
                    tok_len = token_subword_len(tok)

                    if buffer_len + tok_len + special_tokens > max_tokens:
                        result = flush_buffer()
                        if result:
                            yield result

                    buffer_tokens.append(tok)
                    buffer_ids.append(occ)
                    buffer_len += tok_len

            else:
                if buffer_len + sentence_len + special_tokens > max_tokens:
                    result = flush_buffer()
                    if result:
                        yield result

                buffer_tokens.extend(sentence_tokens)
                buffer_ids.extend(sentence_ids)
                buffer_len += sentence_len

            sentence_tokens.clear()
            sentence_ids.clear()
            sentence_len = 0

        for doc_id, token, occ_id in cur:

            if current_doc is None:
                current_doc = doc_id

            # invariant: doc boundary must flush all state
            if doc_id != current_doc:
                yield from flush_sentence_into_buffer()

                result = flush_buffer()
                if result:
                    yield result

                current_doc = doc_id

            tok_len = token_subword_len(token)

            sentence_tokens.append(token)
            sentence_ids.append(occ_id)
            sentence_len += tok_len

            # early pathological detection
            if sentence_len + special_tokens > max_tokens:
                yield from flush_sentence_into_buffer()
                continue

            if is_sentence_end(token):
                yield from flush_sentence_into_buffer()

                if buffer_len + special_tokens > max_tokens:
                    result = flush_buffer()
                    if result:
                        yield result

        # flush trailing sentence
        yield from flush_sentence_into_buffer()

        # final buffer flush
        result = flush_buffer()
        if result:
            yield result
