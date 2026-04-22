from collections.abc import Iterator
from psycopg import Connection
from transformers import PreTrainedTokenizerBase
from typing import List, Tuple
from lib.eebo_logging import logger

SENTENCE_END_TOKENS = {".", "!", "?", ";"}  # basic sentence boundaries

import re
from collections.abc import Iterator
from typing import List, Tuple
from psycopg import Connection
from transformers import PreTrainedTokenizerBase
from lib.eebo_logging import logger

# simple sentence boundary detection
SENTENCE_END_RE = re.compile(r'[.!?…]$')
ABBREV = {"mr.", "dr.", "sr.", "jr.", "&c.", "etc."}

def is_sentence_end(token: str) -> bool:
    t = token.lower()
    if t in ABBREV:
        return False
    return bool(SENTENCE_END_RE.search(token))


def stream_sentences_within_model_limit(
    conn: Connection,
    slice_range: tuple[int, int],
    tokenizer: PreTrainedTokenizerBase,
    safety_margin: int = 64
) -> Iterator[Tuple[str, str, List[int]]]:
    """
    Yields (doc_id, text, token_occurrence_ids) where each chunk
    respects tokenizer.model_max_length - safety_margin.

    Sentences are detected using punctuation awareness, and subword
    lengths are cached per token for efficiency.
    """
    max_tokens = tokenizer.model_max_length - safety_margin
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
        buffer_len = 0  # subword length of buffer

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

        for doc_id, token, occ_id in cur:
            if current_doc is None:
                current_doc = doc_id

            # doc boundary → flush buffer
            if doc_id != current_doc:
                # flush any remaining sentence first
                if sentence_tokens:
                    if sentence_len + special_tokens > max_tokens:
                        # pathological sentence: force split at token level
                        for tok, occ in zip(sentence_tokens, sentence_ids):
                            tok_len = token_subword_len(tok)

                            if buffer_len + tok_len + special_tokens > max_tokens:
                                result = flush_buffer()
                                if result:
                                    yield result

                            buffer_tokens.append(tok)
                            buffer_ids.append(occ)
                            buffer_len += tok_len

                        sentence_tokens.clear()
                        sentence_ids.clear()
                        sentence_len = 0
                        continue
                    sentence_tokens.clear()
                    sentence_ids.clear()
                    sentence_len = 0

                # flush buffer for previous doc
                result = flush_buffer()
                if result:
                    yield result
                current_doc = doc_id

            # append token to current sentence
            tok_len = token_subword_len(token)
            sentence_tokens.append(token)
            sentence_ids.append(occ_id)
            sentence_len += tok_len

            # check if sentence ends
            if is_sentence_end(token):
                if buffer_len + sentence_len + special_tokens > max_tokens:
                    # flush buffer
                    result = flush_buffer()
                    if result:
                        yield result

                    buffer_tokens.extend(sentence_tokens)
                    buffer_ids.extend(sentence_ids)
                    buffer_len = sentence_len
                else:
                    buffer_tokens.extend(sentence_tokens)
                    buffer_ids.extend(sentence_ids)
                    buffer_len += sentence_len

                sentence_tokens.clear()
                sentence_ids.clear()
                sentence_len = 0

                # safety flush if buffer exceeds max_tokens
                if buffer_len + special_tokens > max_tokens:
                    result = flush_buffer()
                    if result:
                        yield result

        # flush any remaining sentence
        if sentence_tokens:
            if buffer_len + sentence_len + special_tokens > max_tokens:
                result = flush_buffer()
                if result:
                    yield result
                buffer_tokens.extend(sentence_tokens)
                buffer_ids.extend(sentence_ids)
                buffer_len = sentence_len
            else:
                buffer_tokens.extend(sentence_tokens)
                buffer_ids.extend(sentence_ids)
                buffer_len += sentence_len

        # final flush
        result = flush_buffer()
        if result:
            yield result
