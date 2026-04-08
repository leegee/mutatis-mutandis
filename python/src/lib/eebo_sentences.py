from collections.abc import Iterator
from psycopg import Connection
from transformers import PreTrainedTokenizerBase
from typing import List, Tuple
from lib.eebo_logging import logger

SENTENCE_END_TOKENS = {".", "!", "?", ";"}  # basic sentence boundaries

def stream_sentences_within_model_limit(
    conn: Connection,
    slice_range: tuple[int, int],
    tokenizer: PreTrainedTokenizerBase,
    safety_margin: int = 64
) -> Iterator[Tuple[str, str, List[int]]]:
    """
    Yields (doc_id, sentence_text, token_occurrence_ids) per sentence or sub-sentence chunk
    such that tokenizer(text) never exceeds model_max_length.

    Preserves 1:1 alignment between token_occurrence_ids and embedding vectors.
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

    with conn.cursor(name="slice_token_cursor") as cur:
        cur.itersize = 2000
        cur.execute(query, {"slice_start": slice_start, "slice_end": slice_end})

        current_doc = None
        buffer_tokens: List[str] = []
        buffer_ids: List[int] = []
        current_len = 0  # subword length of buffer
        token_len_cache: dict[str, int] = {}

        def token_subword_len(tok: str) -> int:
            """Get cached subword length of token."""
            cached = token_len_cache.get(tok)
            if cached is not None:
                return cached
            ids = tokenizer(tok, add_special_tokens=False, return_attention_mask=False)["input_ids"]
            token_len_cache[tok] = len(ids)
            return len(ids)

        def flush():
            nonlocal buffer_tokens, buffer_ids, current_len
            if not buffer_tokens:
                return None
            result = (current_doc, " ".join(buffer_tokens), buffer_ids.copy())
            buffer_tokens.clear()
            buffer_ids.clear()
            current_len = 0
            return result

        for doc_id, token, occ_id in cur:
            if current_doc is None:
                current_doc = doc_id

            # doc boundary → flush
            if doc_id != current_doc:
                result = flush()
                if result:
                    yield result
                current_doc = doc_id

            tok_len = token_subword_len(token)

            if tok_len + special_tokens > max_tokens:
                logger.warning(
                    f"Single token exceeds model limit: doc_id={doc_id}, "
                    f"occ_id={occ_id}, token='{token[:50]}...', subword_len={tok_len}"
                )

            # Check if adding token would overflow model limit
            if current_len + tok_len + special_tokens > max_tokens:
                result = flush()
                if result:
                    yield result

            # Add token to buffer
            buffer_tokens.append(token)
            buffer_ids.append(occ_id)
            current_len += tok_len

            # Sentence boundary check
            if token in SENTENCE_END_TOKENS:
                # flush sentence
                result = flush()
                if result:
                    yield result

        # final flush at end of slice
        result = flush()
        if result:
            yield result
