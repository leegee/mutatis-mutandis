from collections.abc import Iterator
from psycopg import Connection, sql
from transformers import PreTrainedTokenizerBase
from typing import List, Tuple

def OLD_stream_slice_sentences(
    conn: Connection,
    slice_range: tuple[int, int],
    window: int = 64
) -> Iterator[Tuple[int, str, List[int]]]:
    """
    Yields (doc_id, sentence_text, token_occurrence_id_list) per sentence window.
    """
    slice_start, slice_end = slice_range

    query = sql.SQL("""
    WITH numbered AS (
        SELECT
            doc_id,
            token,
            token_occurrence_id,
            (row_number() OVER (PARTITION BY doc_id ORDER BY token_idx) - 1) / {window} AS window_id
        FROM pamphlet_tokens
        WHERE slice_start = %(slice_start)s
        AND slice_end   = %(slice_end)s
    )
    SELECT doc_id,
        STRING_AGG(token, ' ' ORDER BY token_occurrence_id) AS sentence,
        ARRAY_AGG(token_occurrence_id ORDER BY token_occurrence_id) AS token_occurrence_ids
    FROM numbered
    GROUP BY doc_id, window_id
    ORDER BY doc_id, window_id;
    """).format(window=sql.Literal(window))

    with conn.cursor(name="slice_sentence_cursor") as cur:  # server-side cursor
        cur.itersize = 2000
        cur.execute(query, {"slice_start": slice_start, "slice_end": slice_end})
        for doc_id, sentence, token_occurrence_ids in cur:
            yield doc_id, sentence, token_occurrence_ids


def stream_contexts_within_model_limit(
    conn: Connection,
    slice_range: tuple[int, int],
    tokenizer: PreTrainedTokenizerBase,
    safety_margin: int = 64
) -> Iterator[Tuple[str, str, List[int]]]:
    """
    Yields (doc_id, text, token_occurrence_ids) such that
    tokenizer(text) never exceeds model_max_length.

    Invariant:
        tokenizer(text) < model_max_length

    Guarantees no truncation during embedding and preserves
    1:1 alignment between token_occurrence_ids and emitted vectors.

    Implementation note:
        Tracks subword length incrementally instead of re-tokenizing
        the full buffer each step. Uses a cache because EEBO tokens
        are highly repetitive.
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

        # cache: token -> subword length
        token_len_cache: dict[str, int] = {}

        def token_subword_len(tok: str) -> int:
            cached = token_len_cache.get(tok)
            if cached is not None:
                return cached

            ids = tokenizer(
                tok,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]

            length = len(ids)
            token_len_cache[tok] = length
            return length

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

            # check if adding this token would overflow
            if current_len + tok_len + special_tokens > max_tokens:
                result = flush()
                if result:
                    yield result

                # start new buffer
                buffer_tokens = [token]
                buffer_ids = [occ_id]
                current_len = tok_len
            else:
                buffer_tokens.append(token)
                buffer_ids.append(occ_id)
                current_len += tok_len

        # final flush
        result = flush()
        if result:
            yield result
