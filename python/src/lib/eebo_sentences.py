from collections.abc import Iterator
from psycopg import Connection, sql
from typing import List, Tuple

def stream_slice_sentences(
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
