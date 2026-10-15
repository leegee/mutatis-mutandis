"""
passages_1_export_windows.py

Exports selected, not-yet-indexed documents as token windows around target terms.

Each row contains the original token indices and token strings for retained
positions. Gaps between windows are intentional; downstream passage generation
must not treat non-contiguous original token positions as adjacent text.
"""

from collections import defaultdict
from itertools import groupby
from pathlib import Path

import lancedb
import pyarrow as pa
import pyarrow.parquet as pq

from lib.corpus_db import get_connection
from lib.corpus_config import OUT_DIR
from lib.corpus_logging import logger
from lib.corpus_albinism import CORPUS_SQL

EMB_OUTPUT_DIR = OUT_DIR / "export/tokens"
LANCE_DIR = OUT_DIR / "indexes/lance_albinism"

DOC_TABLE = "documents"
TABLE = "tokens"

DOCS_PER_SHARD = 10 # 500
WINDOW_WORDS = 100

# A 40-token window is centred on each hit; an even window has one more
# token to the left than to the right. Original token_idx values are preserved.
WINDOW_LEFT = WINDOW_WORDS // 2
WINDOW_RIGHT = WINDOW_WORDS - WINDOW_LEFT - 1

SCHEMA = pa.schema([
    ("corpus", pa.string()),
    ("doc_id", pa.string()),
    ("pub_year", pa.int32()),
    ("token_idx", pa.list_(pa.int32())),
    ("tokens", pa.list_(pa.string())),
])

# These are the same vocabulary groups used by CORPUS_SQL. Keeping them here
# allows the exporter to retain target windows without exporting whole books.
WHITE = {
    "white", "whyte", "whight", "whit", "whyt", "wight", "wite", "whiet",
}
APPEARANCE = {
    "hair", "haire", "hayre", "hayr", "heare", "hoary", "hoarie",
    "skin", "skinne", "skyn", "complexion", "flesh", "fur", "furre",
    "furr", "coat", "fleece", "pale", "pallid", "pallide", "wan", "wann",
    "wanne", "bloodless", "luminous", "luminouse", "luminus", "luminious",
    "glowing", "glow", "glowe", "glowinge", "glowed", "shining", "shyning",
    "bright", "brightness", "radiant", "radiance", "resplendent",
    "resplendence", "glistering", "glisteringe", "gleaming", "gleaminge",
    "gleam", "gleame", "glittering", "glitteringe", "lucent", "lucid",
}
LIGHT = {
    "luminous", "luminouse", "luminus", "luminious", "glowing", "glow",
    "glowe", "glowinge", "glowed", "shining", "shyning", "bright",
    "brightness", "radiant", "radiance", "resplendent", "resplendence",
    "glistering", "glisteringe", "gleaming", "gleaminge", "gleam", "gleame",
    "glittering", "glitteringe", "lucent", "lucid",
}
SUPERNATURAL = {
    "ghost", "ghosts", "ghostly", "phantom", "phantoms", "phantasme",
    "apparition", "apparitions", "spirit", "spirits", "sprite", "sprites",
    "elf", "elves", "fairy", "fairies", "faerie", "faeries", "fay", "fays",
    "spectre", "spectres", "specter", "specters", "wraith", "wraiths",
    "angel", "angels", "demon", "demons", "daemon", "daemons", "goblin",
    "goblins", "hobgoblin", "hobgoblins", "incubus", "succubus",
}
ALBINISM = {
    "albino", "albinos", "albinoes", "albinism", "albinisms",
    "albinovanus", "albinovani", "albinovanum", "albinovano",
}
TARGET_TERMS = WHITE | APPEARANCE | LIGHT | SUPERNATURAL | ALBINISM


def flush(docs: list[dict], shard_no: int) -> None:
    table = pa.Table.from_pylist(docs, schema=SCHEMA)
    final = EMB_OUTPUT_DIR / f"tokens_{shard_no:05d}.parquet"
    tmp = final.with_suffix(".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(final)
    logger.info("wrote %s (%d docs)", final, len(docs))


def indexed_documents() -> set[tuple[str, str]]:
    """Return documents represented in any existing passage table."""
    db = lancedb.connect(str(LANCE_DIR))
    names = db.list_tables().tables
    indexed: set[tuple[str, str]] = set()

    for name in names:
        if not name.startswith(("passage_l8__", "passage_last__", "passage_mean4__")):
            continue

        table = db.open_table(name)
        rows = table.search().select(["corpus", "doc_id"]).to_arrow()
        indexed.update(
            (corpus, doc_id)
            for corpus, doc_id in zip(rows["corpus"].to_pylist(),
                                      rows["doc_id"].to_pylist())
        )

    logger.info("Found %d indexed documents across %d tables", len(indexed), len(names))
    return indexed


def retained_rows(rows: list[tuple], target_terms: set[str]) -> list[tuple]:
    """Keep rows in the union of fixed-width windows around target occurrences."""
    hits = [
        i for i, row in enumerate(rows)
        if row[4] is not None and row[4].lower() in target_terms
    ]
    if not hits:
        return []

    keep: set[int] = set()
    for hit in hits:
        lo = max(0, hit - WINDOW_LEFT)
        hi = min(len(rows), hit + WINDOW_RIGHT + 1)
        keep.update(range(lo, hi))

    # Original token_idx values remain unchanged. Gaps identify omitted text.
    return [rows[i] for i in sorted(keep)]


def main() -> None:
    EMB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    already_indexed = indexed_documents()
    conn = get_connection(application_name="export-token-windows")

    # Keep the server-side cursor so selected documents do not require a
    # corpus-sized result set in client memory.
    with conn.cursor(name="export_tokens") as cur:
        cur.itersize = 100_000
        cur.execute(
            f"""
            WITH selected_docs AS (
                {CORPUS_SQL}
            )
            SELECT
                t.corpus,
                t.doc_id,
                d.pub_year,
                t.token_idx,
                t.token
            FROM {TABLE} AS t
            JOIN {DOC_TABLE} AS d
              ON d.corpus = t.corpus
             AND d.doc_id = t.doc_id
            JOIN selected_docs AS s
              ON s.corpus = t.corpus
             AND s.doc_id = t.doc_id
            ORDER BY t.corpus, t.doc_id, t.token_idx
            """
        )

        docs: list[dict] = []
        shard_no = 0
        skipped_indexed = 0
        exported = 0

        for (corpus, doc_id), grouped_rows in groupby(
            cur, key=lambda row: (row[0], row[1])
        ):
            if (corpus, doc_id) in already_indexed:
                skipped_indexed += 1
                continue

            rows = list(grouped_rows)
            kept = retained_rows(rows, TARGET_TERMS)
            if not kept:
                continue

            docs.append({
                "corpus": corpus,
                "doc_id": doc_id,
                "pub_year": rows[0][2],
                "token_idx": [row[3] for row in kept],
                "tokens": [row[4] for row in kept],
            })
            exported += 1

            if len(docs) >= DOCS_PER_SHARD:
                flush(docs, shard_no)
                docs, shard_no = [], shard_no + 1

        if docs:
            flush(docs, shard_no)

    conn.close()
    logger.info(
        "Done: exported %d documents; skipped %d already indexed",
        exported, skipped_indexed,
    )


if __name__ == "__main__":
    main()