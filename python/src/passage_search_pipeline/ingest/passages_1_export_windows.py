
"""Export sparse token windows around qualifying historical albinism targets."""

from collections import defaultdict
from itertools import groupby
from pathlib import Path

import lancedb
import pyarrow as pa
import pyarrow.parquet as pq

from lib.corpus_db import get_connection
from lib.corpus_config import OUT_DIR
from lib.corpus_logging import logger

LANCE_DIR = OUT_DIR / "indexes/lance_albinism"
EMB_OUTPUT_DIR = OUT_DIR / "export/source_tokens"
EMB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SHARD_PREFIX = 'tokens_'

FORCE_EXPORT_DOC_IDS = {
    "albinism_case_report_psaila_1972",
}

DOC_TABLE = "documents"
TABLE = "tokens"

DOCS_PER_SHARD = 10
WINDOW_WORDS = 100
WINDOW_LEFT = WINDOW_WORDS // 2
WINDOW_RIGHT = WINDOW_WORDS - WINDOW_LEFT - 1

SCHEMA = pa.schema([
    ("corpus", pa.string()),
    ("doc_id", pa.string()),
    ("pub_year", pa.int32()),
    ("token_idx", pa.list_(pa.int32())),
    ("tokens", pa.list_(pa.string())),
])

WHITE = {
    "white", "whyte", "whight", "whit", "whyt", "wight", "wite", "whiet",
}
APPEARANCE = {
    "hair", "haire", "hayre", "hayr", "heare", "hoary", "hoarie", "heeris", "heer",
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


def flush(docs: list[dict], shard_no: int) -> None:
    table = pa.Table.from_pylist(docs, schema=SCHEMA)
    final = EMB_OUTPUT_DIR / f"{SHARD_PREFIX}_{shard_no:05d}.parquet"
    tmp = final.with_suffix(".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(final)
    logger.info("Wrote %s (%d documents)", final, len(docs))


def indexed_documents() -> set[tuple[str, str]]:
    """Collect document identities present in any passage representation."""
    db = lancedb.connect(str(LANCE_DIR))
    names = db.list_tables().tables
    indexed: set[tuple[str, str]] = set()

    for name in names:
        if not name.startswith((
            "passage_l8__",
            "passage_last__",
            "passage_mean4__",
        )):
            continue

        table = db.open_table(name)
        rows = table.search().select(["corpus", "doc_id"]).to_arrow()
        indexed.update(zip(
            rows["corpus"].to_pylist(),
            rows["doc_id"].to_pylist(),
        ))

    logger.info(
        "Found %d indexed documents across %d Lance tables",
        len(indexed), len(names),
    )
    return indexed


def build_export_query() -> tuple[str, list]:
    groups = {
        "white": WHITE,
        "appearance": APPEARANCE,
        "light": LIGHT,
        "supernatural": SUPERNATURAL,
        "albinism": ALBINISM,
    }

    values = []
    params = []
    for group, terms in groups.items():
        for term in sorted(terms):
            values.append("(%s, %s)")
            params.extend((term, group))

    sql = f"""
        WITH term_groups(term, grp) AS (
            VALUES {", ".join(values)}
        ),
        target_tokens AS MATERIALIZED (
            SELECT DISTINCT
                t.corpus,
                t.doc_id,
                t.token_idx,
                tg.grp
            FROM {TABLE} AS t
            JOIN term_groups AS tg
              ON lower(t.token) = tg.term
        ),
        target_hits AS (
            SELECT w.corpus, w.doc_id, w.token_idx
            FROM target_tokens AS w
            JOIN target_tokens AS a
              ON a.corpus = w.corpus
             AND a.doc_id = w.doc_id
             AND a.grp = 'appearance'
             AND a.token_idx BETWEEN w.token_idx - 10
                                 AND w.token_idx + 10
            WHERE w.grp = 'white'

            UNION

            SELECT l.corpus, l.doc_id, l.token_idx
            FROM target_tokens AS l
            JOIN target_tokens AS s
              ON s.corpus = l.corpus
             AND s.doc_id = l.doc_id
             AND s.grp = 'supernatural'
             AND s.token_idx BETWEEN l.token_idx - 10
                                 AND l.token_idx + 10
            WHERE l.grp = 'light'

            UNION

            SELECT w.corpus, w.doc_id, w.token_idx
            FROM target_tokens AS w
            JOIN target_tokens AS s
              ON s.corpus = w.corpus
             AND s.doc_id = w.doc_id
             AND s.grp = 'supernatural'
             AND s.token_idx BETWEEN w.token_idx - 10
                                 AND w.token_idx + 10
            WHERE w.grp = 'white'

            UNION

            SELECT corpus, doc_id, token_idx
            FROM target_tokens
            WHERE grp = 'albinism'
        ),
        eligible_hits AS (
            SELECT DISTINCT h.corpus, h.doc_id, h.token_idx
            FROM target_hits AS h
            WHERE NOT EXISTS (
                SELECT 1
                FROM export_indexed_docs AS ix
                WHERE ix.corpus = h.corpus
                  AND ix.doc_id = h.doc_id
            )
            OR h.doc_id = ANY(%s)
        )
        SELECT DISTINCT
            t.corpus,
            t.doc_id,
            d.pub_year,
            t.token_idx,
            t.token
        FROM eligible_hits AS h
        JOIN {TABLE} AS t
          ON t.corpus = h.corpus
         AND t.doc_id = h.doc_id
         AND t.token_idx BETWEEN
             h.token_idx - {WINDOW_LEFT}
             AND h.token_idx + {WINDOW_RIGHT}
        JOIN {DOC_TABLE} AS d
          ON d.corpus = t.corpus
         AND d.doc_id = t.doc_id
        ORDER BY t.corpus, t.doc_id, t.token_idx
    """
    return sql, params


def populate_indexed_docs(conn, indexed: set[tuple[str, str]]) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TEMP TABLE export_indexed_docs (
                corpus text NOT NULL,
                doc_id text NOT NULL,
                PRIMARY KEY (corpus, doc_id)
            ) ON COMMIT PRESERVE ROWS
        """)

        ordered = sorted(indexed)
        for offset in range(0, len(ordered), 5000):
            cur.executemany(
                """
                INSERT INTO export_indexed_docs (corpus, doc_id)
                VALUES (%s, %s)
                ON CONFLICT DO NOTHING
                """,
                ordered[offset:offset + 5000],
            )

    conn.commit()


def main() -> None:
    indexed = indexed_documents()
    conn = get_connection(application_name="export-token-windows")

    try:
        populate_indexed_docs(conn, indexed)
        sql, params = build_export_query()
        params.extend([sorted(FORCE_EXPORT_DOC_IDS)])
        docs: list[dict] = []
        shard_no = 0
        exported = 0
        rows_seen = 0

        with conn.cursor(name="export_tokens") as cur:
            cur.itersize = 10_000
            cur.execute(sql, params)

            for (corpus, doc_id), grouped_rows in groupby(
                cur,
                key=lambda row: (row[0], row[1]),
            ):
                rows = list(grouped_rows)
                docs.append({
                    "corpus": corpus,
                    "doc_id": doc_id,
                    "pub_year": rows[0][2],
                    "token_idx": [row[3] for row in rows],
                    "tokens": [row[4] for row in rows],
                })

                exported += 1
                rows_seen += len(rows)

                if len(docs) >= DOCS_PER_SHARD:
                    flush(docs, shard_no)
                    docs = []
                    shard_no += 1

                if exported % 100 == 0:
                    logger.info( "Exported %d documents; %d token rows", exported, rows_seen, )

            if docs:
                flush(docs, shard_no)

        logger.info( "Done: exported %d documents and %d token rows", exported, rows_seen, )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
