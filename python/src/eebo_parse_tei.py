#!/usr/bin/env python

"""

Memory accumulation - monitor
DB connection churn	- pool
Token-only sentences - hmmm
Slice unused - todo

"""


from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from typing import Any, Optional
import argparse
import numpy as np
import re
import io
import torch
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

import eebo_config as config
import eebo_db
import eebo_ocr_fixes
from eebo_logging import logger

MAX_DOCS: Optional[int] = None  # For --limit testing

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    config.MODEL_PATH,
    local_files_only=True
)

model = AutoModel.from_pretrained(
    config.MODEL_PATH,
    local_files_only=True
).to(device)

model.eval()


def get_db_connection():
    """Return a fresh psycopg.Connection for parallel-safe usage"""
    return eebo_db.get_connection()


def normalize_early_modern(text: str) -> str:
    """Normalize Early Modern English text for ingestion/canonicalisation."""
    text = text.lower()
    text = re.sub(r"(\w)[’‘ʼ′´](\w)", r"\1'\2", text)
    text = text.replace("ſ", "s")
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r'-\s*', ' ', text)
    text = re.sub(r'\bv(?=[aeiou])', 'u', text)
    text = re.sub(r'\bj(?=[aeiou])', 'i', text)
    text = re.sub(r'(?<=\w)[^\w\s](?=\w)', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def assign_slice(date_raw: str | None) -> tuple[int | None, int | None]:
    if not date_raw:
        return None, None
    match = re.search(r"\b(\d{4})\b", date_raw)
    if not match:
        return None, None
    year = int(match.group(1))
    for start, end in config.SLICES:
        if start <= year <= end:
            return start, end
    return None, None


# XML processing
def process_file(xml_path: Path) -> Optional[tuple[dict, list[tuple[int, str]]]]:
    """Parse XML file and return doc metadata + token list"""
    import xml.etree.ElementTree as etree
    try:
        tree = etree.parse(str(xml_path))
    except Exception as e:
        logger.error(f"XML parse failed: {xml_path.name}: {e}")
        return None

    doc_id_elem = tree.find(".//HEADER//IDNO[@TYPE='DLPS']")
    doc_id = doc_id_elem.text.strip() if (doc_id_elem is not None and doc_id_elem.text) else None
    if not doc_id:
        logger.error(f"XML rejected: {xml_path.name}: missing document ID")
        return None

    # Metadata extraction
    title_elem = tree.find(".//HEADER//TITLESTMT/TITLE")
    author_elem = tree.find(".//HEADER//TITLESTMT/AUTHOR")
    date_elem = tree.find(".//HEADER//SOURCEDESC//DATE")
    pub_elem = tree.find(".//HEADER//SOURCEDESC//PUBLISHER")
    place_elem = tree.find(".//HEADER//SOURCEDESC//PUBPLACE")

    title = title_elem.text.strip() if title_elem is not None and title_elem.text else None
    author = author_elem.text.strip() if author_elem is not None and author_elem.text else None
    date_raw = date_elem.text.strip() if date_elem is not None and date_elem.text else None
    publisher = pub_elem.text.strip() if pub_elem is not None and pub_elem.text else None
    pub_place = place_elem.text.strip() if place_elem is not None and place_elem.text else None
    slice_start, slice_end = assign_slice(date_raw)

    # Body text
    body_elem = tree.find(".//EEBO//TEXT//BODY")
    if body_elem is None:
        logger.error(f"XML rejected: {xml_path.name}: no body text")
        return None
    raw_text = " ".join(t.strip() for t in body_elem.itertext() if t.strip())
    fixed_text = eebo_ocr_fixes.apply_ocr_fixes(raw_text)
    normalized = normalize_early_modern(fixed_text)
    if len(normalized) < 100:
        logger.error(f"XML rejected: {xml_path.name}: text too short")
        return None

    tokens = normalized.split()
    token_tuples = [(i, tok) for i, tok in enumerate(tokens)]

    doc_meta = dict(
        doc_id=doc_id,
        title=title,
        author=author,
        publisher=publisher,
        pub_place=pub_place,
        source_date_raw=date_raw,
        slice_start=slice_start,
        slice_end=slice_end
    )
    return doc_meta, token_tuples


def process_file_worker(xml_path: Path):
    """Wrapper for parallel XML ingestion: no global DB usage"""
    return process_file(xml_path)


def copy_rows(conn, table: str, columns: list[str], rows: list[list[Any]]) -> None:
    if not rows:
        return

    buf = io.StringIO()
    for row in rows:
        buf.write(
            "\t".join(
                "\\N" if v is None else str(v).replace("\t", " ").replace("\n", " ")
                for v in row
            )
            + "\n"
        )
    buf.seek(0)

    cols = ", ".join(columns)
    copy_sql = (
        f"COPY {table} ({cols}) "
        "FROM STDIN WITH (FORMAT text, NULL '\\N', DELIMITER E'\\t')"
    )

    with conn.cursor() as cur:
        with cur.copy(copy_sql) as copy:
            copy.write(buf.read())

    logger.debug(
        "COPYed %d rows into %s(%s)",
        len(rows),
        table,
        cols,
    )


def batch_update_canonical_dicts(rows: list[tuple[str, int, str]], conn):
    """Batch update canonical forms into tokens table using COPY"""
    buf = io.StringIO()
    for doc_id, token_idx, canonical in rows:
        buf.write(f"{doc_id}\t{token_idx}\t{canonical}\n")
    buf.seek(0)

    with conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE IF NOT EXISTS tmp_canonical(doc_id TEXT, token_idx INT, canonical TEXT)")
        cur.copy("COPY tmp_canonical(doc_id, token_idx, canonical) FROM STDIN WITH (FORMAT text)", buf)
        cur.execute("""
            UPDATE tokens t
            SET canonical = tmp.canonical
            FROM tmp_canonical tmp
            WHERE t.doc_id = tmp.doc_id AND t.token_idx = tmp.token_idx
        """)


# Phase 1 — XML Ingestion
def build_sentences_parallel(max_workers: int = 4, sent_chunk: int = 50_000):
    """
    Build sentences, assign sentence IDs to tokens, and commit in batches.
    """
    logger.info("[PHASE 2] Building sentences and assigning sentence IDs...")

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT d.doc_id, d.slice_start, d.slice_end
                FROM documents d
                JOIN tokens t ON d.doc_id = t.doc_id
                ORDER BY d.doc_id
            """)
            docs = cur.fetchall()

    if MAX_DOCS:
        docs = docs[:MAX_DOCS]

    MAX_SENT_LEN = 60
    MIN_SENT_LEN = 20
    SOFT_BREAK_TOKENS = {"and", "but", "for", "or"}
    HARD_BREAK_TOKENS = {"wherefore", "therefore", "thus"}

    def process_doc(doc_id: str, slice_start: int | None, slice_end: int | None):
        # Fetch tokens
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT token_idx, token
                    FROM tokens
                    WHERE doc_id = %s
                    ORDER BY token_idx
                """, (doc_id,))
                tokens = cur.fetchall()

        if not tokens:
            return [], []

        sentences, bounds = [], []
        sentence_tokens, sentence_id, start_idx = [], 0, tokens[0][0]

        for idx, tok in tokens:
            sentence_tokens.append(tok)
            token_count = len(sentence_tokens)

            hard_len_break = token_count >= MAX_SENT_LEN
            soft_break = token_count >= MIN_SENT_LEN and tok in SOFT_BREAK_TOKENS
            discourse_break = tok in HARD_BREAK_TOKENS

            if hard_len_break or soft_break or discourse_break:
                raw_text = " ".join(sentence_tokens)
                norm_text = normalize_early_modern(raw_text)
                sentences.append((doc_id, sentence_id, raw_text, norm_text))
                bounds.append((doc_id, sentence_id, start_idx, idx))
                sentence_id += 1
                sentence_tokens = []
                start_idx = idx + 1

        if sentence_tokens:
            raw_text = " ".join(sentence_tokens)
            norm_text = normalize_early_modern(raw_text)
            sentences.append((doc_id, sentence_id, raw_text, norm_text))
            bounds.append((doc_id, sentence_id, start_idx, tokens[-1][0]))

        return sentences, bounds

    all_sentences, all_bounds = [], []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_doc, doc_id, s, e): doc_id for doc_id, s, e in docs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Building sentences"):
            try:
                sentences, bounds = future.result()
                all_sentences.extend(sentences)
                all_bounds.extend(bounds)
            except Exception as e:
                logger.error(f"Sentence build failed for {futures[future]}: {e}")

            # Commit in chunks
            while len(all_sentences) >= sent_chunk:
                batch_sentences = all_sentences[:sent_chunk]
                batch_bounds = all_bounds[:sent_chunk]

                with get_db_connection() as conn:
                    with conn.transaction():
                        copy_rows(conn, "sentences",
                                  ["doc_id", "sentence_id", "sentence_text_raw", "sentence_text_norm"],
                                  batch_sentences)

                        # Assign sentence IDs back to tokens
                        with conn.cursor() as cur:
                            cur.execute("""
                                CREATE TEMP TABLE tmp_sentence_bounds (
                                    doc_id TEXT,
                                    sentence_id INT,
                                    start_idx INT,
                                    end_idx INT
                                ) ON COMMIT DROP
                            """)
                            buf: Any = io.StringIO()
                            for doc_id, sentence_id, start, end in batch_bounds:
                                buf.write(f"{doc_id}\t{sentence_id}\t{start}\t{end}\n")
                            buf.seek(0)
                            cur.copy("COPY tmp_sentence_bounds (doc_id, sentence_id, start_idx, end_idx) FROM STDIN", buf)

                            cur.execute("""
                                UPDATE tokens t
                                SET sentence_id = b.sentence_id
                                FROM tmp_sentence_bounds b
                                WHERE t.doc_id = b.doc_id
                                  AND t.token_idx BETWEEN b.start_idx AND b.end_idx
                            """)

                all_sentences = all_sentences[sent_chunk:]
                all_bounds = all_bounds[sent_chunk:]

    # Commit any remaining sentences
    if all_sentences:
        with get_db_connection() as conn:
            with conn.transaction():
                copy_rows(conn, "sentences",
                          ["doc_id", "sentence_id", "sentence_text_raw", "sentence_text_norm"],
                          all_sentences)
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TEMP TABLE tmp_sentence_bounds (
                            doc_id TEXT,
                            sentence_id INT,
                            start_idx INT,
                            end_idx INT
                        ) ON COMMIT DROP
                    """)
                    buf = io.StringIO()
                    for doc_id, sentence_id, start, end in all_bounds:
                        buf.write(f"{doc_id}\t{sentence_id}\t{start}\t{end}\n")
                    buf.seek(0)
                    cur.copy("COPY tmp_sentence_bounds (doc_id, sentence_id, start_idx, end_idx) FROM STDIN", buf)
                    cur.execute("""
                        UPDATE tokens t
                        SET sentence_id = b.sentence_id
                        FROM tmp_sentence_bounds b
                        WHERE t.doc_id = b.doc_id
                          AND t.token_idx BETWEEN b.start_idx AND b.end_idx
                    """)


# Embeddings + Clustering
def embed_texts(texts: list[str], batch_size: int = config.INGEST_BATCH_SIZE, max_length: int = 128) -> np.ndarray:
    if not texts:
        return np.empty((0, model.config.hidden_size))
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            out = model(**enc)
            emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(emb)
    return np.vstack(embeddings)

# Token window builder (streaming, slice-aware)
def build_token_windows_stream(rows, window_size: int = config.INGEST_TOKEN_WINDOW_FALLBACK):
    """
    Incremental generator yielding token contexts one by one.
    Each row: (doc_id, token_idx, token, sentence, slice_start, slice_end)
    """
    doc_tokens = defaultdict(list)

    # Group rows by doc_id incrementally
    for doc_id, token_idx, token, sentence, slice_start, slice_end in rows:
        doc_tokens[doc_id].append((token_idx, token, sentence, slice_start, slice_end))

    for doc_id, tokens in doc_tokens.items():
        tokens.sort(key=lambda x: x[0])
        for i, (token_idx, token, sentence, slice_start, slice_end) in enumerate(tokens):
            if sentence:
                text = sentence
            else:
                start = max(0, i - window_size)
                end = min(len(tokens), i + window_size + 1)
                text = " ".join(t[1] for t in tokens[start:end])
            yield {
                "doc_id": doc_id,
                "token_idx": token_idx,
                "surface_form": token,
                "text": text,
                "slice_start": slice_start,
                "slice_end": slice_end
            }


# Phase 3 — Canonicalisation (streaming, agglomerative)
def canonicalize_tokens_streaming_embeddings(
    batch_size: int = config.INGEST_BATCH_SIZE,
    window_size: int = config.INGEST_TOKEN_WINDOW_FALLBACK,
    embed_batch_size: int = config.EMBED_BATCH_SIZE,
    similarity_threshold: float = 0.85,
    commit_chunk: int = 100_000,  # commit every N canonical forms
):
    """
    Streaming canonicalisation with embeddings & agglomerative clustering.

    Commits canonical forms in batches to avoid huge transactions.
    """
    logger.info("[PHASE 3] Canonicalising tokens with embeddings (agglomerative, slice-aware)")

    canonical_rows: list[tuple[str, int, str]] = []

    with get_db_connection() as conn:
        with conn.cursor(name="token_cursor") as cur:
            cur.itersize = batch_size
            cur.execute("""
                SELECT
                    t.doc_id,
                    t.token_idx,
                    t.token,
                    s.sentence_text_norm,
                    d.slice_start,
                    d.slice_end
                FROM tokens t
                JOIN documents d ON t.doc_id = d.doc_id
                LEFT JOIN sentences s
                  ON t.doc_id = s.doc_id AND t.sentence_id = s.sentence_id
                ORDER BY t.doc_id, t.token_idx
            """)

            current_batch = []
            for row in tqdm(cur, desc="Streaming tokens"):
                current_batch.append(row)
                if len(current_batch) >= batch_size:
                    process_batch_embeddings_agglomerative(
                        current_batch,
                        canonical_rows,
                        window_size,
                        embed_batch_size,
                        similarity_threshold
                    )
                    current_batch.clear()

                    # Commit if we have enough canonical rows
                    while len(canonical_rows) >= commit_chunk:
                        batch_to_commit = canonical_rows[:commit_chunk]
                        with get_db_connection() as batch_conn:
                            batch_update_canonical_dicts(batch_to_commit, batch_conn)
                            batch_conn.commit()
                        canonical_rows = canonical_rows[commit_chunk:]

            # Process leftover batch
            if current_batch:
                process_batch_embeddings_agglomerative(
                    current_batch,
                    canonical_rows,
                    window_size,
                    embed_batch_size,
                    similarity_threshold
                )

        # Commit any remaining canonical forms
        if canonical_rows:
            with get_db_connection() as batch_conn:
                batch_update_canonical_dicts(canonical_rows, batch_conn)
                batch_conn.commit()

    logger.info("[DONE] Canonicalisation complete (agglomerative, slice-aware).")


def process_batch_embeddings_agglomerative(
    rows,
    canonical_rows: list[tuple[str, int, str]],
    window_size,
    embed_batch_size,
    similarity_threshold: float = 0.85,
):
    """
    Slice-aware agglomerative canonicalisation.

    Groups by (slice, surface_form), clusters contexts,
    assigns canonical per cluster.
    """

    from sklearn.cluster import AgglomerativeClustering
    from collections import Counter

    # Build token windows
    contexts = list(build_token_windows_stream(rows, window_size))

    # Group by (slice_start, slice_end, surface_form)
    by_key = defaultdict(list)
    for ctx in contexts:
        key = (
            ctx.get("slice_start"),
            ctx.get("slice_end"),
            ctx["surface_form"]
        )
        by_key[key].append(ctx)

    for (_, _, surface_form), group in by_key.items():  # ignore slice_start to fix B007
        if len(group) == 1:
            ctx = group[0]
            canonical_rows.append(
                (ctx["doc_id"], ctx["token_idx"], surface_form)
            )
            continue

        texts = [ctx["text"] for ctx in group]

        # Embed
        embeddings = []
        for i in range(0, len(texts), embed_batch_size):
            batch = texts[i:i + embed_batch_size]
            with torch.no_grad():
                enc = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(device)
                out = model(**enc)
                emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(emb)

        X = np.vstack(embeddings)

        # Cluster
        clustering = AgglomerativeClustering(
            metric="cosine",
            linkage="average",
            distance_threshold=1 - similarity_threshold,
            n_clusters=None
        )
        labels = clustering.fit_predict(X)

        # Canonical per cluster
        clusters = defaultdict(list)
        for ctx, label in zip(group, labels, strict=True):
            clusters[label].append(ctx)

        for members in clusters.values():
            counts = Counter(ctx["surface_form"] for ctx in members)
            canonical = counts.most_common(1)[0][0]

            for ctx in members:
                canonical_rows.append(
                    (ctx["doc_id"], ctx["token_idx"], canonical)
                )


def log_post_copy_counts(conn, doc_ids: set[str], context: str) -> None:
    if not doc_ids:
        logger.debug("POST-COPY COUNTS (%s): no doc_ids supplied", context)
        return

    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM documents WHERE doc_id = ANY(%s)",
            (list(doc_ids),),
        )
        row = cur.fetchone()
        doc_count = row[0] if row is not None else 0

        cur.execute(
            "SELECT COUNT(*) FROM tokens WHERE doc_id = ANY(%s)",
            (list(doc_ids),),
        )
        row = cur.fetchone()
        tok_count = row[0] if row is not None else 0

    logger.debug(
        "POST-COPY COUNTS (%s): documents=%d tokens=%d",
        context,
        doc_count,
        tok_count,
    )


def extract_year(date_raw: str | None) -> Optional[int]:
    """
    Extract a 4-digit year from a raw date string.
    Returns an int if found, else None.
    """
    if not date_raw:
        return None
    match = re.search(r"\b(\d{4})\b", date_raw)
    if not match:
        return None
    return int(match.group(1))


def ingest_xml_parallel(max_workers: int = 4, batch_size: int = 500) -> None:
    """
    Ingest XML files in parallel and write documents + tokens to DB in batches.
    Commits per batch to avoid huge transactions. Compatible with psycopg v3.
    """

    xml_files: list[Path] = list(Path(config.INPUT_DIR).glob("*.xml"))
    if MAX_DOCS:
        xml_files = xml_files[:MAX_DOCS]

    all_docs: list[dict] = []
    all_tokens: list[tuple[str, int, str]] = []

    def process_and_collect(xml_path: Path):
        res = process_file_worker(xml_path)
        if res:
            doc_meta, tokens = res
            return doc_meta, tokens
        logger.warning(f"File rejected: {xml_path.name}")
        return None

    # --- Parallel XML parsing ---
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_and_collect, f): f for f in xml_files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Ingesting XML"):
            try:
                result = future.result()
                if result:
                    doc_meta, tokens = result
                    all_docs.append(doc_meta)
                    all_tokens.extend([(doc_meta["doc_id"], i, tok) for i, tok in tokens])
                    logger.debug(f"Collected {len(all_tokens)} tokens from {len(all_docs)} docs")
            except Exception as e:
                logger.error(f"Failed to process {futures[future]}: {e}")

    # --- Single connection for all DB writes ---
    if not all_docs:
        logger.warning("No valid documents found for ingestion.")
        return

    logger.info(f"Starting database insertion: {len(all_docs)} docs, {len(all_tokens)} tokens")

    with get_db_connection() as conn:
        with conn.transaction():
            while all_docs:
                batch_docs = all_docs[:batch_size]
                batch_doc_ids = {d["doc_id"] for d in batch_docs}
                batch_tokens = [t for t in all_tokens if t[0] in batch_doc_ids]

                # Prepare documents rows
                doc_rows = [
                    [
                        d["doc_id"],
                        d["title"],
                        d["author"],
                        extract_year(d["source_date_raw"]),
                        d["publisher"],
                        d["pub_place"],
                        d["source_date_raw"],
                        len([t for t in batch_tokens if t[0] == d["doc_id"]]),
                        d["slice_start"],
                        d["slice_end"],
                    ]
                    for d in batch_docs
                ]

                # COPY documents
                copy_rows(
                    conn,
                    "documents",
                    [
                        "doc_id",
                        "title",
                        "author",
                        "pub_year",
                        "publisher",
                        "pub_place",
                        "source_date_raw",
                        "token_count",
                        "slice_start",
                        "slice_end",
                    ],
                    doc_rows
                )

                # COPY tokens
                copy_rows(
                    conn,
                    "tokens",
                    ["doc_id", "token_idx", "token"],
                    [[t[0], t[1], t[2]] for t in batch_tokens]
                )

                # Log post-COPY counts
                log_post_copy_counts(conn, batch_doc_ids, context="batch")

                # Remove committed batch
                all_docs = all_docs[batch_size:]
                all_tokens = [t for t in all_tokens if t[0] not in batch_doc_ids]

    logger.info("XML ingestion complete and committed to database.")



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices={"ingest", "sentences", "canonicalize", "all"}, help="Pipeline phase to run")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of documents to process (for testing)")
    args = parser.parse_args()

    global MAX_DOCS
    MAX_DOCS = args.limit

    logger.info(f"Starting pipeline phase: {args.phase}")

    if args.phase == "all":
        logger.info("Dropping existing token indexes...")
        eebo_db.drop_token_indexes()

        logger.info("Ingesting XML documents...")
        ingest_xml_parallel()

        logger.info("Building sentences...")
        build_sentences_parallel()

        logger.info("Canonicalising tokens...")
        canonicalize_tokens_streaming_embeddings()

        logger.info("Creating token indexes...")
        eebo_db.create_token_indexes()

        logger.info("Creating tiered token indexes...")
        eebo_db.create_tiered_token_indexes()

    elif args.phase == "ingest":
        logger.info("Dropping existing token indexes...")
        eebo_db.drop_token_indexes()
        ingest_xml_parallel()
        logger.info("Creating token indexes...")
        eebo_db.create_token_indexes()

    elif args.phase == "sentences":
        build_sentences_parallel()

    elif args.phase == "canonicalize":
        canonicalize_tokens_streaming_embeddings()

    logger.info("Pipeline phase completed successfully.")


if __name__ == "__main__":
    main()
