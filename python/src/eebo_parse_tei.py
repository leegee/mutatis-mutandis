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


# DB utilities
def copy_rows(conn, table, columns, rows):
    """COPY rows into Postgres using TEXT format"""
    buf = io.StringIO()
    for row in rows:
        buf.write("\t".join(
            "\\N" if v is None else str(v).replace("\t", " ").replace("\n", " ")
            for v in row
        ))
        buf.write("\n")
    buf.seek(0)
    with conn.cursor() as cur:
        cur.copy(
            f"COPY {table} ({', '.join(columns)}) FROM STDIN WITH (FORMAT text)",
            buf
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
def build_sentences_parallel(max_workers: int = 4) -> None:
    """
    Phase 2: Build sentences and assign sentence_id back to tokens.

    Features:
      - Hard cap on sentence length
      - Soft breaks on conjunctions
      - Discourse-marker hard breaks
      - Parallel-safe DB access
      - Bulk COPY for sentences and token updates
    """

    logger.info("[PHASE 2] Building sentences and assigning sentence IDs...")

    # Fetch documents with slice info (slice not used yet, but future-safe)
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

    # Sentence heuristics
    MAX_SENT_LEN = 60
    MIN_SENT_LEN = 20

    SOFT_BREAK_TOKENS = {"and", "but", "for", "or"}
    HARD_BREAK_TOKENS = {"wherefore", "therefore", "thus"}

    def process_doc(doc_id: str, slice_start: int | None, slice_end: int | None):
        """
        Returns:
          sentences: (doc_id, sentence_id, raw_text, norm_text)
          bounds:    (doc_id, sentence_id, start_idx, end_idx)
        """

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

        sentences = []
        bounds = []

        sentence_tokens: list[str] = []
        sentence_id = 0
        start_idx = tokens[0][0]

        for idx, tok in tokens:
            sentence_tokens.append(tok)
            token_count = len(sentence_tokens)

            # Boundary rules
            hard_len_break = token_count >= MAX_SENT_LEN
            soft_break = token_count >= MIN_SENT_LEN and tok in SOFT_BREAK_TOKENS
            discourse_break = tok in HARD_BREAK_TOKENS

            if hard_len_break or soft_break or discourse_break:
                raw_text = " ".join(sentence_tokens)
                norm_text = normalize_early_modern(raw_text)

                sentences.append(
                    (doc_id, sentence_id, raw_text, norm_text)
                )
                bounds.append(
                    (doc_id, sentence_id, start_idx, idx)
                )

                sentence_id += 1
                sentence_tokens = []
                start_idx = idx + 1

        # Flush trailing sentence
        if sentence_tokens:
            raw_text = " ".join(sentence_tokens)
            norm_text = normalize_early_modern(raw_text)

            sentences.append(
                (doc_id, sentence_id, raw_text, norm_text)
            )
            bounds.append(
                (doc_id, sentence_id, start_idx, tokens[-1][0])
            )

        return sentences, bounds

    all_sentences: list[tuple[str, int, str, str]] = []
    all_bounds: list[tuple[str, int, int, int]] = []

    # Parallel sentence construction
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_doc, doc_id, slice_start, slice_end): doc_id
            for doc_id, slice_start, slice_end in docs
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Building sentences"
        ):
            try:
                sentences, bounds = future.result()
                all_sentences.extend(sentences)
                all_bounds.extend(bounds)
            except Exception as e:
                logger.error(
                    f"Sentence build failed for {futures[future]}: {e}"
                )

    # Bulk write to DB
    with get_db_connection() as conn:
        with conn.transaction():

            # Insert sentences
            SENT_CHUNK = 50_000
            for i in range(0, len(all_sentences), SENT_CHUNK):
                copy_rows(
                    conn,
                    "sentences",
                    ["doc_id", "sentence_id",
                     "sentence_text_raw", "sentence_text_norm"],
                    all_sentences[i:i + SENT_CHUNK]
                )

            # Bulk assign sentence_id back to tokens
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
                for doc_id, sentence_id, start, end in all_bounds:
                    buf.write(f"{doc_id}\t{sentence_id}\t{start}\t{end}\n")
                buf.seek(0)

                cur.copy(
                    "COPY tmp_sentence_bounds (doc_id, sentence_id, start_idx, end_idx) FROM STDIN",
                    buf
                )

                cur.execute("""
                    UPDATE tokens t
                    SET sentence_id = b.sentence_id
                    FROM tmp_sentence_bounds b
                    WHERE t.doc_id = b.doc_id
                      AND t.token_idx BETWEEN b.start_idx AND b.end_idx
                """)

    logger.info(
        "[PHASE 2] Sentences built with length caps, discourse breaks, "
        "and bulk sentence_id assignment."
    )


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
):
    """
    Streaming canonicalisation with embeddings & agglomerative clustering.

    - Server-side cursor
    - Incremental token windows
    - Embeddings in batches
    - Agglomerative clustering per slice × surface_form
    - Write canonical forms via COPY in batches
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

            # Process leftover batch
            if current_batch:
                process_batch_embeddings_agglomerative(
                    current_batch,
                    canonical_rows,
                    window_size,
                    embed_batch_size,
                    similarity_threshold
                )

        # Bulk write canonical forms to DB
        if canonical_rows:
            logger.info(f"Writing {len(canonical_rows)} canonical forms to DB in batches...")
            chunk_size = 100_000
            for i in range(0, len(canonical_rows), chunk_size):
                batch_update_canonical_dicts(canonical_rows[i:i+chunk_size], conn)
            conn.commit()

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


def ingest_xml_parallel(max_workers: int = 4):
    """
    Ingest XML files from config.INPUT_DIR and write documents + tokens to DB.
    Parallel-safe version using ThreadPoolExecutor.
    """
    xml_files = list(Path(config.INPUT_DIR).glob("*.xml"))
    if MAX_DOCS:
        xml_files = xml_files[:MAX_DOCS]

    all_docs = []
    all_tokens = []

    def process_and_collect(xml_path):
        res = process_file_worker(xml_path)
        if res:
            doc_meta, tokens = res
            return doc_meta, tokens
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_and_collect, f): f for f in xml_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Ingesting XML"):
            try:
                result = future.result()
                if result:
                    doc_meta, tokens = result
                    all_docs.append(doc_meta)
                    all_tokens.extend([(doc_meta["doc_id"], i, tok) for i, tok in tokens])
            except Exception as e:
                logger.error(f"Failed to process {futures[future]}: {e}")

    # Bulk insert
    with get_db_connection() as conn:
        with conn.transaction():
            # Insert documents
            copy_rows(conn, "documents",
                      ["doc_id", "title", "author", "publisher", "pub_place", "source_date_raw", "slice_start", "slice_end"],
                      [[d[k] for k in ["doc_id", "title", "author", "publisher", "pub_place", "source_date_raw", "slice_start", "slice_end"]]
                       for d in all_docs])
            # Insert tokens
            copy_rows(conn, "tokens", ["doc_id", "token_idx", "token"], all_tokens)


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
