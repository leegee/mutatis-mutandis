#!/usr/bin/env python
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from typing import Optional
import argparse
import numpy as np
import re
import torch
import unicodedata

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

dbh = eebo_db.dbh


def normalize_early_modern(text: str) -> str:
    """
    Normalize Early Modern English text for ingestion/canonicalisation.
    """
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
    """
    Assigns a document to a slice based on its publication year.

    Args:
        date_raw: raw date string from TEI metadata

    Returns:
        tuple(slice_start, slice_end) or (None, None) if unknown
    """
    if not date_raw:
        return None, None

    # Attempt to extract a 4-digit year
    match = re.search(r"\b(\d{4})\b", date_raw)
    if not match:
        return None, None

    year = int(match.group(1))
    for start, end in config.SLICES:
        if start <= year <= end:
            return start, end

    # Year outside defined slices
    return None, None


def process_file(xml_path: Path, conn) -> bool:
    """
    Parse a TEI XML file and insert into documents/tokens tables.
    Returns True if processed successfully, False if rejected.
    """
    import xml.etree.ElementTree as etree

    try:
        tree = etree.parse(str(xml_path))
    except Exception as e:
        logger.error(f"XML parse failed: {xml_path.name}: {e}")
        return False

    doc_id_elem = tree.find(".//HEADER//IDNO[@TYPE='DLPS']")
    doc_id = doc_id_elem.text.strip() if (doc_id_elem is not None and doc_id_elem.text) else None
    if not doc_id:
        logger.error(f"XML rejected: {xml_path.name}: missing document ID")
        return False

    # Extract metadata
    title_elem = tree.find(".//HEADER//TITLESTMT/TITLE")
    author_elem = tree.find(".//HEADER//TITLESTMT/AUTHOR")
    date_elem = tree.find(".//HEADER//SOURCEDESC//DATE")
    pub_elem = tree.find(".//HEADER//SOURCEDESC//PUBLISHER")
    place_elem = tree.find(".//HEADER//SOURCEDESC//PUBPLACE")

    title = title_elem.text.strip() if (title_elem is not None and title_elem.text) else None
    author = author_elem.text.strip() if (author_elem is not None and author_elem.text) else None
    date_raw = date_elem.text.strip() if (date_elem is not None and date_elem.text) else None
    publisher = pub_elem.text.strip() if (pub_elem is not None and pub_elem.text) else None
    pub_place = place_elem.text.strip() if (place_elem is not None and place_elem.text) else None

    slice_start, slice_end = assign_slice(date_raw)

    # Insert into documents
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO documents (
                doc_id, title, author, pub_year,
                publisher, pub_place, source_date_raw,
                slice_start, slice_end
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (doc_id) DO UPDATE SET
                title = EXCLUDED.title,
                author = EXCLUDED.author,
                pub_year = EXCLUDED.pub_year,
                publisher = EXCLUDED.publisher,
                pub_place = EXCLUDED.pub_place,
                source_date_raw = EXCLUDED.source_date_raw,
                slice_start = EXCLUDED.slice_start,
                slice_end = EXCLUDED.slice_end
        """, (doc_id, title, author, None, publisher, pub_place, date_raw, slice_start, slice_end))

    # Extract body text
    body_elem = tree.find(".//EEBO//TEXT//BODY")
    if body_elem is None:
        logger.error(f"XML rejected: {xml_path.name}: no body text")
        return False

    raw_text = " ".join(t.strip() for t in body_elem.itertext() if t.strip())
    fixed_text = eebo_ocr_fixes.apply_ocr_fixes(raw_text)
    normalized = normalize_early_modern(fixed_text)

    if len(normalized) < 100:
        logger.error(f"XML rejected: {xml_path.name}: text too short")
        return False

    # Insert tokens
    tokens = normalized.split()
    with conn.cursor() as cur:
        for idx, tok in enumerate(tokens):
            cur.execute("""
                INSERT INTO tokens (doc_id, token_idx, token)
                VALUES (%s,%s,%s)
            """, (doc_id, idx, tok))

    return True

# ---------------------------------------------------------------------
# Phase 1 — INGEST XML
# ---------------------------------------------------------------------

def ingest_xml() -> None:
    """
    Phase 1: Parse TEI XML and populate documents / tokens tables.
    Respects MAX_DOCS if set.
    """
    xml_files = list(config.INPUT_DIR.glob("*.xml"))
    if MAX_DOCS:
        xml_files = xml_files[:MAX_DOCS]

    print(f"[PHASE 1] Found {len(xml_files)} XML files to ingest...")

    processed = 0
    for xml_file in tqdm(xml_files, desc="Ingesting XML"):
        # call your existing TEI ingestion routine from previous scripts
        # for example:
        try:
            if process_file(xml_file, dbh):
                processed += 1
        except Exception as e:
            print(f"[ERROR] Failed to process {xml_file.name}: {e}")

    dbh.commit()
    print(f"[PHASE 1] Ingested {processed} documents")

# ---------------------------------------------------------------------
# Phase 2 — BUILD SENTENCES
# ---------------------------------------------------------------------

def build_sentences() -> None:
    """
    Phase 2: Derive sentence boundaries from tokens.
    Optional; canonicalisation can fall back to token windows.
    Respects MAX_DOCS if set.
    """
    print("[PHASE 2] Building sentences...")
    with dbh.cursor() as cur:
        cur.execute("SELECT DISTINCT doc_id FROM tokens ORDER BY doc_id")
        doc_ids = [row[0] for row in cur.fetchall()]

    if MAX_DOCS:
        doc_ids = doc_ids[:MAX_DOCS]

    for doc_id in tqdm(doc_ids, desc="Building sentences"):
        # naive sentence splitter fallback
        cur.execute("""
            SELECT token_idx, token
            FROM tokens
            WHERE doc_id = %s
            ORDER BY token_idx
        """, (doc_id,))
        tokens = cur.fetchall()

        sentences = []
        sentence_text = []
        sentence_id = 0

        for _idx, tok in tokens:
            sentence_text.append(tok)
            if tok.endswith((".", "!", "?")):
                sentences.append((sentence_id, " ".join(sentence_text)))
                sentence_text = []
                sentence_id += 1

        # catch any trailing tokens
        if sentence_text:
            sentences.append((sentence_id, " ".join(sentence_text)))

        # write sentences to DB
        for sid, text in sentences:
            normalized = normalize_early_modern(text)
            cur.execute("""
                INSERT INTO sentences(doc_id, sentence_id, sentence_text_raw, sentence_text_norm)
                VALUES (%s,%s,%s,%s)
                ON CONFLICT(doc_id, sentence_id) DO UPDATE
                    SET sentence_text_raw = EXCLUDED.sentence_text_raw, sentence_text_norm = EXCLUDED.sentence_text_norm
            """, (doc_id, sid, text, normalized))

    dbh.commit()
    print("[PHASE 2] Sentences built")

# ---------------------------------------------------------------------
# Embedding utilities
# ---------------------------------------------------------------------

def embed_texts(
    texts: list[str],
    batch_size: int = config.INGEST_BATCH_SIZE,
    max_length: int = 128
) -> np.ndarray:
    if not texts:
        return np.empty((0, model.config.hidden_size))

    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)

            out = model(**enc)
            emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(emb)

    return np.vstack(embeddings)

# ---------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------

def cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: Optional[int] = None,
    distance_threshold: float = 1.0
) -> np.ndarray:
    if embeddings.shape[0] == 0:
        return np.empty((0,), dtype=int)

    if n_clusters is None:
        clustering = AgglomerativeClustering(
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average"
        )
    else:
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average"
        )

    labels = np.asarray(clustering.fit_predict(embeddings), dtype=int)
    return labels

# ---------------------------------------------------------------------
# Token context builder
# ---------------------------------------------------------------------

def build_token_windows(rows, window_size: int = config.INGEST_TOKEN_WINDOW_FALLBACK):
    contexts = []
    doc_tokens = defaultdict(list)

    for doc_id, token_idx, token, sentence in rows:
        doc_tokens[doc_id].append((token_idx, token, sentence))

    for doc_id, tokens in doc_tokens.items():
        tokens.sort(key=lambda x: x[0])
        for i, (token_idx, token, sentence) in enumerate(tokens):
            if sentence:
                text = sentence
            else:
                start = max(0, i - window_size)
                end = min(len(tokens), i + window_size + 1)
                text = " ".join(t[1] for t in tokens[start:end])
            contexts.append({
                "doc_id": doc_id,
                "token_idx": token_idx,
                "surface_form": token,
                "text": text
            })

    return contexts

# ---------------------------------------------------------------------
# Phase 3 — CANONICALISE
# ---------------------------------------------------------------------

def canonicalize_tokens() -> None:
    print("[PHASE 3] Canonicalising tokens...")

    with dbh.cursor() as cur:
        cur.execute("""
            SELECT t.doc_id,
                   t.token_idx,
                   t.token,
                   s.sentence_text_norm
            FROM tokens t
            LEFT JOIN sentences s
              ON t.doc_id = s.doc_id
             AND t.sentence_id = s.sentence_id
            ORDER BY t.doc_id, t.token_idx
        """)
        rows = cur.fetchall()

    if not rows:
        print("[INFO] No tokens found — skipping canonicalisation.")
        return

    contexts = build_token_windows(rows)
    texts = [c["text"] for c in contexts]
    print(f"[INFO] Embedding {len(texts)} contexts...")
    embeddings = embed_texts(texts)

    print("[INFO] Clustering embeddings...")
    labels = cluster_embeddings(embeddings)

    clusters: dict[int, list[str]] = {}
    for ctx, label in zip(contexts, labels, strict=True):
        clusters.setdefault(label, []).append(ctx["surface_form"])

    canonical_map = {label: max(set(forms), key=forms.count) for label, forms in clusters.items()}

    for ctx, label in zip(contexts, labels, strict=True):
        ctx["canonical"] = canonical_map[label]

    print("[INFO] Writing canonical forms back to tokens table...")
    with dbh.cursor() as cur:
        for ctx in tqdm(contexts):
            cur.execute("""
                UPDATE tokens
                SET canonical = %s
                WHERE doc_id = %s
                  AND token_idx = %s
            """, (ctx["canonical"], ctx["doc_id"], ctx["token_idx"]))

    dbh.commit()
    print("[DONE] Canonicalisation complete.")

# ---------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices={"ingest", "sentences", "canonicalize", "all"},
        help="Pipeline phase to run"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of documents to process (for testing)"
    )
    args = parser.parse_args()

    global MAX_DOCS
    MAX_DOCS = args.limit

    if args.phase in {"ingest", "all"}:
        eebo_db.drop_token_indexes()
        eebo_db.create_token_indexes()
        ingest_xml()
    if args.phase in {"sentences", "all"}:
        build_sentences()
    if args.phase in {"canonicalize", "all"}:
        canonicalize_tokens()


if __name__ == "__main__":
    main()
