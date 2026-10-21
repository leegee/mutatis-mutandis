"""
passages_2_0_colab_embed_passages_windows.py

Colab job: full-token parquet (Drive) -> token-window parquet shards (Drive).

No Postgres, no LanceDB. Exports, for each not-yet-embedded document, only the
token windows around target terms. Original token_idx values are preserved, so
gaps between windows remain visible to the embed job (which never builds a
passage across a gap).

Each shard is written to local /content, copied to Drive as .part, validated,
then renamed. Re-running skips documents already exported or already embedded
and continues shard numbering after the last existing shard.

    !pip install -q pyarrow
"""

import os
import re
import shutil
from datetime import datetime
from pathlib import Path

from google.colab import drive
import pyarrow as pa
import pyarrow.parquet as pq

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

BASE = Path("/content/drive/MyDrive/tier1")

# Full-token input. Either layout is accepted (auto-detected):
#   long : one row per token  -> corpus, doc_id, pub_year, token_idx, token
#          (must be sorted by corpus, doc_id, token_idx, across files too)
#   lists: one row per doc    -> corpus, doc_id, pub_year, token_idx[], tokens[]
SOURCE_DIR = BASE / "source_tokens"
SOURCE_GLOB = "*.parquet"

# Optional replacement for CORPUS_SQL: a parquet or CSV on Drive with columns
# corpus, doc_id listing the documents to export. None = every document that
# contains at least one target term.
SELECTED_DOCS_PATH = None  # eg BASE / "selected_docs.parquet"

EMB_OUTPUT_DIR = BASE / "tokens"          # read by the embed job (IN_DIR)
PASSAGES_DIR = BASE / "passages"          # embed job output (OUT_DIR)
LOCAL_TMP = Path("/content/tmp")

DOCS_PER_SHARD = 500
WINDOW_WORDS = 100 # Must match the data-generator script passages_1_export_windows.py WINDOW_WORDS

# An even window has one more token to the left than to the right.
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


def log(message):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}",
          flush=True)


# ------------------------------------------------------------
# Drive-safe writing
# ------------------------------------------------------------

def commit_file(local_path, final_path):
    """Copy to Drive under a .part name, validate, then publish."""
    final_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = final_path.with_name(final_path.name + ".part")
    if part_path.exists():
        part_path.unlink()
    shutil.copyfile(local_path, part_path)
    pq.ParquetFile(part_path).metadata  # raises if the copy is unreadable
    os.replace(part_path, final_path)


def flush(docs, shard_no):
    table = pa.Table.from_pylist(docs, schema=SCHEMA)
    name = f"tokens_{shard_no:05d}.parquet"
    local = LOCAL_TMP / name
    pq.write_table(table, local, compression="zstd")
    commit_file(local, EMB_OUTPUT_DIR / name)
    local.unlink(missing_ok=True)
    log(f"Wrote {name} ({len(docs)} docs)")


# ------------------------------------------------------------
# Resume state (replaces the LanceDB "already indexed" lookup)
# ------------------------------------------------------------

def read_doc_keys(path):
    t = pq.read_table(path, columns=["corpus", "doc_id"])
    return set(zip(t["corpus"].to_pylist(), t["doc_id"].to_pylist()))


def already_done():
    """Documents already exported to token shards or already embedded."""
    done = set()

    # Previously exported shards (makes re-runs idempotent).
    for p in sorted(EMB_OUTPUT_DIR.glob("tokens_*.parquet")):
        try:
            done |= read_doc_keys(p)
        except Exception:
            log(f"WARNING: unreadable existing shard {p.name}; ignoring it")

    # Embedded passages: final shards and per-document checkpoints.
    embedded = list(PASSAGES_DIR.glob("passages_*.parquet"))
    embedded += list((PASSAGES_DIR / "_checkpoints").glob("*/doc_*.parquet"))
    for p in embedded:
        try:
            done |= read_doc_keys(p)
        except Exception:
            log(f"WARNING: unreadable passage file {p}; ignoring it")

    log(f"{len(done)} documents already exported or embedded")
    return done


def next_shard_no():
    nums = [
        int(m.group(1))
        for p in EMB_OUTPUT_DIR.glob("tokens_*.parquet")
        if (m := re.fullmatch(r"tokens_(\d+)\.parquet", p.name))
    ]
    return max(nums) + 1 if nums else 0


def load_selected():
    if SELECTED_DOCS_PATH is None:
        return None
    p = Path(SELECTED_DOCS_PATH)
    if p.suffix == ".csv":
        import csv
        with open(p, newline="") as f:
            return {(r["corpus"], r["doc_id"]) for r in csv.DictReader(f)}
    return read_doc_keys(p)


# ------------------------------------------------------------
# Windowing
# ------------------------------------------------------------

def retained(token_idx, tokens):
    """Union of fixed-width windows around target occurrences."""
    hits = [
        i for i, t in enumerate(tokens)
        if t is not None and t.lower() in TARGET_TERMS
    ]
    if not hits:
        return [], []

    n = len(tokens)
    keep = set()
    for h in hits:
        keep.update(range(max(0, h - WINDOW_LEFT),
                          min(n, h + WINDOW_RIGHT + 1)))

    order = sorted(keep)
    return [token_idx[i] for i in order], [tokens[i] for i in order]


# ------------------------------------------------------------
# Source readers: yield (corpus, doc_id, pub_year, token_idx[], tokens[])
# ------------------------------------------------------------

def iter_docs(files):
    names = set(pq.ParquetFile(files[0]).schema_arrow.names)

    if {"token_idx", "tokens"} <= names:           # one row per document
        for f in files:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=50):
                for r in batch.to_pylist():
                    yield (r["corpus"], r["doc_id"], r["pub_year"],
                           r["token_idx"], r["tokens"])
        return

    if not {"token_idx", "token"} <= names:
        raise SystemExit(
            f"Unrecognised input schema in {files[0].name}: {sorted(names)}"
        )

    # One row per token: stream and group, carrying state across batches
    # and across files so a document may span file boundaries.
    cur_key, cur_year, idx, toks = None, None, [], []
    cols = ["corpus", "doc_id", "pub_year", "token_idx", "token"]

    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(
            batch_size=200_000, columns=cols
        ):
            d = batch.to_pydict()
            for c, did, yr, ti, tk in zip(
                d["corpus"], d["doc_id"], d["pub_year"],
                d["token_idx"], d["token"],
            ):
                key = (c, did)
                if key != cur_key:
                    if cur_key is not None:
                        yield (*cur_key, cur_year, idx, toks)
                    cur_key, cur_year, idx, toks = key, yr, [], []
                idx.append(ti)
                toks.append(tk)

    if cur_key is not None:
        yield (*cur_key, cur_year, idx, toks)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    drive.mount("/content/drive")

    EMB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_TMP.mkdir(parents=True, exist_ok=True)

    files = sorted(SOURCE_DIR.glob(SOURCE_GLOB))
    if not files:
        raise SystemExit(f"No source parquet files in {SOURCE_DIR}")
    log(f"Found {len(files)} source files in {SOURCE_DIR}")

    done = already_done()
    selected = load_selected()
    shard_no = next_shard_no()

    docs, exported, skipped, unselected = [], 0, 0, 0

    for corpus, doc_id, pub_year, token_idx, tokens in iter_docs(files):
        key = (corpus, doc_id)

        if selected is not None and key not in selected:
            unselected += 1
            continue
        if key in done:
            skipped += 1
            continue

        idx, toks = retained(token_idx, tokens)
        if not toks:
            continue

        docs.append({
            "corpus": corpus,
            "doc_id": doc_id,
            "pub_year": pub_year,
            "token_idx": idx,
            "tokens": toks,
        })
        exported += 1

        if len(docs) >= DOCS_PER_SHARD:
            flush(docs, shard_no)
            docs, shard_no = [], shard_no + 1

    if docs:
        flush(docs, shard_no)

    log(f"Done: exported {exported} documents; skipped {skipped} "
        f"already done; {unselected} not selected")


if __name__ == "__main__":
    main()
