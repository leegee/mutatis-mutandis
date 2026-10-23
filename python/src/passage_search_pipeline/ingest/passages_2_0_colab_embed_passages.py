"""
colab_embed_passages.py

Colab job: stand-alone token parquet shards (Drive) -> passage-vector parquet shards (Drive).

Be sure to sync the config with lib.corpus_config and lib.macberth

No Postgres, no Lance. Each output shard is written to local /content first,
copied to Drive under a .part name, then renamed, so a disconnect can never
leave something that looks finished but isn't. Re-running skips finished shards.

    !pip install -q transformers pyarrow

"""

import os
import shutil
import time
from datetime import datetime
from pathlib import Path

from google.colab import drive
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModel, AutoTokenizer


# Configuration

RESUME_FROM_DOC = 0
STOP_AFTER_DOC  = 9999
IN_DIR = Path("/content/drive/MyDrive/passage-pipeline/source_tokens")
OUT_DIR = Path("/content/drive/MyDrive/passage-pipeline/output-passages")
CHECKPOINT_DIR = OUT_DIR / "_checkpoints"
LOCAL_TMP = Path("/content/tmp")
MODEL_NAME = "emanjavacas/MacBERTh"
ASSEMBLE_FINAL_SHARD = True

CHUNK = 510
OVERLAP = 96
BATCH = 16                  # Lower peak GPU memory than the original 32
WINDOW = 40
HOP = 20
DIM = 768

LAYER_NAMES = ("l8", "last", "mean4")


def log(message):
    print(
        f"[{datetime.now().isoformat(timespec='seconds')}] {message}",
        flush=True,
    )



def passage_spans(n_words):
    if n_words <= WINDOW:
        return [(0, n_words)] if n_words else []

    starts = list(range(0, n_words - WINDOW + 1, HOP))
    if starts[-1] + WINDOW < n_words:
        starts.append(n_words - WINDOW)

    return [(s, s + WINDOW) for s in starts]


def pick_layers(hidden_states):
    return {
        "l8": hidden_states[8],
        "last": hidden_states[-1],
        "mean4": torch.stack(hidden_states[-4:]).mean(0),
    }


@torch.inference_mode()
def word_vectors(words, tok, model, device):
    """Return one float32 vector per word for each selected representation."""
    words = [w if w.strip() else "." for w in words]

    enc = tok(
        words,
        is_split_into_words=True,
        add_special_tokens=False,
        truncation=False,
    )
    ids = enc["input_ids"]
    wids = np.asarray(enc.word_ids(), dtype=np.int32)
    n_words = len(words)

    if len(ids) == 0:
        return {
            name: np.zeros((n_words, DIM), dtype=np.float32)
            for name in LAYER_NAMES
        }

    chunks = []
    start = 0
    while True:
        end = min(start + CHUNK, len(ids))
        chunks.append((start, end))
        if end == len(ids):
            break
        start = end - OVERLAP

    sums = {
        name: np.zeros((n_words, DIM), dtype=np.float32)
        for name in LAYER_NAMES
    }
    cnt = np.zeros(n_words, dtype=np.float32)

    # Process shorter chunks first to reduce padding.
    order = sorted(
        range(len(chunks)),
        key=lambda i: chunks[i][1] - chunks[i][0],
    )

    for b in range(0, len(order), BATCH):
        batch = [chunks[i] for i in order[b:b + BATCH]]
        maxlen = max(e - s for s, e in batch) + 2

        inp = torch.full(
            (len(batch), maxlen),
            tok.pad_token_id,
            dtype=torch.long,
        )
        att = torch.zeros((len(batch), maxlen), dtype=torch.long)

        for k, (s, e) in enumerate(batch):
            seq = [tok.cls_token_id] + ids[s:e] + [tok.sep_token_id]
            inp[k, :len(seq)] = torch.tensor(seq)
            att[k, :len(seq)] = 1

        out = model(
            input_ids=inp.to(device),
            attention_mask=att.to(device),
            output_hidden_states=True,
        )

        layers = {
            name: tensor.float().cpu().numpy()
            for name, tensor in pick_layers(out.hidden_states).items()
        }

        for k, (s, e) in enumerate(batch):
            w = wids[s:e]
            if len(w) == 0:
                continue

            first = np.flatnonzero(
                np.r_[True, w[1:] != w[:-1]]
            )
            uniq = w[first]
            sizes = np.diff(
                np.r_[first, len(w)]
            ).astype(np.float32)

            cnt[uniq] += sizes

            for name in LAYER_NAMES:
                h = layers[name][k, 1:1 + (e - s)]
                sums[name][uniq] += np.add.reduceat(
                    h, first, axis=0
                )

        del out, layers, inp, att

    cnt = np.maximum(cnt, 1.0)[:, None]
    return {name: sums[name] / cnt for name in LAYER_NAMES}


def doc_passage_vectors(words, tok, model, device, token_idx=None):
    """
    Embed passages without crossing gaps in original token positions.

    Contiguous legacy exports produce the same spans and vector ordering as
    before. Sparse exports are split into contiguous runs before windowing;
    word offsets remain global offsets into the exported arrays.
    """
    n_words = len(words)
    if not n_words:
        return [], {
            name: np.empty((0, DIM), dtype=np.float16)
            for name in LAYER_NAMES
        }

    if token_idx is None:
        # Preserve the legacy API: callers without source indices imply that
        # every adjacent array entry is also adjacent in the original text.
        token_idx = list(range(n_words))

    if len(token_idx) != n_words:
        raise ValueError(
            f"token_idx/tokens length mismatch: {len(token_idx)} != {n_words}"
        )

    # A passage must never bridge an omitted section of the source document.
    run_starts = [0]
    run_starts.extend(
        i for i in range(1, n_words)
        if token_idx[i] != token_idx[i - 1] + 1
    )
    run_ends = run_starts[1:] + [n_words]

    all_spans = []
    result = {name: [] for name in LAYER_NAMES}

    for run_start, run_end in zip(run_starts, run_ends):
        run_words = words[run_start:run_end]
        local_spans = passage_spans(len(run_words))
        if not local_spans:
            continue

        i = 0
        while i < len(local_spans):
            seg_start = local_spans[i][0]
            j = i

            while (
                j < len(local_spans)
                and local_spans[j][1] - seg_start <= 2000
            ):
                j += 1

            j = max(j, i + 1)
            seg_end = local_spans[j - 1][1]

            wv = word_vectors(
                run_words[seg_start:seg_end], tok, model, device
            )

            for name in LAYER_NAMES:
                result[name].append(
                    np.stack([
                        wv[name][s - seg_start:e - seg_start].mean(axis=0)
                        for s, e in local_spans[i:j]
                    ]).astype(np.float16)
                )

            # Convert run-local offsets to offsets in the exported token arrays.
            all_spans.extend(
                (run_start + s, run_start + e)
                for s, e in local_spans[i:j]
            )

            del wv
            i = j

    vectors = {
        name: (
            np.concatenate(parts, axis=0)
            if parts
            else np.empty((0, DIM), dtype=np.float16)
        )
        for name, parts in result.items()
    }

    return all_spans, vectors


def fixed_f16(arr):
    arr = np.ascontiguousarray(arr, dtype=np.float16)
    return pa.FixedSizeListArray.from_arrays(
        pa.array(arr.ravel()), DIM
    )


def document_table(doc, tok, model, device):
    """Create a Parquet table for one document."""
    words = doc["tokens"]
    tidx = doc["token_idx"]

    if not words:
        return None

    spans, vectors = doc_passage_vectors(
        words, tok, model, device, token_idx=tidx
    )

    if not spans:
        return None

    arrays = {
        "corpus": pa.array([doc["corpus"]] * len(spans)),
        "doc_id": pa.array([doc["doc_id"]] * len(spans)),
        "pub_year": pa.array(
            [doc["pub_year"]] * len(spans), type=pa.int32()
        ),
        "word_start": pa.array(
            [s for s, e in spans], type=pa.int32()
        ),
        "word_end": pa.array(
            [e for s, e in spans], type=pa.int32()
        ),
        "token_start": pa.array(
            [tidx[s] for s, e in spans], type=pa.int32()
        ),
        "token_end": pa.array(
            [tidx[e - 1] + 1 for s, e in spans],
            type=pa.int32(),
        ),
    }

    for name in LAYER_NAMES:
        arrays[f"vec_{name}"] = fixed_f16(vectors[name])

    return pa.table(arrays)


def commit_file(local_path, final_path):
    """Copy to Drive under a temporary name, then publish the final name."""
    final_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = final_path.with_name(final_path.name + ".part")

    if part_path.exists():
        part_path.unlink()

    shutil.copyfile(local_path, part_path)

    # Validate the Drive copy before committing it.
    pq.ParquetFile(part_path).metadata
    os.replace(part_path, final_path)


def valid_checkpoint(path):
    """Trust only readable checkpoints with the expected schema."""
    try:
        pf = pq.ParquetFile(path)
        names = set(pf.schema_arrow.names)
        required = {
            "corpus", "doc_id", "pub_year",
            "word_start", "word_end",
            "token_start", "token_end",
            "vec_l8", "vec_last", "vec_mean4",
        }
        return pf.metadata.num_rows > 0 and required.issubset(names)
    except Exception:
        return False


def process_shard(in_path, tok, model, device):
    shard_name = in_path.name.replace("tokens_", "passages_")
    final_path = OUT_DIR / shard_name

    # Never blindly skip a final file that cannot be read.
    if final_path.exists():
        try:
            pq.ParquetFile(final_path).metadata
            log(f"Already complete; skipping {shard_name}")
            return
        except Exception:
            log(f"Existing final file is unreadable; rebuilding {shard_name}")
            final_path.unlink()

    shard_checkpoints = CHECKPOINT_DIR / in_path.stem
    shard_checkpoints.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(in_path)
    total_docs = pf.metadata.num_rows

    log(f"Processing {in_path.name}: {total_docs} documents")
    doc_index = 0

    for batch in pf.iter_batches(batch_size=1):
        for doc in batch.to_pylist():
            # Keep doc_index zero-based to match checkpoint filenames.
            checkpoint = (
                shard_checkpoints / f"doc_{doc_index:08d}.parquet"
            )
            current_index = doc_index
            doc_index += 1

            if current_index < RESUME_FROM_DOC or current_index > STOP_AFTER_DOC:
                continue

            if checkpoint.exists() and valid_checkpoint(checkpoint):
                if doc_index % 25 == 0:
                    log(f"Resume: document {doc_index}/{total_docs}")
                continue

            if checkpoint.exists():
                checkpoint.unlink()

            if not doc["tokens"]:
                # Empty documents have no passages; no checkpoint needed.
                continue

            log(
                f"Embedding document {doc_index}/{total_docs}: "
                f"{doc['doc_id']}"
            )
            t0 = time.perf_counter()

            table = document_table(doc, tok, model, device)
            if table is None:
                continue

            local = LOCAL_TMP / checkpoint.name
            pq.write_table(table, local, compression="zstd")
            commit_file(local, checkpoint)
            local.unlink(missing_ok=True)

            log(
                f"Committed {doc['doc_id']}: {table.num_rows} passages "
                f"in {time.perf_counter() - t0:.1f}s"
            )

            del table

            if device == "cuda":
                # Releases cached blocks; does not change model results.
                torch.cuda.empty_cache()

    if doc_index != total_docs:
        raise RuntimeError(
            f"Document count mismatch: read {doc_index}, expected {total_docs}"
        )

    if not ASSEMBLE_FINAL_SHARD:
        print(
            f"CHECKPOINTS ONLY: {in_path.name}; "
            "skipping final-shard assembly."
        )
        return

    # Stream one checkpoint at a time into the final local Parquet file.
    # This avoids reading the whole shard into RAM.
    local_final = LOCAL_TMP / shard_name
    if local_final.exists():
        local_final.unlink()

    writer = None
    expected_rows = 0

    try:
        for checkpoint in sorted(shard_checkpoints.glob("doc_*.parquet")):
            if not valid_checkpoint(checkpoint):
                raise RuntimeError(f"Invalid checkpoint: {checkpoint}")

            table = pq.read_table(checkpoint)
            if writer is None:
                writer = pq.ParquetWriter(
                    local_final, table.schema, compression="zstd"
                )
            elif not table.schema.equals(writer.schema, check_metadata=False):
                raise RuntimeError(
                    f"Schema mismatch in checkpoint {checkpoint.name}"
                )

            writer.write_table(table)
            expected_rows += table.num_rows
            del table
    finally:
        if writer is not None:
            writer.close()

    if expected_rows == 0:
        raise RuntimeError(
            "No passage rows were produced; refusing to publish an empty shard"
        )

    # Validate the assembled local file before publishing it.
    assembled = pq.ParquetFile(local_final)
    if assembled.metadata.num_rows != expected_rows:
        raise RuntimeError("Final row-count validation failed")

    commit_file(local_final, final_path)
    local_final.unlink(missing_ok=True)

    log(
        f"COMPLETE {shard_name}: {expected_rows} passages "
        f"from {total_docs} documents"
    )


def main():
    drive.mount("/content/drive")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_TMP.mkdir(parents=True, exist_ok=True)

    shards = sorted(IN_DIR.glob("tokens_*.parquet"))
    log(f"Found {len(shards)} input shards in {IN_DIR}")

    if not shards:
        raise SystemExit("No tokens_*.parquet files found")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")

    if device == "cpu":
        log("WARNING: CPU inference will be slow")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    if device == "cuda":
        model = model.half()

    log("Model ready")

    for i, in_path in enumerate(shards, 1):
        log(f"Shard {i}/{len(shards)}")
        process_shard(in_path, tok, model, device)

    log("Done")


if __name__ == "__main__":
    main()

