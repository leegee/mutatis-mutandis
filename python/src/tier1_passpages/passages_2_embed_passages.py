"""
colab_embed_passages.py

Colab job: token parquet shards (Drive) -> passage-vector parquet shards (Drive).

No Postgres, no Lance. Each output shard is written to local /content first,
copied to Drive under a .part name, then renamed, so a disconnect can never
leave something that looks finished but isn't. Re-running skips finished shards.

Colab cells:
    from google.colab import drive; drive.mount("/content/drive")
    !pip install -q transformers pyarrow
    %run embed_passages_colab.py
"""
import os
import shutil
import time
from pathlib import Path
from google.colab import drive

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModel, AutoTokenizer

IN_DIR = Path("/content/drive/MyDrive/tier1/tokens")
OUT_DIR = Path("/content/drive/MyDrive/tier1/passages")
LOCAL_TMP = Path("/content/tmp")

SEG = 20000          # Max docs in RAM
MODEL_NAME = "emanjavacas/MacBERTh"
CHUNK = 510          # subwords per forward pass, excluding [CLS]/[SEP]
OVERLAP = 96         # subwords shared between consecutive chunks
BATCH = 32
WINDOW = 40          # words per passage
HOP = 20             # words between passage starts
DIM = 768

# Three representations per passage; choose after testing on labelled positives.
LAYER_NAMES = ("l8", "last", "mean4")


drive.mount("/content/drive")

print(os.path.exists("/content/drive/MyDrive"))
print(os.listdir("/content/drive/MyDrive")[:30])



def doc_passage_vectors(words, tok, model, device):
    spans = passage_spans(len(words))
    out = {n: [] for n in LAYER_NAMES}
    i = 0
    while i < len(spans):
        seg_start = spans[i][0]
        j = i
        while j < len(spans) and spans[j][1] - seg_start <= SEG:
            j += 1
        j = max(j, i + 1)
        seg_end = spans[j - 1][1]

        wv = word_vectors(words[seg_start:seg_end], tok, model, device)
        for n in LAYER_NAMES:
            out[n].append(
                np.stack([wv[n][s - seg_start:e - seg_start].mean(0)
                          for s, e in spans[i:j]]).astype(np.float16)
            )
        del wv
        i = j
    return spans, {n: np.concatenate(v) for n, v in out.items()}


def pick_layers(hidden_states):
    return {
        "l8": hidden_states[8],
        "last": hidden_states[-1],
        "mean4": torch.stack(hidden_states[-4:]).mean(0),
    }


@torch.inference_mode()
def word_vectors(words, tok, model, device):
    """One vector per word per layer: mean of its subwords, averaged over chunk overlaps."""
    words = [w if w.strip() else "." for w in words]  # empty tokens would vanish

    enc = tok(words, is_split_into_words=True, add_special_tokens=False,
              truncation=False)
    ids = enc["input_ids"]
    wids = np.array(enc.word_ids())
    n_words = len(words)

    chunks, start = [], 0
    while True:
        end = min(start + CHUNK, len(ids))
        chunks.append((start, end))
        if end == len(ids):
            break
        start = end - OVERLAP

    sums = {n: np.zeros((n_words, DIM), np.float32) for n in LAYER_NAMES}
    cnt = np.zeros(n_words, np.float32)

    order = sorted(range(len(chunks)), key=lambda i: chunks[i][1] - chunks[i][0])

    for b in range(0, len(order), BATCH):
        batch = [chunks[i] for i in order[b:b + BATCH]]
        maxlen = max(e - s for s, e in batch) + 2
        inp = torch.full((len(batch), maxlen), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(batch), maxlen), dtype=torch.long)

        for k, (s, e) in enumerate(batch):
            seq = [tok.cls_token_id] + ids[s:e] + [tok.sep_token_id]
            inp[k, :len(seq)] = torch.tensor(seq)
            att[k, :len(seq)] = 1

        out = model(input_ids=inp.to(device), attention_mask=att.to(device),
                    output_hidden_states=True)
        layers = {n: t.float().cpu().numpy()
                  for n, t in pick_layers(out.hidden_states).items()}

        for k, (s, e) in enumerate(batch):
            w = wids[s:e]
            # word ids are non-decreasing, so group subwords with reduceat
            first = np.flatnonzero(np.r_[True, w[1:] != w[:-1]])
            uniq = w[first]
            sizes = np.diff(np.r_[first, len(w)]).astype(np.float32)
            cnt[uniq] += sizes
            for n in LAYER_NAMES:
                h = layers[n][k, 1:1 + (e - s)]
                sums[n][uniq] += np.add.reduceat(h, first, axis=0)

    cnt = np.maximum(cnt, 1.0)[:, None]
    return {n: sums[n] / cnt for n in LAYER_NAMES}


def passage_spans(n_words):
    if n_words <= WINDOW:
        return [(0, n_words)]
    starts = list(range(0, n_words - WINDOW + 1, HOP))
    if starts[-1] + WINDOW < n_words:
        starts.append(n_words - WINDOW)  # cover the tail
    return [(s, s + WINDOW) for s in starts]


def fixed_f16(arr):
    arr = np.ascontiguousarray(arr.astype(np.float16))
    return pa.FixedSizeListArray.from_arrays(pa.array(arr.ravel()), DIM)


def process_shard(in_path, tok, model, device):
    cols = {k: [] for k in ("corpus", "doc_id", "pub_year", "word_start",
                            "word_end", "token_start", "token_end")}
    vecs = {n: [] for n in LAYER_NAMES}
    d = 0

    pf = pq.ParquetFile(in_path)
    for batch in pf.iter_batches(batch_size=16):
        for doc in batch.to_pylist():
            d += 1
            if d % 25 == 0:
                print(f"  doc {d}/{pf.metadata.num_rows}", flush=True)
            words, tidx = doc["tokens"], doc["token_idx"]
            if not words:
                continue

            spans, dv = doc_passage_vectors(words, tok, model, device)
            for n in LAYER_NAMES:
                vecs[n].append(dv[n])

            for s, e in spans:
                cols["corpus"].append(doc["corpus"])
                cols["doc_id"].append(doc["doc_id"])
                cols["pub_year"].append(doc["pub_year"])
                cols["word_start"].append(s)
                cols["word_end"].append(e)
                cols["token_start"].append(tidx[s])
                cols["token_end"].append(tidx[e - 1] + 1)

    if not vecs["l8"]:
        raise ValueError(
            f"No passage vectors produced from {in_path}"
        )

    arrays = {
        "corpus": pa.array(cols["corpus"]),
        "doc_id": pa.array(cols["doc_id"]),
        "pub_year": pa.array(cols["pub_year"], pa.int32()),
        "word_start": pa.array(cols["word_start"], pa.int32()),
        "word_end": pa.array(cols["word_end"], pa.int32()),
        "token_start": pa.array(cols["token_start"], pa.int32()),
        "token_end": pa.array(cols["token_end"], pa.int32()),
    }
    for n in LAYER_NAMES:
        arrays[f"vec_{n}"] = fixed_f16(np.concatenate(vecs[n]))

    return pa.table(arrays)


def main():
    if not IN_DIR.exists():
        raise SystemExit(f"IN_DIR not found: {IN_DIR} (is Drive mounted?)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_TMP.mkdir(parents=True, exist_ok=True)

    shards = sorted(IN_DIR.glob("tokens_*.parquet"))
    print(f"found {len(shards)} input shards in {IN_DIR}")
    if not shards:
        print("contents:", [p.name for p in IN_DIR.iterdir()][:10])
        raise SystemExit("no tokens_*.parquet files found")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU. Expect this to be very slow.")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    if device == "cuda":
        model = model.half()

    for i, in_path in enumerate(shards, 1):
        name = in_path.name.replace("tokens_", "passages_")
        final = OUT_DIR / name
        if final.exists():
            print(f"[{i}/{len(shards)}] skip {name}")
            continue

        t0 = time.perf_counter()
        table = process_shard(in_path, tok, model, device)

        local = LOCAL_TMP / name
        pq.write_table(table, local, compression="zstd")

        part = OUT_DIR / (name + ".part")
        shutil.copyfile(local, part)
        os.replace(part, final)  # only now does the shard "exist"
        local.unlink()

        print(f"[{i}/{len(shards)}] {name} passages={table.num_rows} "
              f"elapsed={time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
