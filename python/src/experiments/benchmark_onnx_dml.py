#!/usr/bin/env python
"""
benchmark_onnx_dml.py - Quick feasibility test: does DirectML actually
help on this AMD card, and do the embeddings stay numerically close
enough to trust?

Run once, standalone, before touching the real pipeline.
"""

import time
import numpy as np
import torch

from lib.macberth import load_macberth  # your existing loader

# ---- config ----
BATCH_SIZE = 16
SEQ_LEN = 256          # matches your "local" window size
N_BATCHES = 10          # ~160 windows, enough to get a stable timing signal
ONNX_EXPORT_DIR = "./macberth-onnx"


def make_dummy_batch(tokenizer, batch_size, seq_len):
    # Random-ish but real vocab ids, so the forward pass isn't trivially cheap
    vocab_size = tokenizer.vocab_size
    input_ids = np.random.randint(100, vocab_size - 100, size=(batch_size, seq_len), dtype=np.int64)
    attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
    return input_ids, attention_mask


def bench_pytorch(mac, input_ids_np, attention_mask_np, n_batches):
    model = mac.model
    model.eval()
    input_ids = torch.tensor(input_ids_np, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask_np, dtype=torch.long)

    # warmup
    with torch.inference_mode():
        model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(n_batches):
            out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    elapsed = time.perf_counter() - start

    return elapsed, out.last_hidden_state.numpy()


def export_to_onnx(mac, export_dir):
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    print(f"Exporting to ONNX at {export_dir} (one-time, may take a minute)...")
    ort_model = ORTModelForFeatureExtraction.from_pretrained(
        mac.model.config._name_or_path, export=True
    )
    ort_model.save_pretrained(export_dir)
    mac.tokenizer.save_pretrained(export_dir)
    print("Export done.")


def bench_onnx(export_dir, providers, input_ids_np, attention_mask_np, n_batches):
    import onnxruntime as ort

    available = ort.get_available_providers()
    usable = [p for p in providers if p in available]
    if not usable:
        print(f"  None of {providers} available. Available providers: {available}")
        return None, None

    session = ort.InferenceSession(f"{export_dir}/model.onnx", providers=usable)
    print(f"  Using providers: {session.get_providers()}")

    feed = {"input_ids": input_ids_np, "attention_mask": attention_mask_np}

    # warmup
    out = session.run(None, feed)

    start = time.perf_counter()
    for _ in range(n_batches):
        out = session.run(None, feed)
    elapsed = time.perf_counter() - start

    return elapsed, out[0]


def cosine_sim(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    print("Loading MacBERTh via existing pipeline loader...")
    mac = load_macberth()

    input_ids_np, attention_mask_np = make_dummy_batch(mac.tokenizer, BATCH_SIZE, SEQ_LEN)

    print(f"\n--- PyTorch CPU baseline ({N_BATCHES} batches x {BATCH_SIZE}x{SEQ_LEN}) ---")
    pt_time, pt_hidden = bench_pytorch(mac, input_ids_np, attention_mask_np, N_BATCHES)
    pt_per_batch = pt_time / N_BATCHES
    print(f"Total: {pt_time:.2f}s  |  Per batch: {pt_per_batch*1000:.1f}ms")

    export_to_onnx(mac, ONNX_EXPORT_DIR)

    print(f"\n--- ONNX Runtime, CPU only ---")
    cpu_time, cpu_hidden = bench_onnx(
        ONNX_EXPORT_DIR, ["CPUExecutionProvider"],
        input_ids_np, attention_mask_np, N_BATCHES
    )
    if cpu_time:
        print(f"Total: {cpu_time:.2f}s  |  Per batch: {cpu_time/N_BATCHES*1000:.1f}ms  "
              f"|  Speedup vs PyTorch: {pt_time/cpu_time:.2f}x")
        print(f"Cosine similarity vs PyTorch: {cosine_sim(pt_hidden, cpu_hidden):.6f}")

    print(f"\n--- ONNX Runtime, DirectML (AMD GPU) ---")
    dml_time, dml_hidden = bench_onnx(
        ONNX_EXPORT_DIR, ["DmlExecutionProvider", "CPUExecutionProvider"],
        input_ids_np, attention_mask_np, N_BATCHES
    )
    if dml_time:
        print(f"Total: {dml_time:.2f}s  |  Per batch: {dml_time/N_BATCHES*1000:.1f}ms  "
              f"|  Speedup vs PyTorch: {pt_time/dml_time:.2f}x")
        print(f"Cosine similarity vs PyTorch: {cosine_sim(pt_hidden, dml_hidden):.6f}")

    print("\n--- Summary ---")
    print(f"PyTorch CPU:     {pt_per_batch*1000:.1f}ms/batch")
    if cpu_time:
        print(f"ONNX CPU:        {cpu_time/N_BATCHES*1000:.1f}ms/batch  ({pt_time/cpu_time:.2f}x)")
    if dml_time:
        print(f"ONNX DirectML:   {dml_time/N_BATCHES*1000:.1f}ms/batch  ({pt_time/dml_time:.2f}x)")


if __name__ == "__main__":
    main()
