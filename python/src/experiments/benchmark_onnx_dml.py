#!/usr/bin/env python
"""
benchmark_onnx_dml.py - fp32 ONNX export benchmark (CPU vs DirectML)

Compares PyTorch fp32 baseline against ONNX Runtime CPU and DirectML
execution providers, both for speed and for output correctness
(cosine similarity of last_hidden_state).

Runs two correctness checks:
  1. An unpadded batch (all attention_mask == 1)
  2. A padded batch (variable-length sequences, real padding) to make
     sure attention masking traces correctly under ONNX/DML.
"""

import time
import numpy as np
import torch
import os
import shutil

from lib.macberth import load_macberth

BATCH_SIZE = 8
SEQ_LEN = 256
N_BATCHES = 5
ONNX_EXPORT_DIR = "./macberth-onnx-fp32"


def make_dummy_batch(tokenizer, batch_size, seq_len):
    """Unpadded batch: every position is valid (attention_mask all ones)."""
    vocab_size = tokenizer.vocab_size
    input_ids = np.random.randint(100, vocab_size - 100, size=(batch_size, seq_len), dtype=np.int64)
    attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
    token_type_ids = np.zeros((batch_size, seq_len), dtype=np.int64)
    return input_ids, attention_mask, token_type_ids


def make_padded_batch(tokenizer, batch_size, seq_len, pad_token_id=0, min_len_frac=0.3):
    """
    Padded batch: each row has a random valid length between
    min_len_frac*seq_len and seq_len, with the remainder padded.
    This exercises real attention masking, unlike make_dummy_batch.
    """
    vocab_size = tokenizer.vocab_size
    min_len = max(1, int(seq_len * min_len_frac))

    input_ids = np.full((batch_size, seq_len), pad_token_id, dtype=np.int64)
    attention_mask = np.zeros((batch_size, seq_len), dtype=np.int64)
    token_type_ids = np.zeros((batch_size, seq_len), dtype=np.int64)

    rng = np.random.default_rng()
    for i in range(batch_size):
        valid_len = rng.integers(min_len, seq_len + 1)
        input_ids[i, :valid_len] = rng.integers(100, vocab_size - 100, size=valid_len)
        attention_mask[i, :valid_len] = 1

    return input_ids, attention_mask, token_type_ids


def bench_pytorch(mac, input_ids_np, attention_mask_np, n_batches):
    mac.model.eval()
    input_ids = torch.tensor(input_ids_np, dtype=torch.long).to(mac.model.device)
    attention_mask = torch.tensor(attention_mask_np, dtype=torch.long).to(mac.model.device)

    with torch.inference_mode():
        out = mac.encode(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(n_batches):
            out = mac.encode(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    elapsed = time.perf_counter() - start
    return elapsed, out.last_hidden_state.cpu().numpy()


def export_to_onnx(export_dir):
    print(f"Exporting FP32 (unquantized) version to {export_dir}...")

    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)

    mac_fp32 = load_macberth(use_qint8=False)
    mac_fp32.model.eval()

    os.makedirs(export_dir, exist_ok=True)

    # MacBERTh is BertForMaskedLM. The embedding pipeline uses the BERT
    # encoder output, not the 30,000-way masked-language-model logits.
    model = mac_fp32.model.bert
    model.eval()
    model.to("cpu")

    input_ids_np, attention_mask_np, token_type_ids_np = make_dummy_batch(
        mac_fp32.tokenizer, BATCH_SIZE, SEQ_LEN
    )

    input_ids = torch.from_numpy(input_ids_np)
    attention_mask = torch.from_numpy(attention_mask_np)
    token_type_ids = torch.from_numpy(token_type_ids_np)

    output_path = os.path.join(export_dir, "model.onnx")

    with torch.inference_mode():
        torch.onnx.export(
            model,
            args=(input_ids, attention_mask, token_type_ids),
            f=output_path,
            input_names=[
                "input_ids",
                "attention_mask",
                "token_type_ids",
            ],
            output_names=[
                "last_hidden_state",
            ],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "token_type_ids": {0: "batch", 1: "sequence"},
                "last_hidden_state": {0: "batch", 1: "sequence"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

    mac_fp32.tokenizer.save_pretrained(export_dir)

    print("FP32 ONNX export completed.")
    print(f"Model: {output_path}")

    return mac_fp32


def old_export_to_onnx(export_dir):
    from optimum.onnxruntime import ORTModelForFeatureExtraction

    print(f"Exporting FP32 (unquantized) version to {export_dir}...")

    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)

    # Fresh, never-quantized model straight from disk. Dynamic quantization
    # (torch.quantization.quantize_dynamic) repacks Linear weights into a
    # non-tensor packed format with no clean reverse op, so we don't try to
    # dequantize a quantized model -- we just load fp32 directly.
    mac_fp32 = load_macberth(use_qint8=False)
    mac_fp32.model.eval()

    temp_dir = export_dir + "_temp"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    mac_fp32.model.save_pretrained(temp_dir)

    ort_model = ORTModelForFeatureExtraction.from_pretrained(
        temp_dir,
        export=True,
        provider="CPUExecutionProvider",
    )
    ort_model.save_pretrained(export_dir)
    mac_fp32.tokenizer.save_pretrained(export_dir)

    shutil.rmtree(temp_dir, ignore_errors=True)

    print("FP32 ONNX export completed.")
    return mac_fp32


def bench_onnx(export_dir, providers, input_ids_np, attention_mask_np, token_type_ids_np, n_batches):
    import onnxruntime as ort

    usable = [p for p in providers if p in ort.get_available_providers()]
    if not usable:
        return None, None

    provider_options = [{"device_id": 0} if p == "DmlExecutionProvider" else {} for p in usable]

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        f"{export_dir}/model.onnx",
        sess_options=sess_options,
        providers=usable,
        provider_options=provider_options,
    )
    print(f"  Using providers: {session.get_providers()}")

    feed = {
        "input_ids": input_ids_np,
        "attention_mask": attention_mask_np,
        "token_type_ids": token_type_ids_np,
    }

    _ = session.run(None, feed)  # warmup

    start = time.perf_counter()
    for _ in range(n_batches):
        out = session.run(None, feed)
    elapsed = time.perf_counter() - start

    return elapsed, out[0]


def cosine_sim(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def run_comparison(label, mac_fp32, input_ids_np, attention_mask_np, token_type_ids_np, n_batches):
    """Run PyTorch fp32 vs ONNX CPU vs ONNX DirectML for one batch, print results."""
    print(f"\n=== {label} ===")

    print(f"\n--- PyTorch fp32 ---")
    pt_time, pt_hidden = bench_pytorch(mac_fp32, input_ids_np, attention_mask_np, n_batches)
    print(f"Total: {pt_time:.2f}s  |  Per batch: {pt_time/n_batches*1000:.1f}ms")

    print(f"\n--- ONNX CPU (FP32) ---")
    cpu_time, cpu_hidden = bench_onnx(
        ONNX_EXPORT_DIR, ["CPUExecutionProvider"],
        input_ids_np, attention_mask_np, token_type_ids_np, n_batches
    )
    if cpu_time:
        print(f"Total: {cpu_time:.2f}s  |  Per batch: {cpu_time/n_batches*1000:.1f}ms")
        print(f"Cosine sim (ONNX CPU vs PyTorch fp32): {cosine_sim(pt_hidden, cpu_hidden):.6f}")

    print(f"\n--- ONNX DirectML (FP32) ---")
    dml_time, dml_hidden = bench_onnx(
        ONNX_EXPORT_DIR, ["DmlExecutionProvider", "CPUExecutionProvider"],
        input_ids_np, attention_mask_np, token_type_ids_np, n_batches
    )
    if dml_time:
        print(f"Total: {dml_time:.2f}s  |  Per batch: {dml_time/n_batches*1000:.1f}ms")
        print(f"Cosine sim (ONNX DML vs PyTorch fp32): {cosine_sim(pt_hidden, dml_hidden):.6f}")


def main():
    print("Loading MacBERTh (fp32)...")
    mac_fp32 = load_macberth(use_qint8=False)

    export_to_onnx(ONNX_EXPORT_DIR)

    # 1. Unpadded batch -- baseline sanity check
    input_ids_np, attention_mask_np, token_type_ids_np = make_dummy_batch(
        mac_fp32.tokenizer, BATCH_SIZE, SEQ_LEN
    )
    run_comparison(
        "Unpadded batch (attention_mask all ones)",
        mac_fp32, input_ids_np, attention_mask_np, token_type_ids_np, N_BATCHES,
    )

    # 2. Padded batch -- exercises real attention masking under ONNX/DML tracing
    pad_input_ids_np, pad_attention_mask_np, pad_token_type_ids_np = make_padded_batch(
        mac_fp32.tokenizer, BATCH_SIZE, SEQ_LEN
    )
    run_comparison(
        "Padded batch (variable-length sequences)",
        mac_fp32, pad_input_ids_np, pad_attention_mask_np, pad_token_type_ids_np, N_BATCHES,
    )


if __name__ == "__main__":
    main()
