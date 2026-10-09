"""
lib/macberth.py
"""

from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Union
import onnxruntime as ort
from types import SimpleNamespace
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from lib.corpus_logging import logger
from lib.corpus_config import MODELS_DIR

BATCH_SIZE = 64

ONNX_MODEL_DIR = MODELS_DIR / "./macberth-onnx-fp32"
ONNX_MODEL_DIR.mkdir(parents=True, exist_ok=True)

logger.debug(f"ONNX_MODEL_DIR {ONNX_MODEL_DIR}")

# This file lives at .../src/lib/macberth.py
# Model path is D:\src\mutatis-mutandis\python\lib
_THIS_DIR = Path(__file__).resolve().parent.parent.parent
MACBERTH_MODEL_PATH = _THIS_DIR / "lib" / "macberth-huggingface"
MACBERTH_MODEL_NAME = "emanjavacas/MacBERTh"

logger.debug(f"MACBERTH_MODEL_PATH {MACBERTH_MODEL_PATH}")

# Passage support
PASSAGE_CHUNK = 510
PASSAGE_OVERLAP = 96
PASSAGE_BATCH_SIZE = 32
PASSAGE_REPRESENTATIONS = ("l8", "last", "mean4")


@dataclass
class MacberthModel:
    tokenizer: AutoTokenizer
    model: AutoModelForMaskedLM
    device: str

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def encode(self, **kwargs):
        """
        Run the encoder only.

        Used by Tier 1:
            - normal tokens
            - masked tokens

        Returns hidden states.
        """
        return self.model.base_model(**kwargs)

    def predict_masked(self, **kwargs):
        """
        Run the full masked-language model.

        Used by Tier 1.4 substitute vectors.

        Returns logits over vocabulary.
        """
        return self.model(**kwargs)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_macberth() -> MacberthModel:
    """
    Loads MacBERTh with encoder + MLM head.

    The encoder is accessed through model.base_model.

    (The MLM head is currently used by Tier 1.4.)
    """

    logger.debug("Loading MacBERTh model...")
    logger.debug("MacBERTh model path: %s", MACBERTH_MODEL_PATH)

    if not MACBERTH_MODEL_PATH.is_dir():
        raise FileNotFoundError( f"MacBERTh model directory does not exist: {MACBERTH_MODEL_PATH}" )

    tokenizer = AutoTokenizer.from_pretrained(
        MACBERTH_MODEL_PATH,
        local_files_only=True,
    )

    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("Tokenizer must be fast")

    model = AutoModelForMaskedLM.from_pretrained(
        MACBERTH_MODEL_PATH,
        local_files_only=True,
    )

    device = get_device()

    model.to(device)
    model.eval()

    return MacberthModel(
        tokenizer=tokenizer,
        model=model,
        device=device,
    )

def normalize(v: np.ndarray) -> Optional[np.ndarray]:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return None
    return v / n


class MacBERThEmbedder:
    """
    BERTopic-compatible wrapper around MacberthModel.
    """

    def __init__(self, macberth: MacberthModel, pooling: str = "mean"):
        self.macberth = macberth
        self.pooling = pooling
        self.device = macberth.device


    @property
    def hidden_size(self) -> int:
        return self.macberth.hidden_size

    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        **kwargs,
    ) -> np.ndarray:

        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        with torch.no_grad():
            for start in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[start:start + BATCH_SIZE]

                encoded = self.macberth.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )

                encoded = {
                    k: v.to(self.device)
                    for k, v in encoded.items()
                }

                outputs = self.macberth.encode(**encoded)

                if self.pooling == "cls":
                    emb = outputs.last_hidden_state[:, 0, :]

                elif self.pooling == "max":
                    emb = outputs.last_hidden_state.max(dim=1).values

                else:
                    attention_mask = encoded["attention_mask"]
                    emb = self._mean_pooling(
                        outputs.last_hidden_state,
                        attention_mask,
                    )

                all_embeddings.extend(emb.cpu().numpy())

        embeddings = np.array(all_embeddings)

        if convert_to_numpy:
            return embeddings

        return torch.tensor(embeddings)



    def encode_normalized(
        self,
        texts: Union[str, List[str]],
    ) -> np.ndarray:

        embeddings = self.encode(texts)
        return np.array([
            normalize(v)
            for v in embeddings
        ])


    @staticmethod
    def _mean_pooling(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        input_mask_expanded = (
            attention_mask
            .unsqueeze(-1)
            .expand(hidden_states.size())
            .float()
        )

        sum_embeddings = torch.sum(
            hidden_states * input_mask_expanded,
            dim=1,
        )

        sum_mask = torch.clamp(
            input_mask_expanded.sum(dim=1),
            min=1e-9,
        )

        return sum_embeddings / sum_mask


class OnnxMacberthModel:
    """
    ONNX Runtime-backed stand-in for MacberthModel. Exposes the same
    .tokenizer / .device / .encode() surface so MacBERThEmbedder doesn't
    need to know the difference.
    """

    def __init__(self, tokenizer, session):
        self.tokenizer = tokenizer
        self.session = session
        self.device = "cpu"

        # Read once from ONNX metadata rather than requiring transformers config.
        # The exported encoder output shape is [batch, sequence, hidden_size].
        output_shape = session.get_outputs()[0].shape
        self._hidden_size = output_shape[-1]

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def encode(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        # ORT accepts NumPy arrays, so convert the tokenizer tensors once per
        # batch rather than sending them through a PyTorch model.
        input_ids_np = input_ids.cpu().numpy()
        attention_mask_np = attention_mask.cpu().numpy()

        if token_type_ids is None:
            token_type_ids_np = np.zeros_like(input_ids_np)
        else:
            token_type_ids_np = token_type_ids.cpu().numpy()

        feed = {
            "input_ids": input_ids_np,
            "attention_mask": attention_mask_np,
            "token_type_ids": token_type_ids_np,
        }

        outputs = self.session.run(None, feed)

        last_hidden_state = torch.from_numpy(outputs[0])

        return SimpleNamespace(
            last_hidden_state=last_hidden_state
        )


def _export_macberth_onnx(export_dir: Path) -> None:
    """
    Export the MacBERTh encoder to an unquantized FP32 ONNX model.

    The exported graph contains only the transformer encoder and returns
    last_hidden_state:

        [batch, sequence, hidden_size]

    The MLM head is deliberately not exported because the ONNX backend is
    currently used for embedding generation only.

    The tokenizer is copied into export_dir so that the ONNX model directory
    is self-contained.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    logger.info( "[macberth] Exporting MacBERTh encoder to ONNX: %s", export_dir, )

    tokenizer = AutoTokenizer.from_pretrained(
        MACBERTH_MODEL_PATH,
        local_files_only=True,
    )

    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("Tokenizer must be fast")

    model = AutoModelForMaskedLM.from_pretrained(
        MACBERTH_MODEL_PATH,
        local_files_only=True,
    )

    model.eval()
    model.cpu()

    class EncoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.encoder = model.base_model

        def forward(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
        ):
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            return outputs.last_hidden_state

    encoder = EncoderWrapper(model)
    encoder.eval()

    # Representative inputs used only to trace/export the graph.
    sample = tokenizer(
        "This is a sample sentence for MacBERTh ONNX export.",
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=512,
    )

    input_ids = sample["input_ids"]
    attention_mask = sample["attention_mask"]

    # Keep the ONNX interface stable even if the tokenizer/model does not
    # normally need token_type_ids.
    token_type_ids = sample.get(
        "token_type_ids",
        torch.zeros_like(input_ids),
    )

    onnx_path = export_dir / "model.onnx"

    with torch.no_grad():
        torch.onnx.export(
            encoder,
            (
                input_ids,
                attention_mask,
                token_type_ids,
            ),
            str(onnx_path),
            input_names=[
                "input_ids",
                "attention_mask",
                "token_type_ids",
            ],
            output_names=[
                "last_hidden_state",
            ],
            dynamic_axes={
                "input_ids": {
                    0: "batch",
                    1: "sequence",
                },
                "attention_mask": {
                    0: "batch",
                    1: "sequence",
                },
                "token_type_ids": {
                    0: "batch",
                    1: "sequence",
                },
                "last_hidden_state": {
                    0: "batch",
                    1: "sequence",
                },
            },
            opset_version=17,
            do_constant_folding=True,
        )

    # Make the ONNX directory self-contained.
    tokenizer.save_pretrained(export_dir)

    logger.info(
        "[macberth] ONNX export complete: %s",
        onnx_path,
    )


def load_macberth_onnx(
    export_dir: Optional[Path] = None,
    providers: Optional[List[str]] = None,
) -> OnnxMacberthModel:
    """
    Loads the ONNX MacBERTh model for inference. Exports it first if it
    doesn't already exist on disk.
    """
    import os

    if export_dir is None:
        export_dir = ONNX_MODEL_DIR
    export_dir = Path(export_dir)

    if not (export_dir / "model.onnx").exists():
        _export_macberth_onnx(export_dir)

    if providers is None:
        # Auto-select the best available provider
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif "DmlExecutionProvider" in available:
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

    usable = [p for p in providers if p in ort.get_available_providers()]
    if not usable:
        raise RuntimeError( f"None of the requested providers are available: {providers}. Available: {ort.get_available_providers()}" )

    provider_options = [
        {"device_id": 0} if p in ("CUDAExecutionProvider", "DmlExecutionProvider") else {}
        for p in usable
    ]

    sess_options = _configure_ort_session_options()

    session = ort.InferenceSession(
        str(export_dir / "model.onnx"),
        sess_options=sess_options,
        providers=usable,
        provider_options=provider_options,
    )

    actual = session.get_providers()
    logger.info(
        "[macberth.load_macberth_onnx] Loaded ONNX MacBERTh | "
        "requested=%s | actual=%s",
        providers,
        actual,
    )

    tokenizer = AutoTokenizer.from_pretrained(export_dir, local_files_only=True)

    return OnnxMacberthModel(tokenizer=tokenizer, session=session)


def _configure_ort_session_options() -> ort.SessionOptions:
    """
    Centralizes CPU thread/execution tuning for the ONNX session.

    Rationale (all CPU-bound-specific):
      - intra_op_num_threads is set explicitly rather than left to ORT's
        default, and is coordinated with cpu_count() so it doesn't
        silently pick a value that fights the rest of the pipeline
        (tokenization, window bookkeeping, parquet writes) for cores.
        ORT_NUM_THREADS env var still overrides if set.
      - inter_op_num_threads=1 because a single BERT encoder graph has
        essentially no independent branches to parallelize across --
        inter-op parallelism here is pure scheduling overhead.
      - ORT_SEQUENTIAL over the default ORT_PARALLEL for the same reason:
        parallel execution mode is built for graphs with concurrent
        branches, not a single linear encoder stack.
      - allow_spinning=0 so idle ORT worker threads yield the CPU
        instead of busy-waiting, which matters because this process
        interleaves non-trivial Python work between forward passes
        rather than running back-to-back inference in a tight loop.
    """
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    try:
        n = int(os.environ.get("ORT_NUM_THREADS", "0"))
    except ValueError:
        n = 0

    if n <= 0:
        n = os.cpu_count() or 4

    sess_options.intra_op_num_threads = n
    sess_options.inter_op_num_threads = 1

    sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")

    logger.info(
        "[macberth] ORT session config: intra_op_num_threads=%d, "
        "inter_op_num_threads=1, execution_mode=SEQUENTIAL, spinning=off",
        n,
    )

    return sess_options


def load_macberth_onnx(
    export_dir: Optional[Path] = None,
    providers: Optional[List[str]] = None,
) -> OnnxMacberthModel:
    """
    Loads the ONNX MacBERTh model for inference. Exports it first if it
    doesn't already exist on disk.

    Default provider is CPU-only (stable for long Tier 1 runs on Windows).
    Pass `providers=["DmlExecutionProvider", "CPUExecutionProvider"]`
    """
    if export_dir is None:
        export_dir = ONNX_MODEL_DIR
    export_dir = Path(export_dir)

    if not (export_dir / "model.onnx").exists():
        _export_macberth_onnx(export_dir)

    if providers is None:
        # Env override: MACBERTH_ONNX_PROVIDER=dml|cpu
        pref = os.environ.get("MACBERTH_ONNX_PROVIDER", "cpu").strip().lower()
        if pref in ("dml", "directml", "gpu"):
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

    usable = [p for p in providers if p in ort.get_available_providers()]
    if not usable:
        raise RuntimeError(f"None of the requested providers are available: {providers}")

    provider_options = [
        {"device_id": 0} if p == "DmlExecutionProvider" else {}
        for p in usable
    ]

    sess_options = _configure_ort_session_options()

    session = ort.InferenceSession(
        f"{export_dir}/model.onnx",
        sess_options=sess_options,
        providers=usable,
        provider_options=provider_options,
    )
    logger.info( "[macberth.load_macberth_onnx] Loaded ONNX MacBERTh, providers: %s", session.get_providers(), )

    tokenizer = AutoTokenizer.from_pretrained(export_dir, local_files_only=True)

    return OnnxMacberthModel(tokenizer=tokenizer, session=session)


def _best_backend(preferred: str | None = None) -> str:
    """
    Choose the fastest available backend.
    Priority:
      1. Explicit request
      2. PyTorch + CUDA
      3. ONNX + CUDA
      4. ONNX + DirectML
      5. ONNX CPU
      6. PyTorch CPU
    """
    if preferred in ("onnx", "pytorch"):
        return preferred

    # 1. PyTorch CUDA – most reliable on Colab
    if torch.cuda.is_available():
        return "pytorch"

    # 2. ONNX CUDA
    if "CUDAExecutionProvider" in ort.get_available_providers():
        return "onnx"

    # 3. ONNX DirectML (Windows)
    if "DmlExecutionProvider" in ort.get_available_providers():
        return "onnx"

    # 4. Default to ONNX CPU
    return "onnx"


def get_macberth_embedder(
    pooling: str = "mean",
    backend: str | None = None,          # None = auto
) -> MacBERThEmbedder:
    backend = _best_backend(backend)

    logger.info("[macberth] selected backend: %s", backend)

    if backend == "onnx":
        # Prefer CUDA → DirectML → CPU
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif "DmlExecutionProvider" in available:
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        macberth_model = load_macberth_onnx(providers=providers)

    elif backend == "pytorch":
        macberth_model = load_macberth()          # already moves to CUDA if available
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return MacBERThEmbedder(macberth_model, pooling=pooling)


def embed_query(
    text: str,
    *,
    backend: str = "onnx",
    pooling: str = "mean",
) -> np.ndarray:
    """
    Encode a short natural-language query for vector retrieval.

    Returns a single L2-normalized vector with shape (1, hidden_size),
    suitable for FAISS inner-product search.

    Query vectors must follow the same normalization convention as the
    stored Tier 1 vectors; otherwise inner-product scores are not comparable.
    """

    embedder = get_macberth_embedder(
        pooling=pooling,
        backend=backend
    )

    embedding = embedder.encode( text )[0]
    vector = normalize(embedding)

    if vector is None:
        raise RuntimeError( "MacBERTh produced zero-length query embedding" )

    return vector.astype(np.float32)[None, :]

def _passage_layers(hidden_states):
    """Match the layer selection used by colab_embed_passages.py."""
    return {
        "l8": hidden_states[8],
        "last": hidden_states[-1],
        "mean4": torch.stack(hidden_states[-4:]).mean(0),
    }


@torch.inference_mode()
def encode_passage_words(
    words: list[str],
    macberth: MacberthModel,
) -> dict[str, np.ndarray]:
    """
    Produce contextualised word vectors using the same subword,
    chunk-overlap and layer-pooling procedure as the Colab job.

    Returns one (n_words, hidden_size) float32 array per representation.
    """
    if not words:
        raise ValueError("Cannot encode an empty passage.")

    words = [word if word.strip() else "." for word in words]

    tokenizer = macberth.tokenizer
    device = macberth.device
    dim = macberth.hidden_size

    encoded = tokenizer(
        words,
        is_split_into_words=True,
        add_special_tokens=False,
        truncation=False,
    )

    input_ids = encoded["input_ids"]
    word_ids = np.asarray(encoded.word_ids())

    if len(input_ids) == 0:
        raise ValueError("Tokenizer produced no subword tokens.")

    chunks = []
    start = 0

    while True:
        end = min(start + PASSAGE_CHUNK, len(input_ids))
        chunks.append((start, end))

        if end == len(input_ids):
            break

        start = end - PASSAGE_OVERLAP

    sums = {
        name: np.zeros((len(words), dim), dtype=np.float32)
        for name in PASSAGE_REPRESENTATIONS
    }
    counts = np.zeros(len(words), dtype=np.float32)

    # Preserve the chunk ordering used in the Colab implementation.
    order = sorted(
        range(len(chunks)),
        key=lambda i: chunks[i][1] - chunks[i][0],
    )

    for offset in range(0, len(order), PASSAGE_BATCH_SIZE):
        selected = [
            chunks[i]
            for i in order[offset:offset + PASSAGE_BATCH_SIZE]
        ]
        maxlen = max(end - start for start, end in selected) + 2

        input_batch = torch.full(
            (len(selected), maxlen),
            tokenizer.pad_token_id,
            dtype=torch.long,
        )
        attention_batch = torch.zeros(
            (len(selected), maxlen),
            dtype=torch.long,
        )

        for row, (start, end) in enumerate(selected):
            sequence = (
                [tokenizer.cls_token_id]
                + input_ids[start:end]
                + [tokenizer.sep_token_id]
            )
            input_batch[row, :len(sequence)] = torch.tensor(sequence)
            attention_batch[row, :len(sequence)] = 1

        output = macberth.encode(
            input_ids=input_batch.to(device),
            attention_mask=attention_batch.to(device),
            output_hidden_states=True,
        )

        layers = {
            name: tensor.float().cpu().numpy()
            for name, tensor in _passage_layers(
                output.hidden_states
            ).items()
        }

        for row, (start, end) in enumerate(selected):
            chunk_word_ids = word_ids[start:end]

            first = np.flatnonzero(
                np.r_[True, chunk_word_ids[1:] != chunk_word_ids[:-1]]
            )
            unique_words = chunk_word_ids[first]
            sizes = np.diff(
                np.r_[first, len(chunk_word_ids)]
            ).astype(np.float32)

            counts[unique_words] += sizes

            for name in PASSAGE_REPRESENTATIONS:
                hidden = layers[name][row, 1:1 + (end - start)]
                sums[name][unique_words] += np.add.reduceat(
                    hidden, first, axis=0
                )

    counts = np.maximum(counts, 1.0)[:, None]

    return {
        name: sums[name] / counts
        for name in PASSAGE_REPRESENTATIONS
    }


def encode_passage_query(
    text: str,
    representation: str,
    macberth: MacberthModel,
) -> np.ndarray:
    """
    Encode a text query using the same word-vector aggregation as
    the indexed passage vectors.

    Returns an unnormalised float32 vector of shape (hidden_size,).
    """
    if representation not in PASSAGE_REPRESENTATIONS:
        raise ValueError(
            f"Unknown representation {representation!r}; "
            f"choose from {PASSAGE_REPRESENTATIONS}"
        )

    words = text.split()
    if not words:
        raise ValueError("Query must contain at least one word.")

    word_vectors = encode_passage_words(words, macberth)
    vector = word_vectors[representation].mean(axis=0)
    vector = np.asarray(vector, dtype=np.float32)

    if not np.isfinite(vector).all():
        raise ValueError("Query embedding contains non-finite values.")

    if np.linalg.norm(vector) < 1e-12:
        raise ValueError("Query embedding is a zero vector.")

    return vector
