"""
lib/macberth.py
"""

from __future__ import annotations
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

MACBERTH_MODEL_PATH = Path("./lib/macberth-huggingface")
MACBERTH_MODEL_NAME = "emanjavacas/MacBERTh"

ONNX_MODEL_DIR = MODELS_DIR / "./macberth-onnx-fp32"
ONNX_MODEL_DIR.mkdir(parents=True, exist_ok=True)

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


def load_macberth(
    *,
    use_qint8: bool = True,
) -> MacberthModel:
    """
    Loads MacBERTh with encoder + MLM head.

    The encoder is accessed through model.base_model.

    (The MLM head is currently used by Tier 1.4.)
    """

    logger.info("Loading MacBERTh model...")

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

    if use_qint8:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        logger.info("Set model to use qint8")

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
            for text in texts:

                encoded = self.macberth.tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )

                encoded = {
                    k: v.to(self.device)
                    for k, v in encoded.items()
                }

                # Encoder only, not MLM head
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

                all_embeddings.append(
                    emb.cpu().numpy().squeeze()
                )

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
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        feed = {
            "input_ids": input_ids.cpu().numpy(),
            "attention_mask": attention_mask.cpu().numpy(),
            "token_type_ids": token_type_ids.cpu().numpy(),
        }

        outputs = self.session.run(None, feed)

        last_hidden_state = torch.from_numpy(outputs[0])

        return SimpleNamespace(
            last_hidden_state=last_hidden_state
        )


def _export_macberth_onnx(export_dir: Path = ONNX_MODEL_DIR) -> None:
    """
    Exports a fresh fp32 (unquantized) MacBERTh to ONNX format.

    Loads its own fp32 model instance rather than dequantizing an
    existing one -- dynamic quantization (use_qint8=True) repacks
    Linear weights into a non-tensor format with no clean reverse op.
    """
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    import shutil

    logger.info(f"ONNX export not found at {export_dir}, exporting now...")

    if export_dir.exists():
        shutil.rmtree(export_dir)

    mac_fp32 = load_macberth(use_qint8=False)
    mac_fp32.model.eval()

    temp_dir = Path(str(export_dir) + "_temp")
    if temp_dir.exists():
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

    logger.info(f"ONNX export completed at {export_dir}")


def load_macberth_onnx(
    export_dir: Path = ONNX_MODEL_DIR,
    providers: Optional[List[str]] = None,
) -> OnnxMacberthModel:
    """
    Loads the fp32 ONNX MacBERTh model for inference. Exports it first
    if it doesn't already exist on disk.

    Default provider is CPU-only (stable for long Tier 1 runs on Windows).
    Pass `providers=["DmlExecutionProvider", "CPUExecutionProvider"]` or set
    `MACBERTH_ONNX_PROVIDER=dml` to use DirectML.

    """
    import os
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

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Respect ORT_NUM_THREADS if set; helps long CPU runs stay predictable
    try:
        n = int(os.environ.get("ORT_NUM_THREADS", "0"))
        if n > 0:
            sess_options.intra_op_num_threads = n
            sess_options.inter_op_num_threads = 1
    except ValueError:
        pass

    session = ort.InferenceSession(
        f"{export_dir}/model.onnx",
        sess_options=sess_options,
        providers=usable,
        provider_options=provider_options,
    )
    logger.info("Loaded ONNX MacBERTh, providers: %s", session.get_providers())

    tokenizer = AutoTokenizer.from_pretrained(export_dir, local_files_only=True)

    return OnnxMacberthModel(tokenizer=tokenizer, session=session)


def get_macberth_embedder(
    pooling: str = "mean",
    backend: str = "onnx",  # "onnx" or "pytorch"
) -> MacBERThEmbedder:

    if backend == "onnx":
        macberth_model = load_macberth_onnx()
    else:
        macberth_model = load_macberth()

    return MacBERThEmbedder(
        macberth_model,
        pooling=pooling,
    )


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
