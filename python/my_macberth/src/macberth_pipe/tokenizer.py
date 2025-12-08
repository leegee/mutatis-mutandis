# python/my_macberth/src/macberth_pipe/tokenizer.py
from transformers import AutoTokenizer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOCAL_MODEL_DIR = (BASE_DIR / "../../../lib/macberth-huggingface").resolve()

class MacBERThTokenizer:
    """Wrapper for MacBERTh tokenizer using local model files."""

    def __init__(self, model_name=LOCAL_MODEL_DIR):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True
        )

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def encode(self, text, **kwargs):
        return self.tokenizer.encode(text, **kwargs)

    def __call__(self, text, **kwargs):
        return self.tokenizer(text, **kwargs)
