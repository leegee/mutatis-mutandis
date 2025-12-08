# python/my_macberth/src/macberth_pipe/macberth_model.py

import torch
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from .model_loader import get_local_macberth_path

class MacBERThModel:
    def __init__(self, model_path=None, device="cpu"):
        self.device = device

        # Use shared loader if not provided
        if model_path is None:
            model_path = get_local_macberth_path()
        else:
            model_path = Path(model_path).resolve()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True
        )
        self.model.to(device)
        self.model.eval()

    def embed_text(self, text, chunk_size=512):
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        ids = enc["input_ids"][0]

        chunks = [ids[i:i+chunk_size] for i in range(0, len(ids), chunk_size)]
        vecs = []

        for c in chunks:
            with torch.no_grad():
                out = self.model(input_ids=c.unsqueeze(0).to(self.device))
            vecs.append(out.last_hidden_state.mean(dim=1).cpu().numpy()[0])

        return vecs
