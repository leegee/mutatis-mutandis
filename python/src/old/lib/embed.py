import numpy as np
import torch


def embed_window(tokens, tokenizer, model, device, max_length: int):
    """
    Stateless embedding primitive.

    Invariant:
        - No aggregation
        - No IDs
        - No side effects
        - Pure mapping: tokens -> hidden states
    """

    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

    return outputs.last_hidden_state[0].detach().cpu().numpy()
