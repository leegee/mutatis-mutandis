from pprint import pprint
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# -----------------------------
# 1. Load MacBERTh normally
# -----------------------------
model_name = "emanjavacas/MacBERTh"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

nlp = pipeline("token-classification", model=model, tokenizer=tokenizer, device=-1)

text = "Thys is a pamphlet sentence."

# -----------------------------
# 2. DICTIONARY of known EME→ME
# -----------------------------
EME_DICTIONARY = {
    "thys": "this",
    "vnto": "unto",
    "vpon": "upon",
    "bee": "be",
}

# -----------------------------
# 3. SIMPLE RULE ENGINE
# -----------------------------
def apply_rules(token: str):
    rules_applied = []

    result = token

    # v -> u (common)
    if result.startswith("v"):
        rules_applied.append("v->u (initial)")
        result = "u" + result[1:]

    # long ſ → s
    if "ſ" in result:
        rules_applied.append("long-s")
        result = result.replace("ſ", "s")

    return result, rules_applied


# -----------------------------
# 4. AUDIT OBSERVER
# -----------------------------
def normalise_token(token: str):
    audit_events = []
    lower = token.lower()

    # A. DICTIONARY LOOKUP
    if lower in EME_DICTIONARY:
        modern = EME_DICTIONARY[lower]
        audit_events.append({
            "event": "dictionary_match",
            "token": token,
            "modern": modern,
            "source": "dictionary"
        })
        return modern, audit_events

    # B. RULE ENGINE
    ruled, rules = apply_rules(lower)
    if rules:
        audit_events.append({
            "event": "rule_applied",
            "token": token,
            "result": ruled,
            "rules": rules,
            "source": "rules"
        })
        return ruled, audit_events

    # C. FALLBACK (no change)
    audit_events.append({
        "event": "no_change",
        "token": token,
        "result": token,
        "source": "identity"
    })
    return token, audit_events


# -----------------------------
# 5. RUN MACBERTh BUT ADD OUR LAYER
# -----------------------------
raw_result = nlp(text)

tokens = tokenizer.tokenize(text)

combined = []

for i, tok in enumerate(tokens):
    modern, events = normalise_token(tok)

    combined.append({
        "original": tok,
        "normalized": modern,
        "events": events,   # full provenance
    })

# -----------------------------
# 6. Output
# -----------------------------
pprint(combined)
