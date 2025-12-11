# This is just a placeholder.
import re

def normalize(text: str, lowercase=True, strip_punct=False) -> str:
    t = text
    if lowercase:
        t = t.lower()
    if strip_punct:
        t = re.sub(r"[^\w\s]", " ", t)
    return t
