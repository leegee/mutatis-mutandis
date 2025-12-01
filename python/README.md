# MacBERTh on CPU

## Install

    # First time:
    echo "python$ bash ./login" >> "$HOME/.bsahrc"

    # Log in again or run:
    python$ source "$HOME/.bashrc"

## Process

Looking at Lancaster VARD2 as so far the sole OS EME-to-ME system.

But could have had a Python dictionary with rules such as:

```python
    def apply_rules(token):
    if token.startswith("v"):
        token = "u" + token[1:]
    if "ſ" in token:
        token = token.replace("ſ", "s")
    return token
```

Its output after passing human review could create a corpus of EME-to-ME pairs 
to feed a Neural Normaliser....
