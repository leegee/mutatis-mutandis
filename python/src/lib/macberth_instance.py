# lib/macberth_instance.py

from functools import lru_cache
from lib.macberth import get_macberth_embedder


@lru_cache(maxsize=1)
def get_shared_embedder():
    return get_macberth_embedder()
