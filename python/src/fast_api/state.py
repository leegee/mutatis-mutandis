from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import fast_api.connections


@dataclass
class PipelineState:
    # FAISS is process-wide, read-only cache
    _index: Optional[object] = None

    # lookup is per-key cache (NOT global singleton)
    _lookup_cache: Dict[str, object] = field(default_factory=dict)

    # FAISS INDEX (global cache)
    @property
    def index(self):
        if self._index is None:
            self._index = fast_api.connections.get_index()
        return self._index

    # LOOKUP (per-job / per-key)
    def get_tier1_zarr_lookup(self, key: str = "default"):
        if key not in self._lookup_cache:
            self._lookup_cache[key] = fast_api.connections.get_tier1_zarr_lookup()
        return self._lookup_cache[key]

    # Explicit reset hooks (useful in debugging)
    def clear_lookup_cache(self):
        self._lookup_cache.clear()

    def reset_index(self):
        self._index = None


STATE = PipelineState()


def init_state():
    # touch properties to warm caches (NO assignment here)
    _ = STATE.index
    _ = STATE.get_tier1_zarr_lookup()
