from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class ClusterResult:
    """
    Canonical return type for ALL clustering pipelines.

    This is the ONLY structure compare.py is allowed to consume.
    """
    event_ids: List[int]

    # event_id -> cluster_id
    membership: Dict[int, int]

    # cluster_id -> list[event_id]
    communities: Dict[int, List[int]]

    # optional graph structure (graph pipeline only)
    graph: Any | None = None

    # raw labels aligned with event_ids
    labels: List[int] | None = None
