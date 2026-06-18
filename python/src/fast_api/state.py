from dataclasses import dataclass

from lib.eebo_faiss import EeboFaissIndex
from lib.eebo_config import (
    FAISS_TIER1_INDEX,
    ZARR_ROOT,
)

from tier2_0_concept_events import (
    ZarrEventLookup,
)


@dataclass
class PipelineState:
    index = None
    lookup = None


STATE = PipelineState()


def init_state():
    if STATE.index is None:
        STATE.index = EeboFaissIndex.load(
            FAISS_TIER1_INDEX
        )

    if STATE.lookup is None:
        STATE.lookup = ZarrEventLookup(
            ZARR_ROOT / "tier1"
        )
