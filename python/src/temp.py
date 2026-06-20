from lib.eebo_config import ZARR_ROOT
from tier2_0_concept_events import ZarrEventLookup

lookup = ZarrEventLookup(ZARR_ROOT / "tier1")
ids = list(lookup.iter_matching_event_ids({"TEST"}))
print(f"found {len(ids)} events for TEST")