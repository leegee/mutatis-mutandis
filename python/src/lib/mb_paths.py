from lib.eebo_config import (
    MACBERTH_VECTORS_DIR,
    MACBERTH_SLICE_MODEL_DIR,
)

def slice_model_path(slice_range: tuple[int,int]) -> Path:
    start, end = slice_range
    path = MACBERTH_SLICE_MODEL_DIR / f"slice_{start}_{end}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def vectors_path(slice_id: str) -> Path:
    return MACBERTH_VECTORS_DIR / f"{slice_id}.npz"


def faiss_slice_path(slice_range: tuple[int,int]) -> Path:
    start, end = slice_range
    path = MACBERTH_VECTORS_DIR / f"slice_{start}_{end}.faiss"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
