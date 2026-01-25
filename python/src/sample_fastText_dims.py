import fasttext
import lib.eebo_config as config
from lib.eebo_logging import logger

slice_dir = config.FASTTEXT_SLICE_MODEL_DIR

for model_file in sorted(slice_dir.glob("slice_*.bin")):
    model = fasttext.load_model(str(model_file))
    dim = model.get_dimension()
    logger.info(f"{model_file.name}: dimension = {dim}")
