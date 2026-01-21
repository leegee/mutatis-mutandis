# eebo_logging.py
import sys
import os
import logging

import lib.eebo_config as config


def _get_log_level() -> int:
    name = os.getenv("LOG_LEVEL", "INFO").upper()
    return getattr(logging, name, logging.INFO)


LOG_LEVEL = _get_log_level()

# Determine log directory and file
log_dir = getattr(config, "LOG_DIR", config.OUT_DIR)
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "eebo.log"

# Create logger
logger = logging.getLogger("eebo")
logger.setLevel(LOG_LEVEL)
logger.propagate = False  # IMPORTANT: avoid double logging via root

# Avoid duplicate handlers if imported multiple times
if not logger.handlers:
    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(LOG_LEVEL)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(LOG_LEVEL)
    ch.setFormatter(logging.Formatter(
        "[%(levelname)s] %(message)s"
    ))
    logger.addHandler(ch)
