# eebo_logging.py
import sys
import os
import logging
from logging.handlers import RotatingFileHandler

import lib.eebo_config as config

def _in_colab():
    return "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ

name = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, name, logging.INFO)

# Create logger
logger = logging.getLogger("eebo")
logger.setLevel(LOG_LEVEL)
logger.propagate = False  # avoid double logging via root

# Avoid duplicate handlers if imported multiple times
if not logger.handlers:

    # Console handler (always enabled)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(LOG_LEVEL)
    ch.flush = sys.stdout.flush
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    # Attempt file logging only if not in Colab
    if not _in_colab():
        # Determine log directory and file
        log_dir = getattr(config, "LOG_DIR", config.OUT_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        suffix = os.getenv("EEBO_LOG_SUFFIX", str(os.getpid()))
        log_file = log_dir / f"eebo_{suffix}.log"

        # File handler with rotation
        fh = RotatingFileHandler(
            log_file,
            maxBytes=20 * 1024 * 1024,
            backupCount=10,
            encoding="utf-8",
        )
        fh.setLevel(LOG_LEVEL)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        logger.addHandler(fh)
