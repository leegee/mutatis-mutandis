# eebo_logging.py
import sys
import os
import logging
from logging.handlers import RotatingFileHandler

import lib.eebo_config as config



name = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(
    logging,
    name,
    logging.INFO
)

# Determine log directory and file
log_dir = getattr(config, "LOG_DIR", config.OUT_DIR)
log_dir.mkdir(parents=True, exist_ok=True)

suffix = os.getenv("EEBO_LOG_SUFFIX", str(os.getpid()))
log_file = log_dir / f"eebo_{suffix}.log"


# Create logger
logger = logging.getLogger("eebo")
logger.setLevel(LOG_LEVEL)
logger.propagate = False  # IMPORTANT: avoid double logging via root

# Avoid duplicate handlers if imported multiple times
if not logger.handlers:
    # File handler
    # fh = logging.FileHandler(log_file, encoding="utf-8")
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

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(LOG_LEVEL)
    ch.setFormatter(logging.Formatter(
        "[%(levelname)s] %(message)s"
    ))
    logger.addHandler(ch)
