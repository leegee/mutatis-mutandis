# eebo_logging.py
import sys
import logging

import .eebo_config as config

# Determine log directory and file
log_dir = getattr(config, "LOG_DIR", config.OUT_DIR)
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "eebo.log"

# Create logger
logger = logging.getLogger("eebo")
logger.setLevel(logging.DEBUG)

# Avoid duplicate handlers if imported multiple times
if not logger.handlers:
    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)
