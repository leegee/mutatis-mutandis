# eebo_logging.py
import sys
import os
import logging
from logging.handlers import RotatingFileHandler
import lib.eebo_config as config

def _in_colab() -> bool:
    return "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ

# Determine log level from environment or default to INFO
name = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, name, logging.INFO)

# Create logger
logger = logging.getLogger("eebo")
logger.setLevel(LOG_LEVEL)
logger.propagate = False  # avoid double logging via root

# Avoid duplicate handlers if imported multiple times
if not logger.handlers:

    # Subclass StreamHandler to ensure flush works cleanly with mypy
    class StdoutHandler(logging.StreamHandler):
        def flush(self) -> None:
            sys.stdout.flush()
            super().flush()

    # Unified log format for console and file
    LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    # Console handler (always enabled)
    ch = StdoutHandler(sys.stdout)
    ch.setLevel(LOG_LEVEL)
    ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(ch)

    # File handler (only if not in Colab)
    if not _in_colab():
        log_dir = getattr(config, "LOG_DIR", config.OUT_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        suffix = os.getenv("EEBO_LOG_SUFFIX", str(os.getpid()))
        log_file = log_dir / f"eebo_{suffix}.log"

        fh = RotatingFileHandler(
            log_file,
            maxBytes=20 * 1024 * 1024,
            backupCount=10,
            encoding="utf-8",
        )
        fh.setLevel(LOG_LEVEL)
        fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(fh)
