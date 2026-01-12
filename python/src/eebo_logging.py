import logging
import sys
import eebo_config as config


def setup_logging(name="eebo", level=logging.INFO):
    """
    Configure file + console logging.
    Safe to call once at program start.
    """
    log_dir = getattr(config, "LOG_DIR", config.OUT_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger  # avoids duplicate handlers

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    fh.setLevel(level)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    ch.setLevel(level)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
