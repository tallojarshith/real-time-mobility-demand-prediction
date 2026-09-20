import logging
import os
from datetime import datetime


LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a configured logger.

    Logs are written both to:
    1. Console
    2. Daily log file inside logs/
    """

    logger = logging.getLogger(name)

    # Prevent duplicate handlers if get_logger()
    # is called multiple times.
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    log_filename = os.path.join(
        LOG_DIR,
        f"app_{datetime.now().strftime('%Y-%m-%d')}.log"
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    )

    # File logging
    file_handler = logging.FileHandler(
        log_filename,
        encoding="utf-8"
    )

    file_handler.setFormatter(formatter)

    # Console logging
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger