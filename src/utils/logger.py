"""
DataFlow360 — Logging Configuration
=====================================
Purpose : Provides a reusable, professional logger for all pipeline modules.
          Logs to both the terminal (console) and a rotating log file.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Pipeline started")
    logger.error("Something went wrong")
"""

import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------
# Configuration — read from .env with safe defaults
# ---------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "logs/dataflow360.log")

# Ensure the logs directory exists
Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Log Format
# Example output:
# 2026-10-08 21:30:00 | INFO     | ingestion.battery | Pipeline started
# ---------------------------------------------------------
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a named logger with console and file handlers.

    Args:
        name: Usually passed as __name__ from the calling module.
              This makes the log show WHICH file the message came from.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Prevent adding duplicate handlers if get_logger is called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVEL)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # ---------------------------
    # Handler 1: Console (terminal)
    # Shows logs in your terminal while the pipeline runs
    # ---------------------------
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(LOG_LEVEL)

    # ---------------------------
    # Handler 2: Rotating File
    # Writes logs to file. Rotates when file hits 5MB.
    # Keeps last 3 log files so disk doesn't fill up.
    # ---------------------------
    file_handler = RotatingFileHandler(
        filename=LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(LOG_LEVEL)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
