from __future__ import annotations

import logging
import os
from pathlib import Path

from .config import AppConfig


def configure_logging(config: AppConfig) -> Path:
    log_level = os.getenv("RCA_LOG_LEVEL", "INFO").upper()
    log_dir = config.data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "rca_app.log"

    root_logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    file_handler_exists = any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename).resolve() == log_path.resolve()
        for handler in root_logger.handlers
    )
    if not file_handler_exists:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    if not any(isinstance(handler, logging.StreamHandler) for handler in root_logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

    root_logger.setLevel(log_level)

    return log_path
