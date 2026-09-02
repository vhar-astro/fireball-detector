"""External rotating logs for the long-running worker."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(state_root: str | Path) -> Path:
    log_path = Path(state_root) / "logs" / "fireball-edge.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fireball_edge")
    logger.setLevel(logging.INFO)
    if not any(
        isinstance(handler, RotatingFileHandler)
        and Path(handler.baseFilename) == log_path.resolve(strict=False)
        for handler in logger.handlers
    ):
        handler = RotatingFileHandler(
            log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        logger.addHandler(handler)
    return log_path
