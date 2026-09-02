"""External rotating logs for the long-running worker."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


_EDGE_HANDLER_ATTRIBUTE = "_fireball_edge_owned"


def close_logging() -> None:
    """Close handlers owned by the edge worker.

    This matters on Windows, where an open handler prevents removal of its
    state directory. It also keeps repeated in-process CLI calls from
    accumulating handlers for old state roots.
    """
    logger = logging.getLogger("fireball_edge")
    for handler in list(logger.handlers):
        if getattr(handler, _EDGE_HANDLER_ATTRIBUTE, False):
            logger.removeHandler(handler)
            handler.close()


def configure_logging(state_root: str | Path) -> Path:
    log_path = (Path(state_root) / "logs" / "fireball-edge.log").resolve(strict=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fireball_edge")
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        if not getattr(handler, _EDGE_HANDLER_ATTRIBUTE, False):
            continue
        if (
            isinstance(handler, RotatingFileHandler)
            and Path(handler.baseFilename) == log_path
        ):
            return log_path
        logger.removeHandler(handler)
        handler.close()

    handler = RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    setattr(handler, _EDGE_HANDLER_ATTRIBUTE, True)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)
    return log_path
