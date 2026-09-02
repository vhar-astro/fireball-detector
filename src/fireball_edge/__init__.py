"""Durable, external-state primitives for the UFOCapture edge worker.

Importing this package does not load ONNX Runtime, OpenCV, or network code; the
worker loads those pieces lazily. The edge package remains independent of the
legacy training and inference modules.
"""

import logging

from .config import EdgeConfig, StateRootError, default_state_root, load_config
from .event_id import event_id_for_clip_base, normalize_clip_base
from .queue import EventQueue, QueueEvent, WorkerAlreadyRunningError


package_logger = logging.getLogger("fireball_edge")
package_logger.addHandler(logging.NullHandler())
package_logger.propagate = False

__all__ = [
    "EdgeConfig",
    "EventQueue",
    "QueueEvent",
    "StateRootError",
    "WorkerAlreadyRunningError",
    "default_state_root",
    "event_id_for_clip_base",
    "load_config",
    "normalize_clip_base",
]
