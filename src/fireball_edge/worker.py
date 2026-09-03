"""Single-instance queue worker with an injectable event processor."""

from __future__ import annotations

import time
import logging
from collections.abc import Callable
from threading import Event, Thread
from typing import Any

from .config import EdgeConfig
from .contracts import CANDIDATE_EXTRACTOR, SCHEMA_VERSION
from .process_lock import process_lock
from .queue import EventQueue, QueueEvent, WorkerAlreadyRunningError


Processor = Callable[[QueueEvent], dict[str, Any]]
LOGGER = logging.getLogger("fireball_edge.worker")


class EdgeWorker:
    """Own queue execution; candidate extraction/inference is injected later."""

    def __init__(self, queue: EventQueue, config: EdgeConfig, dispatcher: Any | None = None) -> None:
        self.queue = queue
        self.config = config
        self.dispatcher = dispatcher

    def run_once(self, processor: Processor) -> QueueEvent | None:
        """Process one claim. The caller must already own the worker lock."""
        event = self.queue.claim_next()
        if event is None:
            return None
        try:
            result = processor(event)
        except Exception as error:  # Processor failures are durable, not fatal to the worker.
            message = f"{type(error).__name__}: {error}"
            LOGGER.warning("event=%s attempt=%s failed: %s", event.event_id, event.attempts, message)
            if event.attempts >= self.config.max_attempts:
                return self.queue.fail(event.event_id, message)
            delay = min(
                self.config.retry_max_seconds,
                self.config.retry_initial_seconds * (2 ** max(event.attempts - 1, 0)),
            )
            return self.queue.retry(event.event_id, message, delay_seconds=delay)
        notification = None
        if self.config.telegram_enabled and result.get("decision") in {
            "possible_fireball",
            "probable_fireball",
        }:
            notification = {
                # Store the environment-variable name, not a chat ID/secret.
                "destination": self.config.telegram_chat_id_env,
                "image_path": str(result["annotated_image"]),
                "caption": (
                    f"{result['decision']} score={float(result['calibrated_score']):.3f}\n"
                    f"event_id={event.event_id}"
                ),
            }
        completed = self.queue.complete(event.event_id, result, notification=notification)
        LOGGER.info("event=%s complete decision=%s", event.event_id, result.get("decision"))
        return completed

    def run(self, processor: Processor, *, once: bool = False, stop_event: Event | None = None) -> int:
        """Run under one SQLite lease and recover interrupted events on startup."""
        completed = 0
        with process_lock(self.queue.state_root / "worker.lock"), self.queue.worker_lock(
            self.config.worker_lock_lease_seconds,
            replace_existing=True,
        ) as owner_id:
            heartbeat_stop = Event()
            heartbeat_errors: list[Exception] = []

            def heartbeat() -> None:
                interval = max(0.1, self.config.worker_lock_lease_seconds / 3.0)
                while not heartbeat_stop.wait(interval):
                    try:
                        self.queue.renew_worker_lock(
                            owner_id, self.config.worker_lock_lease_seconds
                        )
                    except Exception as error:
                        heartbeat_errors.append(error)
                        return

            heartbeat_thread = Thread(target=heartbeat, name="edge-lock-heartbeat", daemon=True)
            heartbeat_thread.start()
            notification_stop = Event()
            notification_thread = None

            def deliver_notifications() -> None:
                while not notification_stop.is_set():
                    try:
                        delivered = self.dispatcher.dispatch_once()
                    except Exception:
                        LOGGER.exception("notification outbox iteration failed")
                        delivered = False
                    if not delivered:
                        notification_stop.wait(self.config.poll_interval_seconds)

            try:
                rebuilt_results = self.queue.requeue_incompatible_results(
                    required_schema_version=SCHEMA_VERSION,
                    candidate_extractor=CANDIDATE_EXTRACTOR,
                )
                recovered_events = self.queue.recover_processing()
                recovered_notifications = self.queue.recover_notifications()
                LOGGER.info(
                    "worker started rebuilt_results=%s recovered_events=%s recovered_notifications=%s",
                    rebuilt_results,
                    recovered_events,
                    recovered_notifications,
                )
                if self.dispatcher is not None and not once:
                    notification_thread = Thread(
                        target=deliver_notifications,
                        name="edge-notification-outbox",
                        daemon=True,
                    )
                    notification_thread.start()
                while stop_event is None or not stop_event.is_set():
                    if heartbeat_errors:
                        raise WorkerAlreadyRunningError("edge worker lock heartbeat failed")
                    event = self.run_once(processor)
                    if event is not None:
                        completed += 1
                    if once:
                        break
                    if event is None:
                        time.sleep(self.config.poll_interval_seconds)
            finally:
                notification_stop.set()
                if notification_thread is not None:
                    notification_thread.join(timeout=25.0)
                heartbeat_stop.set()
                heartbeat_thread.join(timeout=2.0)
        return completed
