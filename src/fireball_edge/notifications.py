"""Telegram delivery from a durable outbox, separate from inference."""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import secrets
import urllib.error
import urllib.request
from pathlib import Path
from typing import Mapping

from .config import EdgeConfig
from .queue import EventQueue


LOGGER = logging.getLogger("fireball_edge.notifications")


class TelegramDeliveryError(RuntimeError):
    pass


class TelegramTransport:
    """Small synchronous Bot API transport suitable for a worker thread."""

    def __init__(self, timeout_seconds: float = 20.0) -> None:
        self.timeout_seconds = timeout_seconds

    def send_photo(
        self,
        *,
        token: str,
        chat_id: str,
        image_path: str | Path,
        caption: str,
    ) -> str | None:
        path = Path(image_path)
        if path.stat().st_size > 10 * 1024 * 1024:
            raise TelegramDeliveryError("annotated photo exceeds Telegram's 10 MB upload limit")
        boundary = f"----fireball-edge-{secrets.token_hex(12)}"
        line = b"\r\n"
        parts: list[bytes] = []
        for name, value in (("chat_id", chat_id), ("caption", caption)):
            parts.extend(
                (
                    f"--{boundary}".encode(),
                    f'Content-Disposition: form-data; name="{name}"'.encode(),
                    b"",
                    value.encode("utf-8"),
                )
            )
        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        parts.extend(
            (
                f"--{boundary}".encode(),
                f'Content-Disposition: form-data; name="photo"; filename="{path.name}"'.encode(),
                f"Content-Type: {mime_type}".encode(),
                b"",
                path.read_bytes(),
                f"--{boundary}--".encode(),
                b"",
            )
        )
        request = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendPhoto",
            data=line.join(parts),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                document = json.loads(response.read())
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            # The token is part of the request URL. Never persist an exception
            # string that could echo that URL into the SQLite outbox.
            raise TelegramDeliveryError(
                f"Telegram request failed ({type(exc).__name__})"
            ) from exc
        if not document.get("ok"):
            raise TelegramDeliveryError(str(document.get("description", "Telegram rejected request")))
        message_id = document.get("result", {}).get("message_id")
        return str(message_id) if message_id is not None else None


class OutboxDispatcher:
    def __init__(
        self,
        queue: EventQueue,
        config: EdgeConfig,
        transport: TelegramTransport | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.queue = queue
        self.config = config
        self.transport = transport or TelegramTransport()
        self.environ = os.environ if environ is None else environ

    def dispatch_once(self) -> bool:
        row = self.queue.claim_notification()
        if row is None:
            return False
        event_id = row["event_id"]
        destination_env = row["destination"]
        try:
            token = self.environ.get(self.config.telegram_token_env, "")
            chat_id = self.environ.get(destination_env, "")
            if not token or not chat_id:
                raise TelegramDeliveryError(
                    "Telegram token or chat ID environment variable is unavailable"
                )
            remote_id = self.transport.send_photo(
                token=token,
                chat_id=chat_id,
                image_path=row["image_path"],
                caption=row["caption"],
            )
        except Exception as exc:
            LOGGER.warning("event=%s Telegram delivery deferred: %s", event_id, type(exc).__name__)
            self.queue.defer_notification(
                event_id,
                destination_env,
                f"{type(exc).__name__}: {exc}",
                max_attempts=self.config.telegram_max_attempts,
            )
        else:
            self.queue.finish_notification(
                event_id, destination_env, remote_message_id=remote_id
            )
            LOGGER.info("event=%s Telegram delivery complete", event_id)
        return True
