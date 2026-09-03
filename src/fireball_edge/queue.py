"""SQLite-backed, idempotent work queue kept wholly outside capture folders."""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .event_id import event_id_for_clip_base, normalize_clip_base


STATES = frozenset({"queued", "processing", "complete", "retry", "failed"})


class WorkerAlreadyRunningError(RuntimeError):
    """Raised when another live worker owns the queue singleton lease."""


@dataclass(frozen=True)
class QueueEvent:
    event_id: str
    clip_base: str
    state: str
    attempts: int
    created_at: float
    updated_at: float
    claimed_at: float | None
    next_attempt_at: float
    last_error: str | None
    result: dict[str, Any] | None


class EventQueue:
    """Queue state transitions with atomic claims and a renewable worker lease."""

    def __init__(self, state_root: str | Path) -> None:
        self.state_root = Path(state_root).expanduser().resolve(strict=False)
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.database_path = self.state_root / "queue.sqlite3"
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30, isolation_level=None)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 30000")
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    clip_base TEXT NOT NULL,
                    state TEXT NOT NULL
                        CHECK (state IN ('queued', 'processing', 'complete', 'retry', 'failed')),
                    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    claimed_at REAL,
                    last_error TEXT,
                    result_json TEXT
                );
                CREATE INDEX IF NOT EXISTS events_claimable
                    ON events(state, created_at);
                CREATE TABLE IF NOT EXISTS legacy_event_results (
                    event_id TEXT NOT NULL,
                    archived_at REAL NOT NULL,
                    result_json TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    PRIMARY KEY(event_id, archived_at)
                );
                CREATE TABLE IF NOT EXISTS legacy_notification_outbox (
                    event_id TEXT NOT NULL,
                    destination TEXT NOT NULL,
                    archived_at REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    PRIMARY KEY(event_id, destination, archived_at)
                );
                CREATE TABLE IF NOT EXISTS notification_outbox (
                    event_id TEXT NOT NULL REFERENCES events(event_id),
                    destination TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    caption TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'pending'
                        CHECK (state IN ('pending', 'sending', 'retry', 'sent', 'failed')),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at REAL NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL,
                    last_error TEXT,
                    remote_message_id TEXT,
                    PRIMARY KEY(event_id, destination)
                );
                CREATE INDEX IF NOT EXISTS outbox_deliverable
                    ON notification_outbox(state, next_attempt_at, updated_at);
                CREATE TABLE IF NOT EXISTS worker_locks (
                    lock_name TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    acquired_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                );
                """
            )
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(events)")
            }
            if "next_attempt_at" not in columns:
                connection.execute(
                    "ALTER TABLE events ADD COLUMN next_attempt_at REAL NOT NULL DEFAULT 0"
                )
            connection.execute("PRAGMA journal_mode = WAL")

    @staticmethod
    def _event_from_row(row: sqlite3.Row | None) -> QueueEvent | None:
        if row is None:
            return None
        result = json.loads(row["result_json"]) if row["result_json"] else None
        return QueueEvent(
            event_id=row["event_id"],
            clip_base=row["clip_base"],
            state=row["state"],
            attempts=row["attempts"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            claimed_at=row["claimed_at"],
            next_attempt_at=row["next_attempt_at"],
            last_error=row["last_error"],
            result=result,
        )

    @staticmethod
    def _result_is_compatible(
        result_json: str | None,
        schema_version: int,
        candidate_extractor: str,
    ) -> bool:
        try:
            document = json.loads(result_json) if result_json else None
        except json.JSONDecodeError:
            return False
        return bool(
            isinstance(document, dict)
            and document.get("schema_version") == schema_version
            and document.get("candidate_extractor") == candidate_extractor
        )

    @staticmethod
    def _archive_and_requeue(
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        now: float,
        reason: str,
    ) -> None:
        if row["result_json"]:
            connection.execute(
                """
                INSERT INTO legacy_event_results(event_id, archived_at, result_json, reason)
                VALUES (?, ?, ?, ?)
                """,
                (row["event_id"], now, row["result_json"], reason),
            )
        notification_rows = connection.execute(
            "SELECT * FROM notification_outbox WHERE event_id = ?",
            (row["event_id"],),
        ).fetchall()
        for notification in notification_rows:
            connection.execute(
                """
                INSERT INTO legacy_notification_outbox(
                    event_id, destination, archived_at, payload_json, reason
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    row["event_id"],
                    notification["destination"],
                    now,
                    json.dumps(dict(notification), sort_keys=True, allow_nan=False),
                    reason,
                ),
            )
        connection.execute(
            "DELETE FROM notification_outbox WHERE event_id = ?",
            (row["event_id"],),
        )
        connection.execute(
            """
            UPDATE events
            SET state = 'queued', attempts = 0, updated_at = ?, claimed_at = NULL,
                next_attempt_at = 0, last_error = ?, result_json = NULL
            WHERE event_id = ? AND state = 'complete'
            """,
            (now, reason, row["event_id"]),
        )

    def enqueue(
        self,
        clip_base: str | Path,
        *,
        required_schema_version: int | None = None,
        candidate_extractor: str | None = None,
    ) -> QueueEvent:
        """Insert a new event, or return the original event for duplicate calls."""
        if (required_schema_version is None) != (candidate_extractor is None):
            raise ValueError("result schema and candidate extractor must be supplied together")
        normalized = normalize_clip_base(clip_base)
        event_id = event_id_for_clip_base(normalized)
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO events(event_id, clip_base, state, attempts, created_at, updated_at)
                VALUES (?, ?, 'queued', 0, ?, ?)
                ON CONFLICT(event_id) DO NOTHING
                """,
                (event_id, normalized, now, now),
            )
            row = connection.execute(
                "SELECT * FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
            if (
                row is not None
                and row["state"] == "complete"
                and required_schema_version is not None
                and candidate_extractor is not None
                and not self._result_is_compatible(
                    row["result_json"], required_schema_version, candidate_extractor
                )
            ):
                self._archive_and_requeue(
                    connection,
                    row,
                    now,
                    "completed result is incompatible with the current v2 contract",
                )
                row = connection.execute(
                    "SELECT * FROM events WHERE event_id = ?", (event_id,)
                ).fetchone()
            connection.execute("COMMIT")
        event = self._event_from_row(row)
        assert event is not None
        return event

    def requeue_incompatible_results(
        self, *, required_schema_version: int, candidate_extractor: str
    ) -> int:
        """Archive complete legacy rows and make them claimable for a v2 rebuild."""

        now = time.time()
        count = 0
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT * FROM events WHERE state = 'complete'"
            ).fetchall()
            for index, row in enumerate(rows):
                if self._result_is_compatible(
                    row["result_json"], required_schema_version, candidate_extractor
                ):
                    continue
                # Keep the composite primary key unique even on coarse clocks.
                archived_at = now + index * 1e-6
                self._archive_and_requeue(
                    connection,
                    row,
                    archived_at,
                    "completed result is incompatible with the current v2 contract",
                )
                count += 1
            connection.execute("COMMIT")
        return count

    def get(self, event_id: str) -> QueueEvent | None:
        with self._connect() as connection:
            return self._event_from_row(
                connection.execute("SELECT * FROM events WHERE event_id = ?", (event_id,)).fetchone()
            )

    def claim_next(self) -> QueueEvent | None:
        """Atomically claim the oldest queued or retryable event."""
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM events
                WHERE state = 'queued' OR (state = 'retry' AND next_attempt_at <= ?)
                ORDER BY created_at, event_id
                LIMIT 1
                """,
                (now,),
            ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return None
            event_id = row["event_id"]
            connection.execute(
                """
                UPDATE events
                SET state = 'processing', attempts = attempts + 1,
                    claimed_at = ?, updated_at = ?, next_attempt_at = 0,
                    last_error = NULL
                WHERE event_id = ?
                """,
                (now, now, event_id),
            )
            claimed = connection.execute(
                "SELECT * FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
            connection.execute("COMMIT")
        return self._event_from_row(claimed)

    def _transition(
        self,
        event_id: str,
        state: str,
        *,
        error: str | None = None,
        result: dict[str, Any] | None = None,
        next_attempt_at: float = 0.0,
    ) -> QueueEvent:
        if state not in STATES - {"queued", "processing"}:
            raise ValueError(f"unsupported terminal state: {state}")
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                """
                UPDATE events
                SET state = ?, updated_at = ?, last_error = ?, result_json = ?,
                    next_attempt_at = ?
                WHERE event_id = ? AND state = 'processing'
                """,
                (
                    state,
                    now,
                    error,
                    json.dumps(result, sort_keys=True, allow_nan=False)
                    if result is not None
                    else None,
                    next_attempt_at,
                    event_id,
                ),
            )
            if updated.rowcount != 1:
                connection.execute("ROLLBACK")
                raise ValueError(f"event is not processing: {event_id}")
            row = connection.execute(
                "SELECT * FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
            connection.execute("COMMIT")
        event = self._event_from_row(row)
        assert event is not None
        return event

    def complete(
        self,
        event_id: str,
        result: dict[str, Any],
        notification: dict[str, str] | None = None,
    ) -> QueueEvent:
        """Commit inference and its optional notification as one transaction."""
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                """
                UPDATE events SET state = 'complete', updated_at = ?, last_error = NULL,
                    result_json = ? WHERE event_id = ? AND state = 'processing'
                """,
                (now, json.dumps(result, sort_keys=True, allow_nan=False), event_id),
            )
            if updated.rowcount != 1:
                connection.execute("ROLLBACK")
                raise ValueError(f"event is not processing: {event_id}")
            if notification is not None:
                connection.execute(
                    """
                    INSERT INTO notification_outbox(
                        event_id, destination, image_path, caption, state,
                        attempts, next_attempt_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'pending', 0, 0, ?)
                    ON CONFLICT(event_id, destination) DO NOTHING
                    """,
                    (
                        event_id,
                        notification["destination"],
                        notification["image_path"],
                        notification["caption"],
                        now,
                    ),
                )
            row = connection.execute(
                "SELECT * FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
            connection.execute("COMMIT")
        event = self._event_from_row(row)
        assert event is not None
        return event

    def claim_notification(self) -> sqlite3.Row | None:
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM notification_outbox
                WHERE state IN ('pending', 'retry') AND next_attempt_at <= ?
                ORDER BY updated_at LIMIT 1
                """,
                (now,),
            ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return None
            connection.execute(
                """
                UPDATE notification_outbox
                SET state = 'sending', attempts = attempts + 1, updated_at = ?
                WHERE event_id = ? AND destination = ?
                """,
                (now, row["event_id"], row["destination"]),
            )
            claimed = connection.execute(
                """SELECT * FROM notification_outbox
                   WHERE event_id = ? AND destination = ?""",
                (row["event_id"], row["destination"]),
            ).fetchone()
            connection.execute("COMMIT")
            return claimed

    def finish_notification(
        self,
        event_id: str,
        destination: str,
        *,
        remote_message_id: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE notification_outbox SET state = 'sent', updated_at = ?,
                    last_error = NULL, remote_message_id = ?
                WHERE event_id = ? AND destination = ? AND state = 'sending'
                """,
                (time.time(), remote_message_id, event_id, destination),
            )

    def defer_notification(
        self,
        event_id: str,
        destination: str,
        error: str,
        *,
        max_attempts: int,
    ) -> None:
        now = time.time()
        with self._connect() as connection:
            row = connection.execute(
                """SELECT attempts FROM notification_outbox
                   WHERE event_id = ? AND destination = ?""",
                (event_id, destination),
            ).fetchone()
            if row is None:
                raise ValueError("notification does not exist")
            state = "failed" if row["attempts"] >= max_attempts else "retry"
            delay = min(300.0, float(2 ** min(row["attempts"], 8)))
            connection.execute(
                """
                UPDATE notification_outbox SET state = ?, next_attempt_at = ?,
                    updated_at = ?, last_error = ?
                WHERE event_id = ? AND destination = ? AND state = 'sending'
                """,
                (state, now + delay, now, error, event_id, destination),
            )

    def recover_notifications(self) -> int:
        with self._connect() as connection:
            updated = connection.execute(
                """UPDATE notification_outbox SET state = 'retry', updated_at = ?
                   WHERE state = 'sending'""",
                (time.time(),),
            )
            return updated.rowcount

    def retry(self, event_id: str, error: str, *, delay_seconds: float = 0.0) -> QueueEvent:
        if delay_seconds < 0:
            raise ValueError("retry delay must not be negative")
        return self._transition(
            event_id,
            "retry",
            error=error,
            next_attempt_at=time.time() + delay_seconds,
        )

    def fail(self, event_id: str, error: str) -> QueueEvent:
        return self._transition(event_id, "failed", error=error)

    def recover_processing(self, reason: str = "recovered after worker restart") -> int:
        """Make interrupted events eligible for a fresh claim after restart."""
        now = time.time()
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE events
                SET state = 'retry', updated_at = ?, claimed_at = NULL,
                    next_attempt_at = ?, last_error = ?
                WHERE state = 'processing'
                """,
                (now, now, reason),
            )
            return updated.rowcount

    def acquire_worker_lock(
        self, owner_id: str, lease_seconds: float, *, replace_existing: bool = False
    ) -> None:
        """Acquire the SQLite singleton lease, replacing only expired owners."""
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if replace_existing:
                connection.execute("DELETE FROM worker_locks WHERE lock_name = 'edge-worker'")
            else:
                connection.execute(
                    "DELETE FROM worker_locks WHERE lock_name = 'edge-worker' AND expires_at <= ?",
                    (now,),
                )
            try:
                connection.execute(
                    """
                    INSERT INTO worker_locks(lock_name, owner_id, acquired_at, expires_at)
                    VALUES ('edge-worker', ?, ?, ?)
                    """,
                    (owner_id, now, now + lease_seconds),
                )
            except sqlite3.IntegrityError as error:
                connection.execute("ROLLBACK")
                raise WorkerAlreadyRunningError("another edge worker holds the queue lock") from error
            connection.execute("COMMIT")

    def renew_worker_lock(self, owner_id: str, lease_seconds: float) -> None:
        now = time.time()
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE worker_locks SET expires_at = ?
                WHERE lock_name = 'edge-worker' AND owner_id = ? AND expires_at > ?
                """,
                (now + lease_seconds, owner_id, now),
            )
            if updated.rowcount != 1:
                raise WorkerAlreadyRunningError("edge worker lock was lost")

    def release_worker_lock(self, owner_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM worker_locks WHERE lock_name = 'edge-worker' AND owner_id = ?",
                (owner_id,),
            )

    @contextmanager
    def worker_lock(
        self, lease_seconds: float, *, replace_existing: bool = False
    ) -> Iterator[str]:
        owner_id = f"{os.getpid()}-{uuid.uuid4()}"
        self.acquire_worker_lock(
            owner_id, lease_seconds, replace_existing=replace_existing
        )
        try:
            yield owner_id
        finally:
            self.release_worker_lock(owner_id)
