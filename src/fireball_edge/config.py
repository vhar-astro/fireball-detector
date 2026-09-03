"""Configuration for data that must live outside UFOCapture directories."""

from __future__ import annotations

import json
import ntpath
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping


class StateRootError(ValueError):
    """Raised when edge state could be written into a monitored source tree."""


def default_state_root(
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
) -> Path:
    """Return the platform-appropriate, external default state location.

    Windows uses LOCALAPPDATA so the default is independent of the drive where
    UFOCapture stores clips.  The non-Windows branch is useful for development
    and CI only; production is packaged for Windows.
    """
    environment = os.environ if environ is None else environ
    current_platform = sys.platform if platform is None else platform
    if current_platform.startswith("win"):
        local_app_data = environment.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "FireballDetector"
        # LOCALAPPDATA is standard on supported Windows versions.  Falling
        # back to AppData makes a misconfigured shell usable without ever
        # falling back into a monitored folder.
        app_data = environment.get("APPDATA")
        if app_data:
            return Path(app_data) / "FireballDetector"
        return Path.home() / "AppData" / "Local" / "FireballDetector"
    return Path.home() / ".local" / "share" / "FireballDetector"


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_within(child: Path, parent: Path) -> bool:
    if os.name == "nt":
        # pathlib's lexical containment check is case-sensitive even though
        # Windows path lookup is not. Event IDs deliberately normalise case,
        # so queued clip bases must be checked with the same semantics.
        normalized_child = ntpath.normcase(ntpath.normpath(os.fspath(child)))
        normalized_parent = ntpath.normcase(ntpath.normpath(os.fspath(parent)))
        try:
            return ntpath.commonpath((normalized_child, normalized_parent)) == normalized_parent
        except ValueError:
            # Different drives cannot contain one another.
            return False
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_state_root(state_root: str | Path, monitored_roots: Iterable[str | Path]) -> Path:
    """Resolve and validate a state root before any state is created.

    Both directions of overlap are rejected.  In particular, allowing a state
    root below a monitored clip root would make the source tree mutable.
    """
    resolved_state = _resolved(state_root)
    for monitored_root in monitored_roots:
        resolved_monitored = _resolved(monitored_root)
        if _is_within(resolved_state, resolved_monitored) or _is_within(
            resolved_monitored, resolved_state
        ):
            raise StateRootError(
                "state_root must not overlap a monitored root: "
                f"{resolved_state} and {resolved_monitored}"
            )
    return resolved_state


@dataclass(frozen=True)
class EdgeConfig:
    """Runtime configuration. Secrets are named by environment variable only."""

    state_root: Path
    monitored_roots: tuple[Path, ...] = field(default_factory=tuple)
    model_manifest: Path | None = None
    max_inference_threads: int = 2
    poll_interval_seconds: float = 1.0
    max_attempts: int = 3
    retry_initial_seconds: float = 2.0
    retry_max_seconds: float = 30.0
    worker_lock_lease_seconds: float = 120.0
    telegram_enabled: bool = False
    telegram_token_env: str = "FIREBALL_TELEGRAM_BOT_TOKEN"
    telegram_chat_id_env: str = "FIREBALL_TELEGRAM_CHAT_ID"
    telegram_max_attempts: int = 5

    def __post_init__(self) -> None:
        # EdgeConfig is also constructed directly by offline tools and tests,
        # not only through load_config(). Canonicalising here expands Windows
        # 8.3 aliases (for example RUNNER~1) before any containment check.
        resolved_monitored = tuple(_resolved(root) for root in self.monitored_roots)
        resolved_state = validate_state_root(self.state_root, resolved_monitored)
        object.__setattr__(self, "state_root", resolved_state)
        object.__setattr__(self, "monitored_roots", resolved_monitored)
        if self.model_manifest is not None:
            resolved_manifest = _resolved(self.model_manifest)
            models_root = _resolved(resolved_state / "models")
            if not _is_within(resolved_manifest, models_root):
                raise StateRootError(
                    "model_manifest must be stored below state_root/models"
                )
            object.__setattr__(self, "model_manifest", resolved_manifest)
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if self.worker_lock_lease_seconds <= 0:
            raise ValueError("worker_lock_lease_seconds must be positive")
        if self.retry_initial_seconds < 0:
            raise ValueError("retry_initial_seconds must not be negative")
        if self.retry_max_seconds < self.retry_initial_seconds:
            raise ValueError("retry_max_seconds must be at least retry_initial_seconds")
        if self.max_inference_threads not in (1, 2):
            raise ValueError("max_inference_threads must be one or two")
        if self.telegram_max_attempts < 1:
            raise ValueError("telegram_max_attempts must be at least one")

    def validate_source(self, source: str | Path) -> Path:
        """Require sources to remain outside state and within configured roots."""
        resolved_source = _resolved(source)
        if _is_within(resolved_source, self.state_root) or _is_within(
            self.state_root, resolved_source
        ):
            raise StateRootError(
                f"source path must not overlap state_root: {resolved_source}"
            )
        if self.monitored_roots and not any(
            _is_within(resolved_source, root) for root in self.monitored_roots
        ):
            raise StateRootError(
                f"source path is outside configured monitored_roots: {resolved_source}"
            )
        return resolved_source

    def validate_clip_base(self, clip_base: str | Path) -> Path:
        """Validate a clip and its containing capture directory before writes."""
        resolved_clip = self.validate_source(clip_base)
        capture_directory = resolved_clip.parent
        if _is_within(self.state_root, capture_directory) or _is_within(
            capture_directory, self.state_root
        ):
            raise StateRootError(
                "state_root must not overlap the clip's capture directory: "
                f"{self.state_root} and {capture_directory}"
            )
        return resolved_clip


def load_config(
    config_path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
) -> EdgeConfig:
    """Load JSON configuration and refuse source-tree state before writing it."""
    document: dict[str, object] = {}
    if config_path is not None:
        with Path(config_path).expanduser().open("r", encoding="utf-8") as config_file:
            document = json.load(config_file)
        if not isinstance(document, dict):
            raise ValueError("edge configuration must be a JSON object")

    raw_monitored = document.get("monitored_roots", [])
    if not isinstance(raw_monitored, list) or not all(
        isinstance(item, str) for item in raw_monitored
    ):
        raise ValueError("monitored_roots must be a list of paths")
    monitored_roots = tuple(_resolved(item) for item in raw_monitored)

    raw_state_root = document.get("state_root")
    if raw_state_root is not None and not isinstance(raw_state_root, str):
        raise ValueError("state_root must be a path string")
    state_root = validate_state_root(
        raw_state_root or default_state_root(environ=environ, platform=platform), monitored_roots
    )

    raw_model_manifest = document.get("model_manifest")
    if raw_model_manifest is not None and not isinstance(raw_model_manifest, str):
        raise ValueError("model_manifest must be a path string")
    model_manifest = None
    if raw_model_manifest:
        raw_model_path = Path(raw_model_manifest).expanduser()
        model_manifest = _resolved(
            raw_model_path if raw_model_path.is_absolute() else state_root / raw_model_path
        )
    if model_manifest is not None:
        models_root = _resolved(state_root / "models")
        if not _is_within(model_manifest, models_root):
            raise StateRootError("model_manifest must be stored below state_root/models")

    worker = document.get("worker", {})
    if not isinstance(worker, dict):
        raise ValueError("worker must be a JSON object")
    poll_interval = worker.get("poll_interval_seconds", 1.0)
    max_attempts = worker.get("max_attempts", 3)
    retry_initial = worker.get("retry_initial_seconds", 2.0)
    retry_max = worker.get("retry_max_seconds", 30.0)
    lock_lease = worker.get("lock_lease_seconds", 120.0)
    max_threads = worker.get("max_inference_threads", 2)
    if isinstance(poll_interval, bool) or not isinstance(poll_interval, (int, float)):
        raise ValueError("worker.poll_interval_seconds must be numeric")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
        raise ValueError("worker.max_attempts must be an integer")
    if isinstance(retry_initial, bool) or not isinstance(retry_initial, (int, float)):
        raise ValueError("worker.retry_initial_seconds must be numeric")
    if isinstance(retry_max, bool) or not isinstance(retry_max, (int, float)):
        raise ValueError("worker.retry_max_seconds must be numeric")
    if isinstance(lock_lease, bool) or not isinstance(lock_lease, (int, float)):
        raise ValueError("worker.lock_lease_seconds must be numeric")
    if isinstance(max_threads, bool) or not isinstance(max_threads, int):
        raise ValueError("worker.max_inference_threads must be an integer")

    telegram = document.get("telegram", {})
    if not isinstance(telegram, dict):
        raise ValueError("telegram must be a JSON object")
    telegram_enabled = telegram.get("enabled", False)
    token_env = telegram.get("token_env", "FIREBALL_TELEGRAM_BOT_TOKEN")
    chat_id_env = telegram.get("chat_id_env", "FIREBALL_TELEGRAM_CHAT_ID")
    telegram_max_attempts = telegram.get("max_attempts", 5)
    if not isinstance(telegram_enabled, bool):
        raise ValueError("telegram.enabled must be a boolean")
    if not isinstance(token_env, str) or not token_env:
        raise ValueError("telegram.token_env must be a non-empty string")
    if not isinstance(chat_id_env, str) or not chat_id_env:
        raise ValueError("telegram.chat_id_env must be a non-empty string")
    if isinstance(telegram_max_attempts, bool) or not isinstance(telegram_max_attempts, int):
        raise ValueError("telegram.max_attempts must be an integer")
    return EdgeConfig(
        state_root=state_root,
        monitored_roots=monitored_roots,
        model_manifest=model_manifest,
        max_inference_threads=max_threads,
        poll_interval_seconds=float(poll_interval),
        max_attempts=max_attempts,
        retry_initial_seconds=float(retry_initial),
        retry_max_seconds=float(retry_max),
        worker_lock_lease_seconds=float(lock_lease),
        telegram_enabled=telegram_enabled,
        telegram_token_env=token_env,
        telegram_chat_id_env=chat_id_env,
        telegram_max_attempts=telegram_max_attempts,
    )
