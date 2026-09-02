"""Atomic publication of externally stored result artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable


def _fsync_file(path: Path) -> None:
    # Windows' CRT rejects _commit (used by os.fsync) for read-only file
    # descriptors, even though POSIX accepts fsync on them.
    with path.open("r+b") as source:
        os.fsync(source.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":  # Windows does not allow opening directories this way.
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_image_atomic(destination: str | Path, writer: Callable[[Path], Any]) -> Path:
    """Call an image writer under the destination directory, then replace."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.staging{destination.suffix}")
    try:
        writer(temporary)
        _fsync_file(temporary)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def write_json_atomic(destination: str | Path, document: dict[str, Any]) -> Path:
    """Write the result commit marker last and durably replace it."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.staging")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            json.dump(document, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def read_committed_result(
    destination: str | Path,
    *,
    event_id: str,
    clip_base: str,
    model_version: str,
    model_sha256: str,
    model_manifest_sha256: str,
    source_identity: dict[str, Any],
) -> dict[str, Any] | None:
    """Reconcile a complete matching result after an interrupted DB update."""

    path = Path(destination)
    try:
        with path.open("r", encoding="utf-8") as source:
            document = json.load(source)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        return None
    if (
        document.get("event_id") != event_id
        or document.get("clip_base") != clip_base
        or document.get("model_version") != model_version
        or document.get("model_sha256") != model_sha256
        or document.get("model_manifest_sha256") != model_manifest_sha256
        or document.get("source_identity") != source_identity
        or document.get("candidate_extractor")
        != "change-map-red-v1-with-avi-fallback"
    ):
        return None
    annotation = document.get("annotated_image")
    if not isinstance(annotation, str):
        return None
    annotation_path = Path(annotation).resolve(strict=False)
    try:
        annotation_path.relative_to(path.parent.resolve(strict=False))
    except ValueError:
        return None
    if not annotation_path.is_file():
        return None
    return document
