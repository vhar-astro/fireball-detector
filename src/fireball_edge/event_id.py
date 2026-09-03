"""Stable identifiers for a physical UFOCapture event bundle."""

from __future__ import annotations

import hashlib
import ntpath
import os
from pathlib import Path


def normalize_clip_base(clip_base: str | Path) -> str:
    """Return a canonical path representation without touching its sidecars.

    The capture-end action may deliver equivalent spellings of a clip base.
    Resolving dot segments and normalising Windows case makes those calls
    idempotent.  A *clip base* is intentionally not suffixed with ``.avi``:
    it names the AVI and its P/M/XML sidecars as one event bundle.
    """
    raw = os.fspath(clip_base)
    if not raw:
        raise ValueError("clip_base must not be empty")
    stem, suffix = os.path.splitext(raw)
    if suffix.casefold() in {".avi", ".xml"}:
        raw = stem
    elif suffix.casefold() in {".bmp", ".jpg", ".jpeg", ".png"} and stem.endswith(("P", "M")):
        raw = stem[:-1]
    if os.name == "nt":
        return ntpath.normcase(ntpath.normpath(os.path.abspath(raw)))
    return str(Path(raw).expanduser().resolve(strict=False))


def event_id_for_clip_base(clip_base: str | Path) -> str:
    """Create a deterministic event ID suitable as a SQLite primary key."""
    normalized = normalize_clip_base(clip_base)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"evt_{digest}"
