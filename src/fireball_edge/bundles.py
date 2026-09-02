"""Read-only discovery of files that belong to one UFOCapture event."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .event_id import normalize_clip_base as normalize_event_base


@dataclass(frozen=True)
class EventBundle:
    """Paths for an event without opening or modifying any source file."""

    clip_base: Path
    avi: Path | None
    peak: Path | None
    change_map: Path | None
    xml: Path | None

    @property
    def source_directory(self) -> Path:
        return self.clip_base.parent

    def sidecars_used(self) -> list[str]:
        return [
            str(path)
            for path in (self.avi, self.peak, self.change_map, self.xml)
            if path is not None
        ]

    def as_dict(self) -> dict[str, str | None]:
        return {
            key: str(value) if value is not None else None
            for key, value in asdict(self).items()
        }

    def source_identity(self) -> dict[str, dict[str, int | str]]:
        """Cheap identity for stale-result detection; full hashes are an audit tool."""
        result: dict[str, dict[str, int | str]] = {}
        for role, path in (
            ("avi", self.avi),
            ("peak", self.peak),
            ("change_map", self.change_map),
            ("xml", self.xml),
        ):
            if path is not None:
                stat = path.stat()
                result[role] = {
                    "path": str(path),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
        return result


def normalize_clip_base(value: str | Path) -> Path:
    """Return the event base for a base path or any conventional sidecar path.

    UFOCapture normally writes ``<base>.avi``, ``<base>P.bmp``,
    ``<base>M.bmp`` and ``<base>.xml``. The action hook may pass any of those
    paths, so accepting each form makes repeated invocations idempotent.
    """

    return Path(normalize_event_base(value))


def _case_insensitive_file(parent: Path, names: tuple[str, ...]) -> Path | None:
    """Resolve a known sidecar without recursively scanning the source tree."""

    for name in names:
        exact = parent / name
        if exact.is_file():
            return exact
    try:
        wanted = {name.casefold() for name in names}
        for child in parent.iterdir():
            if child.is_file() and child.name.casefold() in wanted:
                return child
    except (FileNotFoundError, PermissionError):
        return None
    return None


def discover_bundle(value: str | Path) -> EventBundle:
    """Discover an event's existing sidecars; the source remains read-only."""

    base = normalize_clip_base(value)
    name = base.name
    parent = base.parent
    return EventBundle(
        clip_base=base,
        avi=_case_insensitive_file(parent, (f"{name}.avi",)),
        peak=_case_insensitive_file(
            parent,
            (f"{name}P.bmp", f"{name}P.jpg", f"{name}P.jpeg", f"{name}P.png"),
        ),
        change_map=_case_insensitive_file(parent, (f"{name}M.bmp",)),
        xml=_case_insensitive_file(parent, (f"{name}.xml",)),
    )
