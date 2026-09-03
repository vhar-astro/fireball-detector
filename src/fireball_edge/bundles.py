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
    stack_image: Path | None
    star_mask: Path | None
    xml: Path | None

    @property
    def source_directory(self) -> Path:
        return self.clip_base.parent

    def source_files(self) -> list[str]:
        """All discovered bundle files, for source provenance only.

        In particular, ``star_mask`` is recorded here but is never an input to
        candidate extraction, temporal measurements, calibration, or scoring.
        """
        paths = (
            self.avi,
            *stack_image_candidates(self),
            self.star_mask,
            self.xml,
        )
        return list(dict.fromkeys(str(path) for path in paths if path is not None))

    # Kept as a method name for result writers from v1.  It intentionally
    # returns provenance, not an assertion that every listed path was scored.
    def sidecars_used(self) -> list[str]:
        return self.source_files()

    def as_dict(self) -> dict[str, str | None]:
        return {
            key: str(value) if value is not None else None
            for key, value in asdict(self).items()
        }

    def source_identity(self) -> dict[str, dict[str, int | str]]:
        """Cheap identity for stale-result detection; full hashes are an audit tool."""
        result: dict[str, dict[str, int | str]] = {}
        identities: list[tuple[str, Path | None]] = [
            ("avi", self.avi),
            ("stack_image", self.stack_image),
            ("star_mask", self.star_mask),
            ("xml", self.xml),
        ]
        alternates = [path for path in stack_image_candidates(self) if path != self.stack_image]
        identities.extend(
            (f"stack_image_alternate_{index}", path)
            for index, path in enumerate(alternates, start=1)
        )
        for role, path in identities:
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
        # A lossless P.bmp is the canonical classifier image.  P.jpg is an
        # explicitly supported fallback only when the BMP is absent; validity
        # is checked later because discovery must remain read-only and cheap.
        stack_image=_case_insensitive_file(
            parent,
            (f"{name}P.bmp", f"{name}P.jpg"),
        ),
        star_mask=_case_insensitive_file(parent, (f"{name}M.bmp",)),
        xml=_case_insensitive_file(parent, (f"{name}.xml",)),
    )


def stack_image_candidates(bundle: EventBundle) -> tuple[Path, ...]:
    """Return P-stack paths in deterministic lossless-first order.

    Discovery selects the BMP name cheaply, but callers that decode images
    must use this function so a corrupt BMP can fall back to a valid JPG.
    """

    name = bundle.clip_base.name
    parent = bundle.clip_base.parent
    paths: list[Path] = []
    for names in ((f"{name}P.bmp",), (f"{name}P.jpg",)):
        path = _case_insensitive_file(parent, names)
        if path is not None and path not in paths:
            paths.append(path)
    return tuple(paths)
