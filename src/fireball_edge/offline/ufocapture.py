"""Bounded UFOCapture XML parsing and offline bundle preflight.

The metadata in this module is dataset-governance data only.  It must never be
passed to the classifier or calibration feature vector.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as element_tree
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from ..bundles import EventBundle, stack_image_candidates
from ..vision import InvalidMediaError, _vision_deps


MAX_XML_BYTES = 1_048_576
MAX_XML_ELEMENTS = 256
MAX_XML_DEPTH = 16
MAX_ATTRIBUTE_LENGTH = 4096


class UfoCaptureMetadataError(ValueError):
    """Raised when a required, bounded UFOCapture record cannot be parsed."""


class BundlePreflightError(ValueError):
    """Raised when a training bundle is incomplete or media is inconsistent."""


@dataclass(frozen=True)
class UfoCaptureRecord:
    """Non-predictive metadata parsed from one ``ufocapture_record`` XML file."""

    timestamp: str
    timezone_seconds: int
    width: int
    height: int
    fps: float | None
    frame_count: int | None
    codec: str | None
    dropped_frames: int | None
    countrycode: str
    lid: str
    sid: str
    cam: str
    lens: str
    station: str
    camera: str
    camera_night: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MediaInfo:
    width: int
    height: int
    fps: float | None
    frame_count: int | None
    codec: str | None


@dataclass(frozen=True)
class PreflightBundle:
    bundle: EventBundle
    metadata: UfoCaptureRecord
    avi: MediaInfo
    stack: MediaInfo
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "clip_base": str(self.bundle.clip_base),
            "avi": str(self.bundle.avi),
            "stack_image": str(self.bundle.stack_image),
            "star_mask": str(self.bundle.star_mask) if self.bundle.star_mask else None,
            "xml": str(self.bundle.xml),
            "star_mask_role": "provenance_only",
            "xml_validation": "valid",
            "metadata": self.metadata.as_dict(),
            "warnings": list(self.warnings),
        }


def _normalise_name(value: str) -> str:
    return value.rsplit("}", 1)[-1].casefold().replace("_", "").replace("-", "")


def _attribute_map(root: element_tree.Element) -> dict[str, str]:
    attributes: dict[str, str] = {}
    pending = [(root, 1)]
    visited = 0
    while pending:
        node, depth = pending.pop()
        visited += 1
        if visited > MAX_XML_ELEMENTS:
            raise UfoCaptureMetadataError("XML contains too many elements")
        if depth > MAX_XML_DEPTH:
            raise UfoCaptureMetadataError("XML nesting is too deep")
        for name, value in node.attrib.items():
            if len(name) > MAX_ATTRIBUTE_LENGTH or len(value) > MAX_ATTRIBUTE_LENGTH:
                raise UfoCaptureMetadataError("XML attribute exceeds parser limits")
            key = _normalise_name(name)
            # First occurrence wins: nested diagnostics cannot silently replace
            # root capture metadata.
            attributes.setdefault(key, value.strip())
        if node.text and not node.attrib:
            text = node.text.strip()
            if text:
                attributes.setdefault(_normalise_name(node.tag), text)
        pending.extend((child, depth + 1) for child in reversed(list(node)))
    return attributes


def _value(attributes: dict[str, str], *names: str, required: bool = False) -> str | None:
    for name in names:
        value = attributes.get(_normalise_name(name))
        if value:
            return value
    if required:
        raise UfoCaptureMetadataError("XML missing required attribute: " + names[0])
    return None


def _integer(value: str | None, name: str, *, required: bool = False) -> int | None:
    if value is None:
        if required:
            raise UfoCaptureMetadataError(f"XML missing required attribute: {name}")
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise UfoCaptureMetadataError(f"XML {name} must be an integer") from exc
    if parsed < 0:
        raise UfoCaptureMetadataError(f"XML {name} must not be negative")
    return parsed


def _signed_integer(value: str | None, name: str, *, required: bool = False) -> int | None:
    if value is None:
        if required:
            raise UfoCaptureMetadataError(f"XML missing required attribute: {name}")
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise UfoCaptureMetadataError(f"XML {name} must be an integer") from exc


def _float(value: str | None, name: str) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise UfoCaptureMetadataError(f"XML {name} must be numeric") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise UfoCaptureMetadataError(f"XML {name} must be finite and positive")
    return parsed


def _parse_timestamp(value: str, timezone_seconds: int) -> datetime:
    cleaned = value.strip().replace("Z", "+00:00")
    # UFOCapture samples use either ISO-like values or a space separated local
    # date/time.  ``fromisoformat`` preserves fractional seconds.
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        parsed = None
        for pattern in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S", "%Y%m%d_%H%M%S.%f", "%Y%m%d_%H%M%S"):
            try:
                parsed = datetime.strptime(cleaned, pattern)
                break
            except ValueError:
                continue
        if parsed is None:
            raise UfoCaptureMetadataError("XML timestamp is not a supported date/time")
    offset = timezone(timedelta(seconds=timezone_seconds))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=offset)
    return parsed.astimezone(offset)


def _component_timestamp(attributes: dict[str, str]) -> str | None:
    """Build the conventional UFOCapture y/mo/d/h/m/s local timestamp."""

    values = [_value(attributes, name) for name in ("y", "mo", "d", "h", "m", "s")]
    if not any(value is not None for value in values):
        return None
    if any(value is None for value in values):
        raise UfoCaptureMetadataError("XML timestamp components must include y/mo/d/h/m/s")
    year, month, day, hour, minute, second = values
    try:
        return (
            f"{int(year):04d}-{int(month):02d}-{int(day):02d}T"
            f"{int(hour):02d}:{int(minute):02d}:{float(second):09.6f}"
        )
    except (TypeError, ValueError) as exc:
        raise UfoCaptureMetadataError("XML timestamp components must be numeric") from exc


def parse_ufocapture_record(path: str | Path) -> UfoCaptureRecord:
    """Parse one small ``ufocapture_record`` file without unbounded XML input."""

    xml_path = Path(path)
    try:
        # Read at most one byte beyond the contract limit; never materialise an
        # arbitrarily large untrusted XML sidecar in memory.
        with xml_path.open("rb") as source:
            payload = source.read(MAX_XML_BYTES + 1)
    except OSError as exc:
        raise UfoCaptureMetadataError(f"cannot read XML: {xml_path}") from exc
    if len(payload) > MAX_XML_BYTES:
        raise UfoCaptureMetadataError("XML exceeds the one-megabyte parser limit")
    if re.search(br"<!\s*(DOCTYPE|ENTITY)\b", payload, flags=re.IGNORECASE):
        raise UfoCaptureMetadataError("DTD and entity declarations are not allowed")
    try:
        root = element_tree.fromstring(payload)
    except element_tree.ParseError as exc:
        raise UfoCaptureMetadataError(f"malformed XML: {exc}") from exc
    if _normalise_name(root.tag) != "ufocapturerecord":
        raise UfoCaptureMetadataError("XML root must be ufocapture_record")
    attributes = _attribute_map(root)
    timezone_seconds = _signed_integer(_value(attributes, "tz", "timezone", required=True), "tz", required=True)
    assert timezone_seconds is not None
    if abs(timezone_seconds) >= 24 * 60 * 60:
        raise UfoCaptureMetadataError("XML tz is outside a one-day offset")
    timestamp = _value(attributes, "timestamp", "datetime", "date_time", "capturetime")
    if timestamp is None:
        date = _value(attributes, "date")
        time = _value(attributes, "time")
        timestamp = f"{date} {time}" if date and time else time
    if timestamp is None:
        timestamp = _component_timestamp(attributes)
    if timestamp is None:
        raise UfoCaptureMetadataError("XML missing required attribute: timestamp")
    assert timestamp is not None
    local_capture = _parse_timestamp(timestamp, timezone_seconds)
    width = _integer(_value(attributes, "width", "imagewidth", "xsize", "cx", "w", required=True), "width", required=True)
    height = _integer(_value(attributes, "height", "imageheight", "ysize", "cy", required=True), "height", required=True)
    assert width is not None and height is not None
    if width == 0 or height == 0 or width > 32768 or height > 32768:
        raise UfoCaptureMetadataError("XML dimensions must be in 1..32768")
    countrycode = _value(attributes, "countrycode", "country", "cc", required=True)
    lid = _value(attributes, "lid", required=True)
    sid = _value(attributes, "sid", required=True)
    cam = _value(attributes, "cam", "camera", required=True)
    lens = _value(attributes, "lens", required=True)
    assert countrycode is not None and lid is not None and sid is not None and cam is not None and lens is not None
    station = f"{countrycode}:{lid}:{sid}"
    camera = f"{station}:{cam}:{lens}"
    camera_night = (local_capture - timedelta(hours=12)).date().isoformat()
    dropped_frames = _signed_integer(
        _value(attributes, "droppedframes", "dropped", "dropframes", "dropframe", "drop"),
        "dropped_frames",
    )
    if dropped_frames is not None and dropped_frames < -1:
        raise UfoCaptureMetadataError("XML dropped_frames must be -1 or non-negative")
    return UfoCaptureRecord(
        timestamp=local_capture.isoformat(),
        timezone_seconds=timezone_seconds,
        width=width,
        height=height,
        fps=_float(_value(attributes, "fps", "framerate"), "fps"),
        frame_count=_integer(_value(attributes, "framecount", "frames", "frame", "nframes"), "frame_count"),
        codec=_value(attributes, "codec", "fourcc", "fourcccode", "fourcc_original"),
        dropped_frames=dropped_frames,
        countrycode=countrycode,
        lid=lid,
        sid=sid,
        cam=cam,
        lens=lens,
        station=station,
        camera=camera,
        camera_night=camera_night,
    )


def _avi_info(path: Path) -> MediaInfo:
    cv2, _ = _vision_deps()
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise BundlePreflightError(f"cannot open AVI: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        codec_value = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((codec_value >> shift) & 0xFF) for shift in range(0, 32, 8)).strip("\0 ") or None
        width = height = 0
        decoded = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_height, frame_width = frame.shape[:2]
            if decoded == 0:
                width, height = frame_width, frame_height
            elif (frame_width, frame_height) != (width, height):
                raise BundlePreflightError(f"AVI frame geometry changes within clip: {path}")
            decoded += 1
        if decoded < 2:
            raise BundlePreflightError(f"AVI has fewer than two decodable frames: {path}")
        return MediaInfo(
            width=width,
            height=height,
            fps=fps if math.isfinite(fps) and fps > 0 else None,
            frame_count=decoded,
            codec=codec,
        )
    finally:
        capture.release()


def _stack_info(path: Path) -> MediaInfo:
    cv2, _ = _vision_deps()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        raise BundlePreflightError(f"cannot read stack image: {path}")
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise BundlePreflightError(f"stack image has invalid geometry: {path}")
    return MediaInfo(width=width, height=height, fps=None, frame_count=None, codec=None)


def _select_valid_stack(
    bundle: EventBundle, expected_dimensions: tuple[int, int]
) -> tuple[EventBundle, MediaInfo]:
    """Decode P.bmp then P.jpg, retaining the first valid classifier image."""

    failures: list[str] = []
    for path in stack_image_candidates(bundle):
        try:
            info = _stack_info(path)
            if (info.width, info.height) != expected_dimensions:
                failures.append(
                    f"stack image geometry {(info.width, info.height)} does not match "
                    f"XML {expected_dimensions}: {path}"
                )
                continue
            return replace(bundle, stack_image=path), info
        except BundlePreflightError as exc:
            failures.append(str(exc))
    detail = "; ".join(failures) if failures else "no P.bmp or P.jpg exists"
    raise BundlePreflightError(f"training bundle has no readable P stack: {detail}")


def preflight_bundle(bundle: EventBundle) -> PreflightBundle:
    """Validate all required training artifacts without creating source files."""

    missing = [
        name
        for name, path in (("AVI", bundle.avi), ("P stack", bundle.stack_image), ("XML", bundle.xml))
        if path is None
    ]
    if missing:
        raise BundlePreflightError("training bundle is missing required artifact(s): " + ", ".join(missing))
    assert bundle.avi is not None and bundle.xml is not None
    try:
        metadata = parse_ufocapture_record(bundle.xml)
    except UfoCaptureMetadataError as exc:
        raise BundlePreflightError(str(exc)) from exc
    try:
        avi = _avi_info(bundle.avi)
        bundle, stack = _select_valid_stack(
            bundle, (metadata.width, metadata.height)
        )
    except (InvalidMediaError, OSError) as exc:
        raise BundlePreflightError(str(exc)) from exc
    expected = (metadata.width, metadata.height)
    if (avi.width, avi.height) != expected:
        raise BundlePreflightError(f"XML geometry {expected} does not match AVI {(avi.width, avi.height)}")
    if (stack.width, stack.height) != expected:
        raise BundlePreflightError(f"XML geometry {expected} does not match stack {(stack.width, stack.height)}")
    warnings: list[str] = []
    if metadata.frame_count is not None and avi.frame_count is not None and metadata.frame_count != avi.frame_count:
        warnings.append(f"frame_count differs: XML={metadata.frame_count}, AVI={avi.frame_count}")
    if metadata.fps is not None and avi.fps is not None and not math.isclose(metadata.fps, avi.fps, rel_tol=0.0, abs_tol=0.01):
        warnings.append(f"fps differs: XML={metadata.fps}, AVI={avi.fps}")
    if metadata.codec and avi.codec and metadata.codec.casefold() != avi.codec.casefold():
        warnings.append(f"codec differs: XML={metadata.codec}, AVI={avi.codec}")
    if metadata.dropped_frames not in (None, 0):
        warnings.append(f"XML records {metadata.dropped_frames} dropped frame(s)")
    return PreflightBundle(bundle=bundle, metadata=metadata, avi=avi, stack=stack, warnings=tuple(warnings))


def preflight_records(records: Iterable[Any]) -> list[PreflightBundle]:
    """Preflight every record, preserving row-level context in failures."""

    checked: list[PreflightBundle] = []
    for index, record in enumerate(records, start=2):
        bundle = getattr(record, "bundle", None)
        if not isinstance(bundle, EventBundle):
            try:
                bundle = EventBundle(
                    clip_base=Path(str(record.clip_base)),
                    avi=Path(str(record.avi)),
                    stack_image=Path(str(record.stack_image)),
                    star_mask=Path(str(record.star_mask)) if record.star_mask else None,
                    xml=Path(str(record.xml)),
                )
            except (AttributeError, TypeError) as exc:
                raise TypeError("preflight_records requires manifest-like records") from exc
        try:
            checked.append(preflight_bundle(bundle))
        except BundlePreflightError as exc:
            raise BundlePreflightError(f"row {index}: {exc}") from exc
    return checked


def validate_xml_provenance(path: str | Path | None) -> str:
    """Return a non-fatal runtime provenance state without exposing XML values.

    Runtime may score a bundle with missing or malformed XML; training may not.
    This narrow helper intentionally reports no metadata to avoid accidental
    model-feature wiring.
    """

    if path is None:
        return "absent"
    try:
        parse_ufocapture_record(path)
    except UfoCaptureMetadataError:
        return "malformed"
    return "valid"
