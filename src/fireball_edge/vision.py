"""Candidate extraction and temporal measurements for UFOCapture bundles.

OpenCV and NumPy are imported lazily so queue and configuration commands remain
usable even on a machine where the model runtime has not yet been installed.
AVI files are always streamed; this module never loads a complete clip in RAM.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .bundles import EventBundle


class VisionDependencyError(RuntimeError):
    pass


class InvalidMediaError(RuntimeError):
    pass


def _vision_deps() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on deployment extras
        raise VisionDependencyError(
            "edge vision requires numpy and opencv-python-headless"
        ) from exc
    return cv2, np


@dataclass(frozen=True)
class CandidateRegion:
    x: int
    y: int
    width: int
    height: int
    changed_pixels: int
    score: float
    source: str
    map_background_brightness: float = 0.0
    map_brightness_above_background: float = 0.0

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    def as_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalFeatures:
    frame_count: int
    fps: float
    active_frame_count: int
    duration_seconds: float
    motion_pixels: float
    linearity: float
    saturated_area_fraction: float
    brightness_above_background: float
    peak_brightness_above_background: float
    halo_growth: float
    temporal_peak_fraction: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateExtraction:
    regions: tuple[CandidateRegion, ...]
    used_change_map: bool
    fallback_reason: str | None = None

    @property
    def region(self) -> CandidateRegion:
        return self.regions[0]


def _regions_from_signal(
    signal: Any,
    *,
    source: str,
    threshold: int,
    min_pixels: int,
    padding_fraction: float,
    background: Any | None = None,
) -> list[CandidateRegion]:
    cv2, np = _vision_deps()
    if signal.ndim != 2 or signal.size == 0:
        raise InvalidMediaError("candidate signal must be a non-empty 2D image")
    mask = (signal >= threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    height, width = signal.shape
    candidates: list[CandidateRegion] = []
    for index in range(1, count):
        x, y, box_width, box_height, area = (int(v) for v in stats[index])
        if area < min_pixels:
            continue
        padding = max(8, int(max(box_width, box_height) * padding_fraction))
        left = max(0, x - padding)
        top = max(0, y - padding)
        right = min(width, x + box_width + padding)
        bottom = min(height, y + box_height + padding)
        local = signal[y : y + box_height, x : x + box_width]
        mean_signal = float(local[local >= threshold].mean()) if area else 0.0
        local_background = (
            background[y : y + box_height, x : x + box_width]
            if background is not None
            else None
        )
        background_brightness = (
            float(local_background.mean()) if local_background is not None else 0.0
        )
        above_background = (
            float(np.maximum(local.astype(np.float32) - local_background, 0.0).mean())
            if local_background is not None
            else 0.0
        )
        candidates.append(
            CandidateRegion(
                x=left,
                y=top,
                width=right - left,
                height=bottom - top,
                changed_pixels=area,
                score=float(area * mean_signal),
                source=source,
                map_background_brightness=background_brightness,
                map_brightness_above_background=above_background,
            )
        )
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates


def candidate_from_change_map(
    path: str | Path,
    *,
    absolute_red_threshold: int = 24,
    min_pixels: int = 12,
    padding_fraction: float = 0.15,
) -> CandidateRegion:
    """Extract the strongest ROI from the red changed-pixel map channel.

    UFOCapture already defines nonzero red as pixels that passed its change
    detector, so red alone drives candidate membership. Green is retained as
    long-term background evidence for scoring. Blue is a capture mask and is
    deliberately not interpreted as brightness.
    """

    cv2, np = _vision_deps()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        raise InvalidMediaError(f"invalid change map: {path}")
    red = image[:, :, 2].astype(np.int16)
    # UFOCapture documents G as the long-term averaged brightness. B is an
    # area/scintillation mask, so it must not be treated as background light.
    background = image[:, :, 1].astype(np.int16)
    detected = red.astype(np.uint8)
    detected[red < absolute_red_threshold] = 0
    regions = _regions_from_signal(
        detected,
        source="change_map_red_channel",
        threshold=absolute_red_threshold,
        min_pixels=min_pixels,
        padding_fraction=padding_fraction,
        background=background,
    )
    if not regions:
        raise InvalidMediaError(f"change map contains no valid candidate: {path}")
    return regions[0]


def candidate_regions_from_change_map(
    path: str | Path,
    *,
    max_candidates: int = 3,
    expected_dimensions: tuple[int, int] | None = None,
) -> tuple[CandidateRegion, ...]:
    cv2, np = _vision_deps()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        raise InvalidMediaError(f"invalid change map: {path}")
    if expected_dimensions is not None and (image.shape[1], image.shape[0]) != expected_dimensions:
        raise InvalidMediaError(
            f"change map dimensions {(image.shape[1], image.shape[0])} do not match AVI "
            f"{expected_dimensions}"
        )
    red = image[:, :, 2].astype(np.int16)
    background = image[:, :, 1].astype(np.int16)
    detected = red.astype(np.uint8)
    detected[red < 24] = 0
    regions = _regions_from_signal(
        detected,
        source="change_map_red_channel",
        threshold=24,
        min_pixels=12,
        padding_fraction=0.15,
        background=background,
    )
    if not regions:
        raise InvalidMediaError(f"change map contains no valid candidate: {path}")
    return tuple(regions[:max_candidates])


def candidate_from_avi(
    path: str | Path,
    *,
    difference_threshold: int = 18,
    min_pixels: int = 12,
    padding_fraction: float = 0.15,
) -> CandidateRegion:
    """Fall back to a max sequential-frame-difference signal."""

    cv2, np = _vision_deps()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise InvalidMediaError(f"cannot open AVI: {path}")
    previous = None
    maximum = None
    decoded = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if previous is not None:
                difference = cv2.absdiff(gray, previous)
                maximum = difference if maximum is None else np.maximum(maximum, difference)
            previous = gray
    finally:
        capture.release()
    if decoded < 2 or maximum is None:
        raise InvalidMediaError(f"AVI has fewer than two decodable frames: {path}")
    regions = _regions_from_signal(
        maximum,
        source="avi_frame_difference",
        threshold=difference_threshold,
        min_pixels=min_pixels,
        padding_fraction=padding_fraction,
    )
    if not regions:
        raise InvalidMediaError(f"AVI contains no valid changed region: {path}")
    return regions[0]


def candidate_regions_from_avi(
    path: str | Path, *, max_candidates: int = 3
) -> tuple[CandidateRegion, ...]:
    """Return retained frame-difference regions using one streamed pass."""
    cv2, np = _vision_deps()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise InvalidMediaError(f"cannot open AVI: {path}")
    previous = maximum = None
    decoded = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if previous is not None:
                difference = cv2.absdiff(gray, previous)
                maximum = difference if maximum is None else np.maximum(maximum, difference)
            previous = gray
    finally:
        capture.release()
    if decoded < 2 or maximum is None:
        raise InvalidMediaError(f"AVI has fewer than two decodable frames: {path}")
    regions = _regions_from_signal(
        maximum,
        source="avi_frame_difference",
        threshold=18,
        min_pixels=12,
        padding_fraction=0.15,
    )
    if not regions:
        raise InvalidMediaError(f"AVI contains no valid changed region: {path}")
    return tuple(regions[:max_candidates])


def extract_candidate(bundle: EventBundle) -> CandidateExtraction:
    """Prefer the change map and explicitly record any AVI fallback."""

    fallback_reason: str | None = None
    expected_dimensions = None
    if bundle.avi is not None:
        cv2, _ = _vision_deps()
        capture = cv2.VideoCapture(str(bundle.avi))
        try:
            if capture.isOpened():
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if width > 0 and height > 0:
                    expected_dimensions = (width, height)
        finally:
            capture.release()
    if bundle.change_map is not None:
        try:
            return CandidateExtraction(
                candidate_regions_from_change_map(
                    bundle.change_map, expected_dimensions=expected_dimensions
                ),
                True,
            )
        except InvalidMediaError as exc:
            fallback_reason = str(exc)
    else:
        fallback_reason = "change map is missing"
    if bundle.avi is None:
        raise InvalidMediaError(f"{fallback_reason}; AVI fallback is missing")
    return CandidateExtraction(
        candidate_regions_from_avi(bundle.avi),
        False,
        fallback_reason=fallback_reason,
    )


def measure_temporal_features(
    avi_path: str | Path,
    region: CandidateRegion,
    *,
    activity_threshold: float = 12.0,
    saturation_threshold: int = 250,
    map_background: Any | None = None,
) -> TemporalFeatures:
    """Stream the AVI once and measure morphology inside the candidate ROI."""

    cv2, np = _vision_deps()
    capture = cv2.VideoCapture(str(avi_path))
    if not capture.isOpened():
        raise InvalidMediaError(f"cannot open AVI: {avi_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    profiles: list[float] = []
    active_indices: list[int] = []
    centroids: list[tuple[float, float]] = []
    saturation_fractions: list[float] = []
    active_halo_areas: list[int] = []
    frame_count = 0
    running_background = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_count += 1
            height, width = frame.shape[:2]
            x1, y1 = max(0, region.x), max(0, region.y)
            x2, y2 = min(width, region.x2), min(height, region.y2)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if map_background is not None:
                reference = map_background[y1:y2, x1:x2].astype(np.float32)
                if reference.shape != gray.shape:
                    raise InvalidMediaError("map background dimensions do not match the AVI")
            else:
                if running_background is None:
                    running_background = gray.copy()
                reference = running_background
            excess = np.maximum(gray - reference, 0.0)
            if map_background is None:
                stable = excess < activity_threshold
                running_background[stable] = (
                    0.98 * running_background[stable] + 0.02 * gray[stable]
                )
            bright_mask = excess >= activity_threshold
            active_pixels = int(bright_mask.sum())
            mean_excess = float(excess[bright_mask].mean()) if active_pixels else 0.0
            profiles.append(mean_excess)
            saturation_fractions.append(float((roi >= saturation_threshold).any(axis=2).mean()))
            halo = int((excess >= activity_threshold / 2.0).sum())
            if active_pixels:
                active_indices.append(frame_count - 1)
                active_halo_areas.append(halo)
                ys, xs = np.nonzero(bright_mask)
                weights = excess[bright_mask]
                total = float(weights.sum())
                if total > 0:
                    centroids.append(
                        (float((xs * weights).sum() / total), float((ys * weights).sum() / total))
                    )
    finally:
        capture.release()
    if frame_count == 0:
        raise InvalidMediaError(f"AVI has no decodable frames: {avi_path}")

    duration = (
        (active_indices[-1] - active_indices[0] + 1) / fps if active_indices else 0.0
    )
    motion = 0.0
    linearity = 0.0
    if len(centroids) >= 2:
        points = np.asarray(centroids, dtype=np.float32)
        motion = float(np.linalg.norm(points[-1] - points[0]))
        centered = points - points.mean(axis=0)
        covariance = np.cov(centered.T)
        eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0.0))
        if eigenvalues[-1] > 1e-9:
            linearity = float(1.0 - eigenvalues[0] / eigenvalues[-1])

    halo_growth = 0.0
    if active_halo_areas and active_halo_areas[0] > 0:
        halo_growth = max(0.0, max(active_halo_areas) / active_halo_areas[0] - 1.0)

    peak = max(profiles, default=0.0)
    peak_index = profiles.index(peak) if profiles and peak > 0 else 0
    return TemporalFeatures(
        frame_count=frame_count,
        fps=fps,
        active_frame_count=len(active_indices),
        duration_seconds=float(duration),
        motion_pixels=motion,
        linearity=max(0.0, min(1.0, linearity)),
        saturated_area_fraction=max(saturation_fractions, default=0.0),
        brightness_above_background=float(sum(profiles) / max(len(profiles), 1)),
        peak_brightness_above_background=float(peak),
        halo_growth=halo_growth,
        temporal_peak_fraction=float(peak_index / max(len(profiles) - 1, 1)),
    )


def load_map_background(
    path: str | Path | None, expected_dimensions: tuple[int, int]
) -> Any | None:
    """Return UFOCapture's green long-term-average channel when valid."""

    if path is None:
        return None
    cv2, _ = _vision_deps()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or (image.shape[1], image.shape[0]) != expected_dimensions:
        return None
    return image[:, :, 1]


def read_peak_or_avi_frame(bundle: EventBundle) -> Any:
    """Load the peak-hold image, or build a streamed maximum composite."""

    cv2, _ = _vision_deps()
    expected_dimensions = None
    if bundle.avi is not None:
        metadata = cv2.VideoCapture(str(bundle.avi))
        try:
            if metadata.isOpened():
                width = int(metadata.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(metadata.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if width > 0 and height > 0:
                    expected_dimensions = (width, height)
        finally:
            metadata.release()
    if bundle.peak is not None:
        image = cv2.imread(str(bundle.peak), cv2.IMREAD_COLOR)
        if image is not None and (
            expected_dimensions is None
            or (image.shape[1], image.shape[0]) == expected_dimensions
        ):
            return image
    if bundle.avi is None:
        raise InvalidMediaError("neither a valid peak-hold image nor AVI is available")
    capture = cv2.VideoCapture(str(bundle.avi))
    maximum = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            maximum = frame.copy() if maximum is None else cv2.max(maximum, frame)
    finally:
        capture.release()
    if maximum is None:
        raise InvalidMediaError(f"AVI has no decodable frame: {bundle.avi}")
    return maximum


def prepare_roi(image: Any, region: CandidateRegion, size: int = 224) -> Any:
    """Crop, aspect-fit, pad, normalize, and return a NCHW float32 tensor."""

    cv2, np = _vision_deps()
    height, width = image.shape[:2]
    roi = image[max(0, region.y) : min(height, region.y2), max(0, region.x) : min(width, region.x2)]
    if roi.size == 0:
        raise InvalidMediaError("candidate ROI is outside the source image")
    scale = min(size / roi.shape[1], size / roi.shape[0])
    resized_width = max(1, round(roi.shape[1] * scale))
    resized_height = max(1, round(roi.shape[0] * scale))
    resized = cv2.resize(roi, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    fill = tuple(int(value) for value in np.median(roi.reshape(-1, 3), axis=0))
    canvas = np.full((size, size, 3), fill, dtype=np.uint8)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    normalized = (rgb - mean) / std
    return np.transpose(normalized, (2, 0, 1))[None, ...].astype(np.float32)


def write_annotated_image(
    image: Any,
    region: CandidateRegion,
    label: str,
    score: float,
    destination: str | Path,
) -> Path:
    """Write only to a caller-provided external destination."""

    cv2, _ = _vision_deps()
    output = image.copy()
    colors = {
        "no_alert": (0, 180, 0),
        "possible_fireball": (0, 165, 255),
        "probable_fireball": (0, 0, 255),
    }
    color = colors.get(label, (255, 255, 255))
    cv2.rectangle(output, (region.x, region.y), (region.x2, region.y2), color, 3)
    cv2.putText(
        output,
        f"{label} {score:.3f}",
        (region.x, max(24, region.y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), output):
        raise OSError(f"failed to write annotated image: {destination}")
    return destination
