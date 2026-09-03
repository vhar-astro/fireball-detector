"""Candidate extraction and temporal measurements for UFOCapture bundles.

OpenCV and NumPy are imported lazily so queue and configuration commands remain
usable even on a machine where the model runtime has not yet been installed.
AVI files are always streamed; this module never loads a complete clip in RAM.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .bundles import EventBundle, stack_image_candidates


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

    @property
    def region(self) -> CandidateRegion:
        return self.regions[0]


@dataclass(frozen=True)
class StackImage:
    """Pixels used to crop classifier ROIs and their auditable origin."""

    image: Any
    selected_path: Path | None
    source: str

    @property
    def is_avi_composite(self) -> bool:
        return self.selected_path is None


def _regions_from_signal(
    signal: Any,
    *,
    source: str,
    threshold: int,
    min_pixels: int,
    padding_fraction: float,
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
        candidates.append(
            CandidateRegion(
                x=left,
                y=top,
                width=right - left,
                height=bottom - top,
                changed_pixels=area,
                score=float(area * mean_signal),
                source=source,
            )
        )
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates


def candidate_from_avi(
    path: str | Path,
    *,
    difference_threshold: int = 18,
    min_pixels: int = 12,
    padding_fraction: float = 0.15,
) -> CandidateRegion:
    """Return the strongest candidate from sequential AVI differences only."""

    return candidate_regions_from_avi(
        path,
        max_candidates=1,
        difference_threshold=difference_threshold,
        min_pixels=min_pixels,
        padding_fraction=padding_fraction,
    )[0]


def candidate_regions_from_avi(
    path: str | Path,
    *,
    max_candidates: int = 3,
    difference_threshold: int = 18,
    min_pixels: int = 12,
    padding_fraction: float = 0.15,
) -> tuple[CandidateRegion, ...]:
    """Return up to three candidates from sequential AVI differences.

    This is intentionally the only candidate-source implementation in v2.
    The M.bmp star mask is not an input and is not even opened here.
    """
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
        source="avi_sequential_difference",
        threshold=difference_threshold,
        min_pixels=min_pixels,
        padding_fraction=padding_fraction,
    )
    if not regions:
        raise InvalidMediaError(f"AVI contains no valid changed region: {path}")
    return tuple(regions[:max_candidates])


def extract_candidate(bundle: EventBundle) -> CandidateExtraction:
    if bundle.avi is None:
        raise InvalidMediaError("AVI is required for v2 candidate extraction")
    return CandidateExtraction(candidate_regions_from_avi(bundle.avi))


def measure_temporal_features(
    avi_path: str | Path,
    region: CandidateRegion,
    *,
    activity_threshold: float = 12.0,
    saturation_threshold: int = 250,
) -> TemporalFeatures:
    """Measure temporal features from sequential AVI frames only.

    Candidate extraction uses frame-to-frame differences, while temporal
    brightness uses a slowly updated background learned solely from AVI frames.
    A star-mask or XML field must never influence either calculation.
    """

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
            if running_background is None:
                running_background = gray.copy()
                profiles.append(0.0)
                saturation_fractions.append(
                    float((roi >= saturation_threshold).any(axis=2).mean())
                )
                continue
            difference = np.abs(gray - running_background)
            excess = np.maximum(gray - running_background, 0.0)
            stable = difference < activity_threshold
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


def _avi_dimensions(path: str | Path) -> tuple[int, int]:
    cv2, _ = _vision_deps()
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise InvalidMediaError(f"cannot open AVI: {path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise InvalidMediaError(f"AVI has invalid geometry: {path}")
    return width, height


def _maximum_composite(avi_path: str | Path) -> Any:
    cv2, _ = _vision_deps()
    capture = cv2.VideoCapture(str(avi_path))
    if not capture.isOpened():
        raise InvalidMediaError(f"cannot open AVI: {avi_path}")
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
        raise InvalidMediaError(f"AVI has no decodable frame: {avi_path}")
    return maximum


def read_stack_or_avi_composite(bundle: EventBundle) -> StackImage:
    """Select a valid P stack, otherwise reconstruct one from the AVI.

    Runtime deliberately tolerates a missing, corrupt, or geometry-mismatched
    P stack.  Training preflight is stricter and rejects that bundle.  M.bmp is
    not considered at all.
    """

    if bundle.avi is None:
        raise InvalidMediaError("AVI is required to select classifier pixels")
    cv2, _ = _vision_deps()
    expected_dimensions = _avi_dimensions(bundle.avi)
    for path in stack_image_candidates(bundle):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None and (image.shape[1], image.shape[0]) == expected_dimensions:
            return StackImage(image=image, selected_path=path, source="p_stack")
    return StackImage(
        image=_maximum_composite(bundle.avi),
        selected_path=None,
        source="avi_maximum_composite",
    )


def read_peak_or_avi_frame(bundle: EventBundle) -> Any:
    """Compatibility wrapper for v1 callers; uses only v2 stack semantics."""

    return read_stack_or_avi_composite(bundle).image


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
