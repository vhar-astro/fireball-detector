"""Fail-closed ONNX inference and score calibration."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .vision import CandidateRegion, TemporalFeatures


class ModelPackageError(RuntimeError):
    pass


CALIBRATION_FEATURES = frozenset(
    {
        "roi_logit",
        "log_changed_pixels",
        "roi_area_fraction",
        "roi_aspect_ratio",
        "map_background_brightness",
        "map_brightness_above_background",
        "frame_count",
        "fps",
        "active_frame_count",
        "duration_seconds",
        "motion_pixels",
        "linearity",
        "saturated_area_fraction",
        "brightness_above_background",
        "peak_brightness_above_background",
        "halo_growth",
        "temporal_peak_fraction",
    }
)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelManifest:
    manifest_path: Path
    model_path: Path
    model_version: str
    manifest_sha256: str
    model_sha256: str
    input_name: str
    output_name: str | None
    image_size: int
    feature_order: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    possible_threshold: float
    probable_threshold: float
    quantization: str
    quantization_evidence: dict[str, Any] | None

    @classmethod
    def load(cls, path: str | Path) -> "ModelManifest":
        manifest_path = Path(path).expanduser().resolve(strict=True)
        with manifest_path.open("r", encoding="utf-8") as source:
            document = json.load(source)
        if not isinstance(document, dict) or document.get("schema_version") != 1:
            raise ModelPackageError("unsupported or missing model manifest schema_version")
        try:
            model = document["model"]
            preprocessing = document["preprocessing"]
            calibration = document["calibration"]
            thresholds = document["thresholds"]
            feature_order = tuple(calibration["feature_order"])
            coefficients = tuple(float(value) for value in calibration["coefficients"])
            calibration_payload = {
                key: value for key, value in calibration.items() if key != "sha256"
            }
            calibration_hash = hashlib.sha256(
                json.dumps(
                    calibration_payload, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            if calibration_hash != calibration.get("sha256"):
                raise ModelPackageError("calibration hash mismatch")
            result = cls(
                manifest_path=manifest_path,
                model_path=(manifest_path.parent / model["file"]).resolve(strict=True),
                model_version=str(document["model_version"]),
                manifest_sha256=sha256_file(manifest_path),
                model_sha256=str(model["sha256"]).lower(),
                input_name=str(model.get("input_name", "image")),
                output_name=str(model["output_name"]) if model.get("output_name") else None,
                image_size=int(preprocessing["image_size"]),
                feature_order=feature_order,
                coefficients=coefficients,
                intercept=float(calibration["intercept"]),
                possible_threshold=float(thresholds["possible_fireball"]),
                probable_threshold=float(thresholds["probable_fireball"]),
                quantization=str(model["quantization"]),
                quantization_evidence=(
                    dict(document["quantization_evidence"])
                    if document.get("quantization_evidence") is not None
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelPackageError(f"invalid model manifest: {exc}") from exc
        if len(result.feature_order) != len(result.coefficients):
            raise ModelPackageError("calibration feature_order and coefficients have different lengths")
        if not result.model_version:
            raise ModelPackageError("model_version must not be empty")
        if not all(isinstance(name, str) for name in result.feature_order):
            raise ModelPackageError("calibration feature names must be strings")
        if "roi_logit" not in result.feature_order:
            raise ModelPackageError("calibration must include roi_logit")
        if len(result.feature_order) < 2:
            raise ModelPackageError(
                "calibration must combine roi_logit with temporal or geometric features"
            )
        unknown_features = set(result.feature_order) - CALIBRATION_FEATURES
        if unknown_features:
            raise ModelPackageError(
                "unknown calibration features: " + ", ".join(sorted(unknown_features))
            )
        numeric_values = (*result.coefficients, result.intercept)
        if not all(math.isfinite(value) for value in numeric_values):
            raise ModelPackageError("calibration coefficients must be finite")
        if len(set(result.feature_order)) != len(result.feature_order):
            raise ModelPackageError("calibration feature_order contains duplicates")
        if result.image_size < 32:
            raise ModelPackageError("preprocessing.image_size is invalid")
        expected_preprocessing = {
            "color_order": "RGB",
            "resize": "aspect_preserving_letterbox",
            "padding": "roi_median",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
        for key, expected in expected_preprocessing.items():
            if preprocessing.get(key) != expected:
                raise ModelPackageError(f"unsupported preprocessing value: {key}")
        if document.get("candidate_extractor") != "change-map-red-v1-with-avi-fallback":
            raise ModelPackageError("candidate extractor version mismatch")
        if result.quantization not in {"fp32", "qdq_int8"}:
            raise ModelPackageError("unsupported quantization mode")
        if result.quantization == "qdq_int8":
            evidence = result.quantization_evidence
            try:
                if evidence is None:
                    raise KeyError("quantization_evidence")
                fp32_recall = float(evidence["fp32_recall"])
                int8_recall = float(evidence["int8_recall"])
                fp32_p95 = float(evidence["fp32_p95_ms"])
                int8_p95 = float(evidence["int8_p95_ms"])
                report_hash = str(evidence["locked_report_sha256"])
                report_file = str(evidence["locked_report_file"])
                target_cpu = str(evidence["target_cpu"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ModelPackageError("invalid INT8 quantization gate evidence") from exc
            if not all(
                math.isfinite(value)
                for value in (fp32_recall, int8_recall, fp32_p95, int8_p95)
            ) or not (
                0 <= fp32_recall <= 1
                and 0 <= int8_recall <= 1
                and fp32_p95 > 0
                and int8_p95 > 0
            ):
                raise ModelPackageError(
                    "INT8 gate metrics must be finite probabilities and positive latencies"
                )
            try:
                valid_report_hash = len(report_hash) == 64 and bool(bytes.fromhex(report_hash))
            except ValueError:
                valid_report_hash = False
            try:
                report_path = (manifest_path.parent / report_file).resolve(strict=True)
                report_path.relative_to(manifest_path.parent)
                report_matches = sha256_file(report_path) == report_hash
                with report_path.open("r", encoding="utf-8") as source:
                    report = json.load(source)
                report_metrics_match = (
                    report.get("schema_version") == 1
                    and report.get("target_cpu") == target_cpu
                    and report.get("fp32", {}).get("recall") == fp32_recall
                    and report.get("fp32", {}).get("p95_ms") == fp32_p95
                    and report.get("int8", {}).get("recall") == int8_recall
                    and report.get("int8", {}).get("p95_ms") == int8_p95
                    and report.get("ship_int8") is True
                )
            except (AttributeError, OSError, ValueError, json.JSONDecodeError):
                report_matches = False
                report_metrics_match = False
            recall_pass = fp32_recall - int8_recall <= 0.01 + 1e-12
            latency_pass = (fp32_p95 - int8_p95) / fp32_p95 >= 0.15 - 1e-12
            if (
                not valid_report_hash
                or not report_matches
                or not report_metrics_match
                or "i7-4500u" not in target_cpu.casefold()
                or evidence.get("ship_int8") is not True
                or not recall_pass
                or not latency_pass
            ):
                raise ModelPackageError(
                    "INT8 model has not passed the target-hardware release gate"
                )
        if not (0.0 <= result.possible_threshold <= result.probable_threshold <= 1.0):
            raise ModelPackageError("thresholds must satisfy 0 <= possible <= probable <= 1")
        actual_hash = sha256_file(result.model_path)
        if actual_hash.lower() != result.model_sha256:
            raise ModelPackageError(
                f"model hash mismatch: expected {result.model_sha256}, got {actual_hash}"
            )
        try:
            result.model_path.relative_to(manifest_path.parent)
        except ValueError as exc:
            raise ModelPackageError("model file must remain in the manifest package") from exc
        return result


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp = math.exp(-value)
        return 1.0 / (1.0 + exp)
    exp = math.exp(value)
    return exp / (1.0 + exp)


def calibration_features(
    roi_logit: float,
    region: CandidateRegion,
    temporal: TemporalFeatures,
    image_width: int,
    image_height: int,
) -> dict[str, float]:
    """Build the fixed, named feature vector bound by the model manifest."""

    image_area = max(image_width * image_height, 1)
    box_area = region.width * region.height
    return {
        "roi_logit": float(roi_logit),
        "log_changed_pixels": math.log1p(region.changed_pixels),
        "roi_area_fraction": float(box_area / image_area),
        "roi_aspect_ratio": float(region.width / max(region.height, 1)),
        "map_background_brightness": float(region.map_background_brightness),
        "map_brightness_above_background": float(region.map_brightness_above_background),
        **{key: float(value) for key, value in temporal.as_dict().items()},
    }


class OnnxCandidateModel:
    """CPU-only ONNX model with a manifest-bound logistic calibrator."""

    def __init__(self, manifest: ModelManifest, max_threads: int = 2) -> None:
        if not 1 <= max_threads <= 2:
            raise ValueError("max_threads must be one or two")
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise ModelPackageError("onnxruntime is required for edge inference") from exc
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = max_threads
        options.inter_op_num_threads = 1
        self.manifest = manifest
        self.session = ort.InferenceSession(
            str(manifest.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        input_names = {item.name for item in self.session.get_inputs()}
        if manifest.input_name not in input_names:
            raise ModelPackageError(
                f"manifest input {manifest.input_name!r} not found in ONNX graph"
            )
        model_input = next(
            item for item in self.session.get_inputs() if item.name == manifest.input_name
        )
        if model_input.type != "tensor(float)" or model_input.shape != [
            1,
            3,
            manifest.image_size,
            manifest.image_size,
        ]:
            raise ModelPackageError(
                "ONNX input must be fixed float32 NCHW [1, 3, image_size, image_size]"
            )
        if manifest.output_name is not None:
            output_names = {item.name for item in self.session.get_outputs()}
            if manifest.output_name not in output_names:
                raise ModelPackageError(
                    f"manifest output {manifest.output_name!r} not found in ONNX graph"
                )

    @staticmethod
    def _roi_logit(output: Any) -> float:
        try:
            flattened = output.reshape(-1)
            values = [float(value) for value in flattened]
        except (AttributeError, TypeError, ValueError) as exc:
            raise ModelPackageError("ONNX output is not a numeric tensor") from exc
        if not all(math.isfinite(value) for value in values):
            raise ModelPackageError("ONNX output contains a non-finite value")
        if len(values) == 1:
            return values[0]
        if len(values) == 2:
            return values[1] - values[0]
        raise ModelPackageError("ONNX output must contain one logit or two class logits")

    def score(
        self,
        input_tensor: Any,
        region: CandidateRegion,
        temporal: TemporalFeatures,
        image_width: int,
        image_height: int,
    ) -> tuple[float, float, dict[str, float]]:
        output_names = [self.manifest.output_name] if self.manifest.output_name else None
        outputs = self.session.run(output_names, {self.manifest.input_name: input_tensor})
        roi_logit = self._roi_logit(outputs[0])
        available = calibration_features(
            roi_logit, region, temporal, image_width, image_height
        )
        missing = [name for name in self.manifest.feature_order if name not in available]
        if missing:
            raise ModelPackageError(f"unknown calibration features: {', '.join(missing)}")
        vector = [available[name] for name in self.manifest.feature_order]
        if not all(math.isfinite(value) for value in vector):
            raise ModelPackageError("calibration input contains a non-finite value")
        calibrated_logit = self.manifest.intercept + sum(
            coefficient * value
            for coefficient, value in zip(self.manifest.coefficients, vector, strict=True)
        )
        if not math.isfinite(calibrated_logit):
            raise ModelPackageError("calibrated logit is not finite")
        calibrated_score = _sigmoid(calibrated_logit)
        if not math.isfinite(calibrated_score):
            raise ModelPackageError("calibrated score is not finite")
        return calibrated_score, roi_logit, {
            name: value for name, value in zip(self.manifest.feature_order, vector, strict=True)
        }

    def decision(self, score: float) -> str:
        if not math.isfinite(score):
            raise ModelPackageError("decision score is not finite")
        if score >= self.manifest.probable_threshold:
            return "probable_fireball"
        if score >= self.manifest.possible_threshold:
            return "possible_fireball"
        return "no_alert"


def load_model(manifest_path: str | Path, max_threads: int = 2) -> OnnxCandidateModel:
    return OnnxCandidateModel(ModelManifest.load(manifest_path), max_threads=max_threads)
