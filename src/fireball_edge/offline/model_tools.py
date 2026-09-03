"""Training-only MobileNet, calibration, ONNX export, and INT8 comparison."""

from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ..artifacts import write_json_atomic
from ..contracts import CANDIDATE_EXTRACTOR, SCHEMA_VERSION
from ..inference import CALIBRATION_FEATURES, sha256_file


def create_mobilenet_v3_small_binary():
    """Create the ImageNet-pretrained ROI classifier; never used by production."""

    try:
        from torch import nn
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    except ImportError as exc:  # pragma: no cover - training environment only
        raise RuntimeError("training requires torch and torchvision") from exc
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    final = model.classifier[-1]
    model.classifier[-1] = nn.Linear(final.in_features, 1)
    return model


def export_onnx(
    checkpoint: str | Path,
    destination: str | Path,
    image_size: int = 224,
    *,
    model_factory: Callable[[], Any] | None = None,
) -> Path:
    """Export a fixed batch-1 graph so production preprocessing stays explicit."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - training environment only
        raise RuntimeError("ONNX export requires torch") from exc
    model = (model_factory or create_mobilenet_v3_small_binary)()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros((1, 3, image_size, image_size), dtype=torch.float32)
    torch.onnx.export(
        model,
        (example,),
        str(destination),
        input_names=["image"],
        output_names=["logit"],
        dynamo=True,
        opset_version=18,
    )
    return destination


class _NpyCalibrationReader:
    def __init__(self, paths: Sequence[str | Path], input_name: str = "image") -> None:
        self.paths = [Path(path) for path in paths]
        self.input_name = input_name
        self._iterator = iter(())
        self.rewind()

    def get_next(self):
        try:
            path = next(self._iterator)
        except StopIteration:
            return None
        import numpy as np

        array = np.load(path).astype(np.float32)
        if array.ndim == 3:
            array = array[None, ...]
        return {self.input_name: array}

    def rewind(self) -> None:
        self._iterator = iter(self.paths)


def quantize_static_qdq(
    fp32_model: str | Path,
    int8_model: str | Path,
    calibration_arrays: Sequence[str | Path],
) -> Path:
    """Create the non-VNNI candidate; this does not declare it shippable."""

    if not calibration_arrays:
        raise ValueError("static quantization requires calibration arrays")
    try:
        from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
        from onnxruntime.quantization.shape_inference import quant_pre_process
    except ImportError as exc:  # pragma: no cover - training environment only
        raise RuntimeError("quantization requires onnxruntime") from exc
    destination = Path(int8_model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    preprocessed = destination.with_name(f".{destination.stem}.preprocessed.onnx")
    try:
        quant_pre_process(
            input_model=str(fp32_model),
            output_model_path=str(preprocessed),
            # MobileNet uses fixed image shapes; ONNX shape inference and graph
            # optimization are sufficient and avoid a SymPy runtime dependency.
            skip_symbolic_shape=True,
        )
        quantize_static(
            model_input=str(preprocessed),
            model_output=str(destination),
            calibration_data_reader=_NpyCalibrationReader(calibration_arrays),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            reduce_range=True,
            per_channel=False,
        )
    finally:
        preprocessed.unlink(missing_ok=True)
    return destination


def benchmark_onnx(
    model_path: str | Path,
    arrays: Sequence[str | Path],
    *,
    repeats: int = 10,
    max_threads: int = 2,
) -> dict[str, float]:
    """Warm and time individual batch-1 calls on the machine under test."""

    if not arrays or repeats < 1:
        raise ValueError("benchmark requires arrays and a positive repeat count")
    import numpy as np
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = max_threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path), options, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    sample_arrays = []
    for path in arrays:
        sample = np.load(path).astype(np.float32)
        sample_arrays.append(sample[None, ...] if sample.ndim == 3 else sample)
    session.run(None, {input_name: sample_arrays[0]})
    latencies: list[float] = []
    for _ in range(repeats):
        for sample in sample_arrays:
            started = time.perf_counter()
            session.run(None, {input_name: sample})
            latencies.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(latencies)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "calls": float(len(ordered)),
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_index],
        "max_ms": ordered[-1],
    }


def fit_logistic_calibrator(
    matrix: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    iterations: int = 4000,
    learning_rate: float = 0.02,
    l2: float = 1e-4,
) -> tuple[float, list[float]]:
    """Fit a small deterministic logistic combiner to OOF-only features."""

    import numpy as np

    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(y) or len(x) == 0:
        raise ValueError("matrix and labels have incompatible shapes")
    if set(y.tolist()) - {0.0, 1.0}:
        raise ValueError("calibration labels must be binary")
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales < 1e-9] = 1.0
    normalized = (x - means) / scales
    weights = np.zeros(x.shape[1], dtype=np.float64)
    intercept = 0.0
    for _ in range(iterations):
        logits = np.clip(normalized @ weights + intercept, -40, 40)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        error = probabilities - y
        weights -= learning_rate * (normalized.T @ error / len(y) + l2 * weights)
        intercept -= learning_rate * float(error.mean())
    # Bake feature standardization into raw-space parameters for edge simplicity.
    raw_weights = weights / scales
    raw_intercept = intercept - float((weights * means / scales).sum())
    return raw_intercept, raw_weights.tolist()


def fit_event_max_logistic_calibrator(
    event_matrices: Sequence[Sequence[Sequence[float]]],
    labels: Sequence[int],
    *,
    iterations: int = 4000,
    learning_rate: float = 0.02,
    l2: float = 1e-4,
) -> tuple[float, list[float]]:
    """Fit a deterministic multi-instance calibrator on grouped OOF events.

    The score of an event is the maximum calibrated candidate score, exactly
    matching runtime.  Gradient updates therefore use the current winning
    candidate for each event instead of incorrectly treating every ROI as an
    independently labelled observation.
    """

    import numpy as np

    if len(event_matrices) != len(labels) or not event_matrices:
        raise ValueError("event matrices and labels have incompatible lengths")
    arrays = [np.asarray(matrix, dtype=np.float64) for matrix in event_matrices]
    if any(array.ndim != 2 or len(array) == 0 for array in arrays):
        raise ValueError("every event must contain a non-empty candidate matrix")
    feature_count = arrays[0].shape[1]
    if feature_count == 0 or any(array.shape[1] != feature_count for array in arrays):
        raise ValueError("candidate feature matrices have incompatible shapes")
    y = np.asarray(labels, dtype=np.float64)
    if set(y.tolist()) - {0.0, 1.0}:
        raise ValueError("calibration labels must be binary")
    all_candidates = np.concatenate(arrays, axis=0)
    if not np.isfinite(all_candidates).all():
        raise ValueError("calibration features must be finite")
    means = all_candidates.mean(axis=0)
    scales = all_candidates.std(axis=0)
    scales[scales < 1e-9] = 1.0
    normalized = [(array - means) / scales for array in arrays]
    weights = np.zeros(feature_count, dtype=np.float64)
    # roi_logit is always the first configured feature in the orchestrator.
    # Seeding that direction avoids an arbitrary first-ROI tie for positives.
    weights[0] = 1e-3
    intercept = 0.0
    for _ in range(iterations):
        winners = []
        logits = []
        for candidates in normalized:
            candidate_logits = candidates @ weights + intercept
            winner = int(np.argmax(candidate_logits))
            winners.append(candidates[winner])
            logits.append(float(candidate_logits[winner]))
        selected = np.stack(winners)
        clipped = np.clip(np.asarray(logits), -40, 40)
        probabilities = 1.0 / (1.0 + np.exp(-clipped))
        error = probabilities - y
        weights -= learning_rate * (selected.T @ error / len(y) + l2 * weights)
        intercept -= learning_rate * float(error.mean())
    raw_weights = weights / scales
    raw_intercept = intercept - float((weights * means / scales).sum())
    return raw_intercept, raw_weights.tolist()


def event_max_score(
    matrix: Sequence[Sequence[float]], intercept: float, coefficients: Sequence[float]
) -> float:
    """Return the runtime-equivalent maximum calibrated candidate probability."""

    if not matrix:
        raise ValueError("event has no candidates")
    if any(len(row) != len(coefficients) for row in matrix):
        raise ValueError("candidate features and coefficients have incompatible shapes")
    best_logit = max(
        intercept + sum(value * coefficient for value, coefficient in zip(row, coefficients, strict=True))
        for row in matrix
    )
    if best_logit >= 0:
        return 1.0 / (1.0 + math.exp(-best_logit))
    exp = math.exp(best_logit)
    return exp / (1.0 + exp)


def write_model_manifest(
    destination: str | Path,
    *,
    model_path: str | Path,
    model_version: str,
    feature_order: Sequence[str],
    coefficients: Sequence[float],
    intercept: float,
    possible_threshold: float,
    probable_threshold: float,
    quantization: str,
    quantization_evidence: dict[str, object] | None = None,
    image_size: int = 224,
) -> Path:
    if quantization not in {"fp32", "qdq_int8"}:
        raise ValueError("quantization must be fp32 or qdq_int8")
    if quantization == "qdq_int8" and quantization_evidence is None:
        raise ValueError("qdq_int8 requires passed locked-set and target-CPU evidence")
    if len(feature_order) != len(coefficients):
        raise ValueError("feature_order and coefficients must have equal lengths")
    if "roi_logit" not in feature_order or len(feature_order) < 2:
        raise ValueError("calibration must combine roi_logit with another feature")
    unknown = set(feature_order) - CALIBRATION_FEATURES
    if unknown:
        raise ValueError("unknown calibration features: " + ", ".join(sorted(unknown)))
    if not 0 <= possible_threshold <= probable_threshold <= 1:
        raise ValueError("thresholds must satisfy 0 <= possible <= probable <= 1")
    destination_path = Path(destination).resolve(strict=False)
    model = Path(model_path).resolve(strict=True)
    if model.parent != destination_path.parent:
        raise ValueError("model and manifest must be in the same model package directory")
    calibration = {
        "feature_order": list(feature_order),
        "coefficients": list(coefficients),
        "intercept": intercept,
        "fit_source": "grouped_out_of_fold_predictions",
    }
    import hashlib

    calibration["sha256"] = hashlib.sha256(
        json.dumps(calibration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    document: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "model_version": model_version,
        "model": {
            "file": model.name,
            "sha256": sha256_file(model),
            "input_name": "image",
            "output_name": "logit",
            "quantization": quantization,
        },
        "preprocessing": {
            "image_size": image_size,
            "color_order": "RGB",
            "resize": "aspect_preserving_letterbox",
            "padding": "roi_median",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "candidate_extractor": CANDIDATE_EXTRACTOR,
        "calibration": calibration,
        "thresholds": {
            "possible_fireball": possible_threshold,
            "probable_fireball": probable_threshold,
        },
    }
    if quantization_evidence is not None:
        document["quantization_evidence"] = quantization_evidence
    return write_json_atomic(destination_path, document)
