"""End-to-end schema-v2 offline training orchestration.

This module is deliberately outside the edge entry point: importing the
runtime never imports Torch, torchvision, or ONNX training dependencies.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Sequence

from ..artifacts import write_image_atomic, write_json_atomic
from ..bundles import normalize_clip_base
from ..config import EdgeConfig
from ..contracts import CANDIDATE_EXTRACTOR, SCHEMA_VERSION
from ..inference import ModelManifest, sha256_file
from .cache import build_roi_cache
from .evaluation import (
    Prediction,
    locked_report,
    select_possible_threshold,
    select_probable_threshold,
)
from .manifest import (
    ManifestRecord,
    build_records,
    compare_snapshots,
    load_expert_labels,
    snapshot_tree,
    write_manifest,
)
from .model_tools import (
    event_max_score,
    export_onnx,
    fit_event_max_logistic_calibrator,
    write_model_manifest,
)
from .splits import assign_grouped_partitions
from .training import event_feature_matrices, predict_cached_candidates, train_classifier


DEFAULT_FEATURE_ORDER = (
    "roi_logit",
    "log_changed_pixels",
    "roi_area_fraction",
    "roi_aspect_ratio",
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
)


def _contract_document(path: Path, kind: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as source:
            document = json.load(source)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot resume from {kind}: {path}") from exc
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != SCHEMA_VERSION
        or document.get("candidate_extractor") != CANDIDATE_EXTRACTOR
    ):
        raise ValueError(f"unsupported {kind} schema or extractor; rebuild with v2")
    return document


def _write_immutable_manifest(path: Path, records: Sequence[ManifestRecord]) -> Path:
    # Normalize tuples to their JSON list representation before comparing an
    # existing immutable manifest during resume.
    expected_records = json.loads(json.dumps([asdict(record) for record in records]))
    if path.exists():
        existing = _contract_document(path, "manifest")
        if existing.get("records") != expected_records:
            raise ValueError(
                f"immutable manifest already exists with different content: {path}"
            )
        return path
    return write_manifest(path, records)


def _event_predictions(
    events: Sequence[dict[str, Any]],
    feature_order: tuple[str, ...],
    intercept: float,
    coefficients: Sequence[float],
) -> list[Prediction]:
    matrices, _ = event_feature_matrices(events, feature_order)
    result: list[Prediction] = []
    for event, matrix in zip(events, matrices, strict=True):
        result.append(
            Prediction(
                event_id=str(event["event_id"]),
                physical_event_id=str(event["physical_event_id"]),
                fold=str(event["partition"]),
                label=1 if event["label"] == "fireball" else 0,
                score=event_max_score(matrix, intercept, coefficients),
                camera=str(event["camera"]),
                night=str(event["night"]),
                nuisance_tags=tuple(str(tag) for tag in event.get("nuisance_tags", [])),
            )
        )
    return result


def _prediction_document(rows: Sequence[Prediction]) -> list[dict[str, Any]]:
    return [asdict(row) for row in rows]


def _verify_source_immutability(function: Callable[..., Path]) -> Callable[..., Path]:
    """Check capture trees even when preflight, training, or export raises."""

    @wraps(function)
    def wrapped(config: EdgeConfig, **kwargs: Any) -> Path:
        label_rows = load_expert_labels(kwargs["labels_path"])
        for row in label_rows:
            config.validate_clip_base(normalize_clip_base(row["clip_base"]))
        roots = sorted(
            {str(normalize_clip_base(row["clip_base"]).parent) for row in label_rows}
        )
        before = {root: snapshot_tree(root) for root in roots}

        def guard() -> None:
            differences = {
                root: compare_snapshots(before[root], snapshot_tree(root))
                for root in roots
            }
            changed = {root: items for root, items in differences.items() if items}
            if changed:
                raise RuntimeError(f"source tree changed during training: {changed}")

        kwargs["_source_immutability_guard"] = guard
        try:
            return function(config, **kwargs)
        finally:
            guard()

    return wrapped


@_verify_source_immutability
def run_training(
    config: EdgeConfig,
    *,
    labels_path: str | Path,
    dataset_name: str,
    model_name: str,
    locked_nights: set[str],
    locked_cameras: set[str],
    resume: bool = False,
    folds: int = 5,
    epochs: int = 12,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    seed: int = 1729,
    feature_order: tuple[str, ...] = DEFAULT_FEATURE_ORDER,
    model_factory: Callable[[], Any] | None = None,
    onnx_exporter: Callable[..., Path] = export_onnx,
    _source_immutability_guard: Callable[[], None] | None = None,
) -> Path:
    """Build, validate, train, export, and evaluate one immutable v2 dataset.

    The function publishes a candidate model package and an activation gate;
    it never replaces an active package.  INT8 and target-i7 benchmarking stay
    separate behind their existing evidence gate.
    """

    for kind, value in (("dataset", dataset_name), ("model", model_name)):
        if not value or value in {".", ".."} or Path(value).parts != (value,):
            raise ValueError(f"{kind} name must be one path-safe component")
    if not locked_nights and not locked_cameras:
        raise ValueError("train requires at least one locked night or camera")
    model_root = config.state_root / "models" / "candidates" / "v2" / model_name
    candidate_model_path = model_root / "candidate-fp32.onnx"
    candidate_manifest_path = model_root / "model-manifest.json"
    label_rows = load_expert_labels(labels_path)
    for row in label_rows:
        config.validate_clip_base(normalize_clip_base(row["clip_base"]))
    source_roots = sorted(
        {str(normalize_clip_base(row["clip_base"]).parent) for row in label_rows}
    )
    records = build_records(label_rows)
    for record in records:
        config.validate_clip_base(record.clip_base)
    preflight_path = config.state_root / "validation" / f"{dataset_name}-preflight-v2.json"
    write_json_atomic(
        preflight_path,
        {
            "schema_version": SCHEMA_VERSION,
            "candidate_extractor": CANDIDATE_EXTRACTOR,
            "status": "passed",
            "records": [
                {
                    "event_id": record.event_id,
                    "clip_base": record.clip_base,
                    "station": record.station,
                    "camera": record.camera,
                    "night": record.night,
                    "stack_image": record.stack_image,
                    "star_mask_role": record.star_mask_role,
                    "xml_validation": record.xml_validation,
                    "warnings": list(record.metadata_warnings),
                }
                for record in records
            ],
        },
    )

    dataset_root = config.state_root / "datasets" / dataset_name
    manifest_path = dataset_root / "manifest-v2.json"
    _write_immutable_manifest(manifest_path, records)
    manifest_sha256 = sha256_file(manifest_path)

    assigned = assign_grouped_partitions(
        records,
        locked_nights=locked_nights,
        locked_cameras=locked_cameras,
        fold_count=folds,
        seed=f"fireball-edge-v2:{seed}",
    )
    partition_path = dataset_root / "partitioned-manifest-v2.json"
    partition_document = json.loads(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "candidate_extractor": CANDIDATE_EXTRACTOR,
        "manifest_sha256": manifest_sha256,
        "grouping_key": "physical_event_id",
        "locked_selection": {
            "nights": sorted(locked_nights),
            "cameras": sorted(locked_cameras),
        },
        "records": assigned,
    }))
    if partition_path.exists():
        if _contract_document(partition_path, "partition manifest") != partition_document:
            raise ValueError("partition manifest already exists with different content")
    else:
        write_json_atomic(partition_path, partition_document)

    cache_root = dataset_root / "roi-cache-v2"
    cache_path = cache_root / "index.json"
    if cache_path.exists():
        cache_document = _contract_document(cache_path, "cache")
        if cache_document.get("manifest_sha256") != manifest_sha256:
            raise ValueError("cache is not bound to the immutable manifest")
    else:
        cache_path = build_roi_cache(
            assigned,
            cache_root,
            manifest_sha256=manifest_sha256,
        )

    training_root = config.state_root / "training" / dataset_name / model_name
    state_path = training_root / "training-state-v2.json"
    parameters = {
        "folds": folds,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "feature_order": list(feature_order),
    }
    completed_folds: list[str] = []
    fold_checkpoint_hashes: dict[str, str] = {}
    if not state_path.exists() and (
        candidate_model_path.exists() or candidate_manifest_path.exists()
    ):
        raise ValueError("candidate package already exists without matching resume state")
    if candidate_manifest_path.exists():
        _contract_document(candidate_manifest_path, "model manifest")
    if state_path.exists():
        state = _contract_document(state_path, "training state")
        if not resume:
            raise ValueError("training state already exists; use --resume or a new model name")
        if state.get("manifest_sha256") != manifest_sha256 or state.get("parameters") != parameters:
            raise ValueError("resume state does not match the manifest or training parameters")
        if state.get("status") == "complete":
            report = Path(str(state.get("report", ""))).resolve(strict=False)
            try:
                report.relative_to((config.state_root / "validation").resolve(strict=False))
            except ValueError as exc:
                raise ValueError("completed resume report is outside validation state") from exc
            if report.is_file():
                completed_report = _contract_document(report, "training report")
                if Path(str(completed_report.get("model_manifest", ""))).resolve(
                    strict=False
                ) != candidate_manifest_path.resolve(strict=False):
                    raise ValueError("completed report references a different model package")
                loaded_model = ModelManifest.load(completed_report["model_manifest"])
                final_checkpoint = Path(str(state.get("final_checkpoint", "")))
                if (
                    not final_checkpoint.is_file()
                    or state.get("final_checkpoint_sha256") != sha256_file(final_checkpoint)
                    or state.get("model_manifest_sha256")
                    != loaded_model.manifest_sha256
                ):
                    raise ValueError("completed resume artifacts do not match their hashes")
                return report
            raise ValueError("completed resume state references a missing report")
        completed_folds = [str(item) for item in state.get("completed_folds", [])]
        raw_hashes = state.get("fold_checkpoint_sha256")
        if not isinstance(raw_hashes, dict) or set(raw_hashes) != set(completed_folds):
            raise ValueError("resume state is missing fold checkpoint hashes")
        fold_checkpoint_hashes = {
            str(name): str(digest) for name, digest in raw_hashes.items()
        }

    fold_names = sorted(
        {str(record["partition"]) for record in assigned if record["partition"] != "locked"}
    )
    if len(fold_names) != folds:
        raise ValueError("grouped split did not populate every requested fold")
    oof_events: list[dict[str, Any]] = []
    for fold_index, fold_name in enumerate(fold_names):
        checkpoint = training_root / "folds" / f"{fold_name}.pt"
        if fold_name not in completed_folds:
            train_classifier(
                cache_path,
                checkpoint,
                train_partitions=set(fold_names) - {fold_name},
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed + fold_index,
                model_factory=model_factory,
            )
            completed_folds.append(fold_name)
            fold_checkpoint_hashes[fold_name] = sha256_file(checkpoint)
            write_json_atomic(
                state_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "candidate_extractor": CANDIDATE_EXTRACTOR,
                    "manifest_sha256": manifest_sha256,
                    "parameters": parameters,
                    "completed_folds": completed_folds,
                    "fold_checkpoint_sha256": fold_checkpoint_hashes,
                    "status": "fold_training",
                },
            )
        elif (
            not checkpoint.is_file()
            or fold_checkpoint_hashes.get(fold_name) != sha256_file(checkpoint)
        ):
            raise ValueError(f"resume checkpoint is missing or has changed: {checkpoint}")
        oof_events.extend(
            predict_cached_candidates(
                cache_path,
                checkpoint,
                partitions={fold_name},
                model_factory=model_factory,
            )
        )

    event_matrices, event_labels = event_feature_matrices(oof_events, feature_order)
    intercept, coefficients = fit_event_max_logistic_calibrator(
        event_matrices, event_labels
    )
    oof_rows = _event_predictions(
        oof_events, feature_order, intercept, coefficients
    )
    possible = select_possible_threshold(oof_rows)
    probable = max(possible, select_probable_threshold(oof_rows))
    write_json_atomic(
        training_root / "oof-predictions-v2.json",
        {
            "schema_version": SCHEMA_VERSION,
            "candidate_extractor": CANDIDATE_EXTRACTOR,
            "event_aggregation": "maximum_calibrated_candidate_score",
            "predictions": _prediction_document(oof_rows),
        },
    )

    final_checkpoint = training_root / "final-fp32.pt"
    train_classifier(
        cache_path,
        final_checkpoint,
        train_partitions=set(fold_names),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        model_factory=model_factory,
    )
    locked_events = predict_cached_candidates(
        cache_path,
        final_checkpoint,
        partitions={"locked"},
        model_factory=model_factory,
    )
    locked_rows = _event_predictions(
        locked_events, feature_order, intercept, coefficients
    )
    locked_evaluation = locked_report(locked_rows, possible)
    gate_passed = float(locked_evaluation["possible_fireball_recall"]) >= 0.95

    staged_model = onnx_exporter(
        final_checkpoint,
        training_root / "package-staging" / "candidate-fp32.onnx",
        image_size=224,
        model_factory=model_factory,
    )
    # Recheck after every source-consuming and export step but before publishing
    # the candidate package. The decorator repeats this on every exit path.
    if _source_immutability_guard is not None:
        _source_immutability_guard()

    model_path = write_image_atomic(
        candidate_model_path,
        lambda temporary: shutil.copyfile(staged_model, temporary),
    )
    model_manifest = write_model_manifest(
        candidate_manifest_path,
        model_path=model_path,
        model_version=model_name,
        feature_order=feature_order,
        coefficients=coefficients,
        intercept=intercept,
        possible_threshold=possible,
        probable_threshold=probable,
        quantization="fp32",
    )

    report_path = config.state_root / "validation" / f"{dataset_name}-{model_name}-v2.json"
    write_json_atomic(
        report_path,
        {
            "schema_version": SCHEMA_VERSION,
            "candidate_extractor": CANDIDATE_EXTRACTOR,
            "manifest": str(manifest_path),
            "preflight": str(preflight_path),
            "manifest_sha256": manifest_sha256,
            "cache": str(cache_path),
            "event_aggregation": "maximum_calibrated_candidate_score",
            "oof": {
                "event_count": len(oof_rows),
                "possible_threshold": possible,
                "probable_threshold": probable,
            },
            "locked_test": locked_evaluation,
            "rollout_gate": {
                "minimum_possible_recall": 0.95,
                "passed": gate_passed,
            },
            "activation": {
                "eligible": gate_passed,
                "performed": False,
                "reason": "training never replaces the configured active package",
            },
            "model_manifest": str(model_manifest),
            "source_immutability": {
                "verified": True,
                "roots": source_roots,
            },
        },
    )
    write_json_atomic(
        state_path,
        {
            "schema_version": SCHEMA_VERSION,
            "candidate_extractor": CANDIDATE_EXTRACTOR,
            "manifest_sha256": manifest_sha256,
            "parameters": parameters,
            "completed_folds": completed_folds,
            "fold_checkpoint_sha256": fold_checkpoint_hashes,
            "status": "complete",
            "report": str(report_path),
            "final_checkpoint": str(final_checkpoint),
            "final_checkpoint_sha256": sha256_file(final_checkpoint),
            "model_manifest_sha256": sha256_file(model_manifest),
        },
    )
    return report_path
