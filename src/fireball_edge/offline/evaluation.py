"""Grouped-threshold selection and locked-set reporting without sklearn."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ..artifacts import write_json_atomic


@dataclass(frozen=True)
class Prediction:
    event_id: str
    physical_event_id: str
    fold: str
    label: int
    score: float
    camera: str
    night: str
    nuisance_tags: tuple[str, ...] = ()


def load_predictions(path: str | Path) -> list[Prediction]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {
            "event_id", "physical_event_id", "fold", "label", "score", "camera", "night",
            "nuisance_tags",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"prediction CSV is missing: {', '.join(sorted(missing))}")
        predictions = [
            Prediction(
                event_id=row["event_id"],
                physical_event_id=row["physical_event_id"],
                fold=row["fold"],
                label=int(row["label"]),
                score=float(row["score"]),
                camera=row["camera"],
                night=row["night"],
                nuisance_tags=tuple(tag for tag in row["nuisance_tags"].split(";") if tag),
            )
            for row in reader
        ]
    if not predictions:
        raise ValueError("prediction CSV contains no rows")
    if any(item.label not in (0, 1) or not 0 <= item.score <= 1 for item in predictions):
        raise ValueError("labels must be binary and scores must be in [0, 1]")
    return predictions


def _recall(rows: Sequence[Prediction], threshold: float) -> float:
    positives = sum(item.label for item in rows)
    if positives == 0:
        raise ValueError("recall is undefined for a set without positive examples")
    true_positives = sum(item.label == 1 and item.score >= threshold for item in rows)
    return true_positives / positives


def select_possible_threshold(
    rows: Sequence[Prediction],
    *,
    target_recall: float = 0.98,
    minimum_fold_recall: float = 0.95,
) -> float:
    """Choose the highest score cutoff satisfying both OOF recall constraints."""

    groups: dict[str, list[Prediction]] = {}
    physical_folds: dict[str, set[str]] = {}
    for item in rows:
        groups.setdefault(item.fold, []).append(item)
        physical_folds.setdefault(item.physical_event_id, set()).add(item.fold)
    leaking = [group for group, folds in physical_folds.items() if len(folds) > 1]
    if leaking:
        raise ValueError("physical events span folds: " + ", ".join(sorted(leaking)))
    candidates = sorted({0.0, *(item.score for item in rows)}, reverse=True)
    for threshold in candidates:
        if _recall(rows, threshold) < target_recall:
            continue
        if all(_recall(fold_rows, threshold) >= minimum_fold_recall for fold_rows in groups.values()):
            return threshold
    raise ValueError("no possible-fireball threshold satisfies recall constraints")


def select_probable_threshold(rows: Sequence[Prediction]) -> float:
    """Choose maximum F2, preferring the higher threshold on a tie."""

    best = (-1.0, 0.0)
    for threshold in sorted({0.0, *(item.score for item in rows)}, reverse=True):
        tp = sum(item.label == 1 and item.score >= threshold for item in rows)
        fp = sum(item.label == 0 and item.score >= threshold for item in rows)
        fn = sum(item.label == 1 and item.score < threshold for item in rows)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f2 = 5 * precision * recall / max(4 * precision + recall, 1e-12)
        best = max(best, (f2, threshold))
    return best[1]


def pr_auc(rows: Sequence[Prediction]) -> float:
    ordered = sorted(rows, key=lambda item: item.score, reverse=True)
    positives = sum(item.label for item in ordered)
    if positives == 0:
        raise ValueError("PR-AUC is undefined without positive examples")
    tp = fp = 0
    previous_recall = 0.0
    area = 0.0
    for item in ordered:
        tp += item.label
        fp += 1 - item.label
        recall = tp / positives
        precision = tp / (tp + fp)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def expected_calibration_error(rows: Sequence[Prediction], bins: int = 10) -> float:
    total = len(rows)
    result = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        bucket = [
            item for item in rows
            if lower <= item.score < upper or (index == bins - 1 and item.score == 1.0)
        ]
        if bucket:
            confidence = sum(item.score for item in bucket) / len(bucket)
            accuracy = sum(item.label for item in bucket) / len(bucket)
            result += len(bucket) / total * abs(confidence - accuracy)
    return result


def locked_report(rows: Sequence[Prediction], possible_threshold: float) -> dict[str, object]:
    slices = (
        "moon_only", "fireball_with_moon", "ordinary_meteor", "cloud_glare",
        "unseen_camera", "saturation",
    )
    slice_report: dict[str, object] = {}
    for name in slices:
        selected = [item for item in rows if name in item.nuisance_tags]
        if selected:
            positives = sum(item.label for item in selected)
            slice_report[name] = {
                "count": len(selected),
                "recall": _recall(selected, possible_threshold) if positives else None,
                "alerts": sum(item.score >= possible_threshold for item in selected),
            }
    camera_nights = {(item.camera, item.night) for item in rows}
    false_alerts = sum(
        item.label == 0 and item.score >= possible_threshold for item in rows
    )
    return {
        "count": len(rows),
        "possible_fireball_recall": _recall(rows, possible_threshold),
        "false_alerts_per_camera_night": false_alerts / max(len(camera_nights), 1),
        "pr_auc": pr_auc(rows),
        "expected_calibration_error": expected_calibration_error(rows),
        "slices": slice_report,
    }


def quantization_gate(
    *, fp32_recall: float, int8_recall: float, fp32_p95_ms: float, int8_p95_ms: float
) -> dict[str, object]:
    recall_drop = fp32_recall - int8_recall
    latency_improvement = (fp32_p95_ms - int8_p95_ms) / fp32_p95_ms
    return {
        "ship_int8": recall_drop <= 0.01 + 1e-12 and latency_improvement >= 0.15 - 1e-12,
        "recall_drop": recall_drop,
        "p95_latency_improvement": latency_improvement,
    }


def write_quantization_report(
    destination: str | Path,
    *,
    target_cpu: str,
    locked_predictions_sha256: str,
    fp32_recall: float,
    int8_recall: float,
    fp32_p95_ms: float,
    int8_p95_ms: float,
) -> Path:
    """Publish the evidence file that an INT8 model manifest must bind."""

    gate = quantization_gate(
        fp32_recall=fp32_recall,
        int8_recall=int8_recall,
        fp32_p95_ms=fp32_p95_ms,
        int8_p95_ms=int8_p95_ms,
    )
    return write_json_atomic(
        destination,
        {
            "schema_version": 1,
            "target_cpu": target_cpu,
            "locked_predictions_sha256": locked_predictions_sha256,
            "fp32": {"recall": fp32_recall, "p95_ms": fp32_p95_ms},
            "int8": {"recall": int8_recall, "p95_ms": int8_p95_ms},
            **gate,
        },
    )
