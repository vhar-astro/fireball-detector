"""MobileNetV3-Small training on cached candidate ROIs."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable, Iterable

from ..contracts import CANDIDATE_EXTRACTOR, SCHEMA_VERSION
from ..inference import calibration_features, sha256_file
from ..vision import CandidateRegion, TemporalFeatures
from .model_tools import create_mobilenet_v3_small_binary


HARD_NEGATIVE_TAGS = frozenset(
    {"moon", "moon_only", "cloud_glare", "aircraft", "sensor_artifact", "ordinary_meteor"}
)


def _load_cache_document(index_path: str | Path) -> dict[str, Any]:
    with Path(index_path).open("r", encoding="utf-8") as source:
        document = json.load(source)
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != SCHEMA_VERSION
        or document.get("candidate_extractor") != CANDIDATE_EXTRACTOR
    ):
        raise ValueError("unsupported cache schema or candidate extractor; rebuild cache")
    records = document.get("records")
    if not isinstance(records, list):
        raise ValueError("cache records must be a list")
    manifest_sha256 = document.get("manifest_sha256")
    try:
        manifest_hash_valid = (
            isinstance(manifest_sha256, str)
            and len(manifest_sha256) == 64
            and len(bytes.fromhex(manifest_sha256)) == 32
        )
    except ValueError:
        manifest_hash_valid = False
    if not manifest_hash_valid:
        raise ValueError("cache is missing its manifest hash; rebuild cache")
    for record in records:
        candidates = record.get("candidates") if isinstance(record, dict) else None
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("cached event has no candidate ROIs")
        for candidate in candidates:
            if (
                candidate.get("candidate_extractor") != CANDIDATE_EXTRACTOR
                or candidate.get("candidate_source") != "avi_sequential_difference"
                or not isinstance(candidate.get("roi"), dict)
                or not isinstance(candidate.get("image_geometry"), dict)
                or not isinstance(candidate.get("temporal_features"), dict)
                or not isinstance(candidate.get("source_identity"), dict)
                or not {"avi", "stack_image"}.issubset(candidate["source_identity"])
            ):
                raise ValueError("cache candidate contract mismatch; rebuild cache")
            try:
                roi_path = Path(candidate["roi_npy"]).resolve(strict=True)
            except (KeyError, OSError, TypeError) as exc:
                raise ValueError("cached ROI file is missing; rebuild cache") from exc
            if candidate.get("roi_sha256") != sha256_file(roi_path):
                raise ValueError("cached ROI hash mismatch; rebuild cache")
    return document


class CachedRoiDataset:
    """Torch-compatible multi-instance event dataset."""

    def __init__(self, index_path: str | Path, partitions: set[str]) -> None:
        document = _load_cache_document(index_path)
        records = document["records"]
        self.records = [item for item in records if item.get("partition") in partitions]
        if not self.records:
            raise ValueError("no cached ROI records match the requested partitions")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        import numpy as np
        import torch

        record = self.records[index]
        candidates = record.get("candidates", [])
        if not candidates:
            raise ValueError("cached event has no candidate ROIs")
        arrays = [
            np.load(candidate["roi_npy"]).astype(np.float32)
            for candidate in candidates[:3]
        ]
        while len(arrays) < 3:
            arrays.append(np.zeros_like(arrays[0]))
        tensor = torch.from_numpy(np.stack(arrays))
        mask = torch.tensor([True] * min(len(candidates), 3) + [False] * max(0, 3 - len(candidates)))
        target = torch.tensor(1.0 if record["label"] == "fireball" else 0.0)
        tags = set(record.get("nuisance_tags", []))
        # Moon-containing positives retain ordinary positive weight: there is
        # deliberately no Moon veto. Only nuisance negatives are emphasized.
        sample_weight = 2.0 if target.item() == 0.0 and tags & HARD_NEGATIVE_TAGS else 1.0
        return tensor, mask, target, torch.tensor(sample_weight)


def train_classifier(
    index_path: str | Path,
    output_checkpoint: str | Path,
    *,
    train_partitions: set[str],
    epochs: int = 12,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    seed: int = 1729,
    model_factory: Callable[[], Any] | None = None,
) -> Path:
    """Fine-tune ImageNet weights; callers generate each grouped fold separately."""

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = CachedRoiDataset(index_path, train_partitions)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    model = (model_factory or create_mobilenet_v3_small_binary)()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(epochs):
        for images, masks, targets, weights in loader:
            optimizer.zero_grad(set_to_none=True)
            batch, candidates, channels, height, width = images.shape
            logits = model(images.reshape(batch * candidates, channels, height, width)).reshape(
                batch, candidates
            )
            masked_logits = logits.masked_fill(~masks, -1e9)
            positive_loss = torch.nn.functional.softplus(-masked_logits.max(dim=1).values)
            negative_losses = torch.nn.functional.softplus(logits) * masks
            negative_loss = negative_losses.sum(dim=1) / masks.sum(dim=1).clamp_min(1)
            event_loss = torch.where(targets > 0.5, positive_loss, negative_loss)
            loss = (event_loss * weights).mean()
            loss.backward()
            optimizer.step()
    destination = Path(output_checkpoint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination)
    return destination


def predict_cached_candidates(
    index_path: str | Path,
    checkpoint: str | Path,
    *,
    partitions: set[str],
    model_factory: Callable[[], Any] | None = None,
) -> list[dict[str, Any]]:
    """Return candidate logits and runtime-identical feature dictionaries.

    Each returned row is still an event containing up to three candidates.  It
    is intentionally not flattened: calibration must apply the same maximum
    over candidates that production inference applies.
    """

    import numpy as np
    import torch

    document = _load_cache_document(index_path)
    model = (model_factory or create_mobilenet_v3_small_binary)()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    result: list[dict[str, Any]] = []
    with torch.no_grad():
        for record in document.get("records", []):
            if record.get("partition") not in partitions:
                continue
            predicted: list[dict[str, Any]] = []
            for candidate in record.get("candidates", [])[:3]:
                tensor = np.load(candidate["roi_npy"]).astype(np.float32)
                batch = torch.from_numpy(tensor[None, ...] if tensor.ndim == 3 else tensor)
                roi_logit = float(model(batch).reshape(-1)[0].item())
                region = CandidateRegion(**candidate["roi"])
                temporal = TemporalFeatures(**candidate["temporal_features"])
                image_geometry = candidate["image_geometry"]
                features = calibration_features(
                    roi_logit,
                    region,
                    temporal,
                    int(image_geometry["width"]),
                    int(image_geometry["height"]),
                )
                predicted.append(
                    {
                        "roi_logit": roi_logit,
                        "features": features,
                        "roi": candidate["roi"],
                    }
                )
            if not predicted:
                raise ValueError(f"cached event {record.get('event_id')} has no candidates")
            result.append(
                {
                    "event_id": record["event_id"],
                    "physical_event_id": record["physical_event_id"],
                    "partition": record["partition"],
                    "label": record["label"],
                    "camera": record["camera"],
                    "night": record["night"],
                    "nuisance_tags": list(record.get("nuisance_tags", [])),
                    "candidates": predicted,
                }
            )
    if not result:
        raise ValueError("no cached events match prediction partitions")
    return result


def event_feature_matrices(
    events: Iterable[dict[str, Any]], feature_order: tuple[str, ...]
) -> tuple[list[list[list[float]]], list[int]]:
    """Project candidate dictionaries to ordered per-event matrices."""

    matrices: list[list[list[float]]] = []
    labels: list[int] = []
    for event in events:
        candidates = event.get("candidates", [])
        matrix = [
            [float(candidate["features"][name]) for name in feature_order]
            for candidate in candidates
        ]
        if not matrix:
            raise ValueError(f"event {event.get('event_id')} has no calibration candidates")
        matrices.append(matrix)
        labels.append(1 if event["label"] == "fireball" else 0)
    return matrices, labels
