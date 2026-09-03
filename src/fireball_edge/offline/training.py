"""MobileNetV3-Small training on cached candidate ROIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model_tools import create_mobilenet_v3_small_binary


HARD_NEGATIVE_TAGS = frozenset(
    {"moon", "moon_only", "cloud_glare", "aircraft", "sensor_artifact", "ordinary_meteor"}
)


class CachedRoiDataset:
    """Torch-compatible multi-instance event dataset."""

    def __init__(self, index_path: str | Path, partitions: set[str]) -> None:
        with Path(index_path).open("r", encoding="utf-8") as source:
            records = json.load(source)["records"]
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
) -> Path:
    """Fine-tune ImageNet weights; callers generate each grouped fold separately."""

    import torch
    from torch.utils.data import DataLoader

    dataset = CachedRoiDataset(index_path, train_partitions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model = create_mobilenet_v3_small_binary()
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
