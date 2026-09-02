"""Leakage-resistant grouping for CV and an unseen-night/camera locked set."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import asdict
from typing import Iterable

from .manifest import ManifestRecord


def assign_grouped_partitions(
    records: Iterable[ManifestRecord],
    *,
    locked_nights: set[str],
    locked_cameras: set[str],
    fold_count: int = 5,
    seed: str = "fireball-edge-v1",
) -> list[dict[str, object]]:
    """Keep physical events together and preserve the locked-set prevalence."""

    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    grouped: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        grouped[record.physical_event_id].append(record)
    if not grouped:
        raise ValueError("cannot split an empty manifest")

    locked_groups = {
        group
        for group, observations in grouped.items()
        if any(
            item.night in locked_nights or item.camera in locked_cameras
            for item in observations
        )
    }
    train_groups = [group for group in grouped if group not in locked_groups]
    if not locked_groups:
        raise ValueError("locked set is empty; select at least one unseen night or camera")
    if len(train_groups) < fold_count:
        raise ValueError("not enough non-locked physical events for grouped folds")

    def stable_key(group: str) -> str:
        return hashlib.sha256(f"{seed}\0{group}".encode()).hexdigest()

    # Large multi-station groups are placed first; the seeded digest resolves ties.
    ordered = sorted(train_groups, key=lambda group: (-len(grouped[group]), stable_key(group)))
    fold_groups: list[list[str]] = [[] for _ in range(fold_count)]
    fold_observations = [0] * fold_count
    fold_positives = [0] * fold_count
    for group in ordered:
        observations = grouped[group]
        positive = int(observations[0].label == "fireball")
        target = min(
            range(fold_count),
            key=lambda index: (
                fold_positives[index] if positive else fold_observations[index] - fold_positives[index],
                fold_observations[index],
                index,
            ),
        )
        fold_groups[target].append(group)
        fold_observations[target] += len(observations)
        fold_positives[target] += positive * len(observations)

    assignments = {
        group: "locked" for group in locked_groups
    } | {
        group: f"fold_{index + 1}"
        for index, groups in enumerate(fold_groups)
        for group in groups
    }
    result: list[dict[str, object]] = []
    for record in records:
        document = asdict(record)
        document["partition"] = assignments[record.physical_event_id]
        result.append(document)
    return result
