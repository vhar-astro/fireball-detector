"""Expert-label manifest construction for complete UFOCapture bundles."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from ..artifacts import write_json_atomic
from ..bundles import discover_bundle, normalize_clip_base
from ..inference import sha256_file


VALID_LABELS = frozenset({"fireball", "non_fireball"})


@dataclass(frozen=True)
class ManifestRecord:
    event_id: str
    clip_base: str
    physical_event_id: str
    station: str
    camera: str
    night: str
    label: str
    nuisance_tags: tuple[str, ...]
    avi: str
    peak: str | None
    change_map: str | None
    xml: str | None


def _event_id(physical_event_id: str, station: str, camera: str, clip_base: Path) -> str:
    import hashlib

    key = "\0".join((physical_event_id, station, camera, str(clip_base)))
    return f"obs_{hashlib.sha256(key.encode('utf-8')).hexdigest()}"


def load_expert_labels(path: str | Path) -> list[dict[str, str]]:
    label_path = Path(path).expanduser().resolve(strict=True)
    with label_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {
            "clip_base",
            "label",
            "physical_event_id",
            "station",
            "camera",
            "night",
            "nuisance_tags",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"expert-label CSV is missing: {', '.join(sorted(missing))}")
        rows = [dict(row) for row in reader]
    for row in rows:
        clip_base = Path(row["clip_base"]).expanduser()
        if not clip_base.is_absolute():
            row["clip_base"] = str((label_path.parent / clip_base).resolve(strict=False))
    return rows


def build_records(rows: Iterable[dict[str, str]]) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    seen_observations: set[tuple[str, str, str]] = set()
    for row_number, row in enumerate(rows, start=2):
        label = row["label"].strip().casefold()
        if label not in VALID_LABELS:
            raise ValueError(f"row {row_number}: label must be fireball or non_fireball")
        required_values = ("clip_base", "physical_event_id", "station", "camera", "night")
        empty = [name for name in required_values if not row[name].strip()]
        if empty:
            raise ValueError(f"row {row_number}: empty values: {', '.join(empty)}")
        bundle = discover_bundle(row["clip_base"])
        if bundle.avi is None:
            raise ValueError(f"row {row_number}: complete bundle requires an AVI")
        observation_key = (
            row["physical_event_id"].strip(),
            row["station"].strip(),
            row["camera"].strip(),
        )
        if observation_key in seen_observations:
            raise ValueError(f"row {row_number}: duplicate physical-event/station/camera")
        seen_observations.add(observation_key)
        tags = tuple(
            sorted({tag.strip().casefold() for tag in row["nuisance_tags"].split(";") if tag.strip()})
        )
        records.append(
            ManifestRecord(
                event_id=_event_id(*observation_key, bundle.clip_base),
                clip_base=str(bundle.clip_base),
                physical_event_id=observation_key[0],
                station=observation_key[1],
                camera=observation_key[2],
                night=row["night"].strip(),
                label=label,
                nuisance_tags=tags,
                avi=str(bundle.avi),
                peak=str(bundle.peak) if bundle.peak else None,
                change_map=str(bundle.change_map) if bundle.change_map else None,
                xml=str(bundle.xml) if bundle.xml else None,
            )
        )
    if not records:
        raise ValueError("expert-label CSV contains no records")
    return records


def write_manifest(path: str | Path, records: Iterable[ManifestRecord]) -> Path:
    """Write one deterministic JSON document under an external root."""

    serialized = [asdict(record) for record in records]
    groups: dict[str, set[str]] = {}
    for record in serialized:
        groups.setdefault(record["physical_event_id"], set()).add(record["label"])
    conflicts = [group for group, labels in groups.items() if len(labels) > 1]
    if conflicts:
        raise ValueError(
            "physical events have conflicting labels: " + ", ".join(sorted(conflicts))
        )
    return write_json_atomic(
        path,
        {
            "schema_version": 1,
            "grouping_key": "physical_event_id",
            "label_source": "human_expert",
            "records": serialized,
        },
    )


def snapshot_tree(root: str | Path) -> dict[str, object]:
    """Capture the immutable attributes used by before/after validation."""

    root_path = Path(root).resolve(strict=True)
    records: list[dict[str, object]] = []
    for path in sorted(item for item in root_path.rglob("*") if item.is_file()):
        stat = path.stat()
        records.append(
            {
                "path": path.relative_to(root_path).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                # On Windows this is creation time; on Unix it is metadata-change time.
                "ctime_ns": stat.st_ctime_ns,
                "sha256": sha256_file(path),
            }
        )
    return {"root": str(root_path), "files": records}


def compare_snapshots(before: dict[str, object], after: dict[str, object]) -> list[str]:
    if before.get("root") != after.get("root"):
        return ["snapshot roots differ"]
    before_files = {item["path"]: item for item in before.get("files", [])}  # type: ignore[index]
    after_files = {item["path"]: item for item in after.get("files", [])}  # type: ignore[index]
    differences: list[str] = []
    for missing in sorted(before_files.keys() - after_files.keys()):
        differences.append(f"removed: {missing}")
    for added in sorted(after_files.keys() - before_files.keys()):
        differences.append(f"added: {added}")
    for common in sorted(before_files.keys() & after_files.keys()):
        if before_files[common] != after_files[common]:
            differences.append(f"changed: {common}")
    return differences
