"""Expert-label manifest construction for complete UFOCapture bundles."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from ..artifacts import write_json_atomic
from ..bundles import discover_bundle, normalize_clip_base
from ..contracts import CANDIDATE_EXTRACTOR, SCHEMA_VERSION
from ..inference import sha256_file
from .ufocapture import BundlePreflightError, preflight_bundle


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
    stack_image: str
    star_mask: str | None
    xml: str
    star_mask_role: str
    xml_validation: str
    capture_metadata: dict[str, object]
    metadata_warnings: tuple[str, ...]
    source_identity: dict[str, dict[str, int | str]] = field(default_factory=dict)
    source_sha256: dict[str, str] = field(default_factory=dict)


def _event_id(physical_event_id: str, station: str, camera: str, clip_base: Path) -> str:
    import hashlib

    key = "\0".join((physical_event_id, station, camera, str(clip_base)))
    return f"obs_{hashlib.sha256(key.encode('utf-8')).hexdigest()}"


def load_expert_labels(path: str | Path) -> list[dict[str, str]]:
    label_path = Path(path).expanduser().resolve(strict=True)
    with label_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"clip_base", "label", "physical_event_id", "nuisance_tags"}
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
        required_values = ("clip_base", "physical_event_id")
        empty = [name for name in required_values if not row[name].strip()]
        if empty:
            raise ValueError(f"row {row_number}: empty values: {', '.join(empty)}")
        bundle = discover_bundle(row["clip_base"])
        try:
            preflight = preflight_bundle(bundle)
        except BundlePreflightError as exc:
            raise ValueError(f"row {row_number}: {exc}") from exc
        bundle = preflight.bundle
        metadata = preflight.metadata
        assertions = {
            "station": metadata.station,
            "camera": metadata.camera,
            "night": metadata.camera_night,
        }
        for name, derived in assertions.items():
            supplied = row.get(name, "").strip()
            if supplied and supplied != derived:
                raise ValueError(
                    f"row {row_number}: {name} assertion {supplied!r} does not match XML-derived {derived!r}"
                )
        observation_key = (
            row["physical_event_id"].strip(),
            metadata.station,
            metadata.camera,
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
                station=metadata.station,
                camera=metadata.camera,
                night=metadata.camera_night,
                label=label,
                nuisance_tags=tags,
                avi=str(bundle.avi),
                stack_image=str(bundle.stack_image),
                star_mask=str(bundle.star_mask) if bundle.star_mask else None,
                xml=str(bundle.xml),
                star_mask_role="provenance_only",
                xml_validation="valid",
                capture_metadata=metadata.as_dict(),
                metadata_warnings=preflight.warnings,
                source_identity=bundle.source_identity(),
                source_sha256={
                    path: sha256_file(path) for path in bundle.source_files()
                },
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
        required_sources = {record["avi"], record["stack_image"], record["xml"]}
        if not required_sources.issubset(record["source_sha256"]):
            raise ValueError("manifest records must hash AVI, selected stack, and XML")
        groups.setdefault(record["physical_event_id"], set()).add(record["label"])
    conflicts = [group for group, labels in groups.items() if len(labels) > 1]
    if conflicts:
        raise ValueError(
            "physical events have conflicting labels: " + ", ".join(sorted(conflicts))
        )
    return write_json_atomic(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "candidate_extractor": CANDIDATE_EXTRACTOR,
            "grouping_key": "physical_event_id",
            "label_source": "human_expert",
            "records": serialized,
        },
    )


def read_manifest(path: str | Path) -> list[ManifestRecord]:
    """Load only an exact schema-v2 manifest; v1 datasets must be rebuilt."""

    with Path(path).open("r", encoding="utf-8") as source:
        document = json.load(source)
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != SCHEMA_VERSION
        or document.get("candidate_extractor") != CANDIDATE_EXTRACTOR
    ):
        raise ValueError("unsupported manifest schema or candidate extractor; rebuild with v2")
    raw_records = document.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("manifest records must be a list")
    try:
        records = [
            ManifestRecord(
                **{
                    **item,
                    "nuisance_tags": tuple(item.get("nuisance_tags", [])),
                    "metadata_warnings": tuple(item.get("metadata_warnings", [])),
                }
            )
            for item in raw_records
        ]
        for record in records:
            required_sources = {record.avi, record.stack_image, record.xml}
            if not required_sources.issubset(record.source_sha256):
                raise ValueError("manifest record is missing required source hashes")
        return records
    except (TypeError, AttributeError) as exc:
        raise ValueError(f"invalid manifest record: {exc}") from exc


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
