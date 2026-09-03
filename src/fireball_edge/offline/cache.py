"""External normalized-ROI cache generation from labeled bundles."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping

from ..artifacts import write_json_atomic
from ..bundles import EventBundle
from ..contracts import CANDIDATE_EXTRACTOR, SCHEMA_VERSION
from ..inference import sha256_file
from ..vision import _vision_deps, extract_candidate, measure_temporal_features, prepare_roi
from .manifest import ManifestRecord
from .ufocapture import preflight_bundle


def build_roi_cache(
    records: Iterable[ManifestRecord | Mapping[str, object]],
    cache_root: str | Path,
    *,
    manifest_sha256: str,
) -> Path:
    """Write NCHW arrays externally; never create a crop beside source media."""

    import numpy as np

    root = Path(cache_root).resolve(strict=False)
    arrays = root / "arrays"
    arrays.mkdir(parents=True, exist_ok=True)
    manifest_index: list[dict[str, object]] = []
    for record in records:
        document = asdict(record) if isinstance(record, ManifestRecord) else dict(record)
        bundle = EventBundle(
            clip_base=Path(str(document["clip_base"])),
            avi=Path(str(document["avi"])),
            stack_image=(
                Path(str(document["stack_image"]))
                if document.get("stack_image")
                else None
            ),
            star_mask=(
                Path(str(document["star_mask"])) if document.get("star_mask") else None
            ),
            xml=Path(str(document["xml"])) if document.get("xml") else None,
        )
        checked = preflight_bundle(bundle)
        if checked.bundle.stack_image != bundle.stack_image:
            raise ValueError(
                "selected stack differs from immutable manifest; rebuild the manifest"
            )
        bundle = checked.bundle
        assert bundle.avi is not None and bundle.stack_image is not None
        expected_hashes = document.get("source_sha256")
        if not isinstance(expected_hashes, dict):
            raise ValueError("manifest record is missing source hashes; rebuild manifest")
        for source_path in (bundle.avi, bundle.stack_image, bundle.xml):
            if source_path is None or expected_hashes.get(str(source_path)) != sha256_file(source_path):
                raise ValueError(
                    "AVI, selected stack, or XML changed after manifest creation; rebuild manifest"
                )
        extraction = extract_candidate(bundle)
        cv2, _ = _vision_deps()
        image = cv2.imread(str(bundle.stack_image), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"training stack became unreadable: {bundle.stack_image}")
        image_height, image_width = image.shape[:2]
        event_id = str(document["event_id"])
        candidates: list[dict[str, object]] = []
        for candidate_index, region in enumerate(extraction.regions):
            temporal = measure_temporal_features(bundle.avi, region)
            array = prepare_roi(image, region)[0]
            destination = arrays / f"{event_id}-{candidate_index}.npy"
            temporary = arrays / f".{event_id}-{candidate_index}.staging.npy"
            try:
                with temporary.open("wb") as output:
                    np.save(output, array, allow_pickle=False)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            candidates.append(
                {
                    "roi_npy": str(destination),
                    "roi_sha256": sha256_file(destination),
                    "roi": region.as_dict(),
                    "image_geometry": {
                        "width": image_width,
                        "height": image_height,
                    },
                    "temporal_features": temporal.as_dict(),
                    "candidate_source": region.source,
                    "candidate_extractor": CANDIDATE_EXTRACTOR,
                    "source_identity": {
                        role: identity
                        for role, identity in bundle.source_identity().items()
                        if role in {"avi", "stack_image"}
                    },
                }
            )
        document.update(
            {
                "candidates": candidates,
                "training_objective": "multi_instance_event_label",
                "event_aggregation": "maximum_calibrated_candidate_score",
                "source_identity": bundle.source_identity(),
            }
        )
        manifest_index.append(document)
    return write_json_atomic(
        root / "index.json",
        {
            "schema_version": SCHEMA_VERSION,
            "candidate_extractor": CANDIDATE_EXTRACTOR,
            "manifest_sha256": manifest_sha256,
            "records": manifest_index,
        },
    )
