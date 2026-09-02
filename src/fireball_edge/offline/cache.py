"""External normalized-ROI cache generation from labeled bundles."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping

from ..artifacts import write_json_atomic
from ..bundles import EventBundle
from ..vision import extract_candidate, prepare_roi, read_peak_or_avi_frame
from .manifest import ManifestRecord


def build_roi_cache(
    records: Iterable[ManifestRecord | Mapping[str, object]], cache_root: str | Path
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
            peak=Path(str(document["peak"])) if document.get("peak") else None,
            change_map=Path(str(document["change_map"])) if document.get("change_map") else None,
            xml=Path(str(document["xml"])) if document.get("xml") else None,
        )
        extraction = extract_candidate(bundle)
        image = read_peak_or_avi_frame(bundle)
        event_id = str(document["event_id"])
        candidates: list[dict[str, object]] = []
        for candidate_index, region in enumerate(extraction.regions):
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
                    "roi": region.as_dict(),
                    "candidate_source": region.source,
                }
            )
        document.update(
            {
                "candidates": candidates,
                "training_objective": "multi_instance_event_label",
            }
        )
        manifest_index.append(document)
    return write_json_atomic(
        root / "index.json", {"schema_version": 1, "records": manifest_index}
    )
