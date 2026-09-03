"""Production edge cascade, deliberately isolated from all legacy models."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .artifacts import read_committed_result, write_image_atomic, write_json_atomic
from .bundles import discover_bundle
from .config import EdgeConfig
from .contracts import CANDIDATE_EXTRACTOR, SCHEMA_VERSION
from .inference import OnnxCandidateModel, load_model
from .offline.ufocapture import validate_xml_provenance
from .queue import QueueEvent
from .vision import (
    InvalidMediaError,
    extract_candidate,
    measure_temporal_features,
    prepare_roi,
    read_stack_or_avi_composite,
    write_annotated_image,
)


RESULT_SCHEMA_VERSION = SCHEMA_VERSION


class EventProcessor:
    """Resolve, measure, infer, and atomically publish one queued event."""

    def __init__(self, config: EdgeConfig, model: OnnxCandidateModel | None = None) -> None:
        if config.model_manifest is None:
            raise ValueError("model_manifest is required by the edge worker")
        self.config = config
        self.model = model or load_model(
            config.model_manifest, max_threads=config.max_inference_threads
        )

    def __call__(self, event: QueueEvent) -> dict[str, Any]:
        started = time.perf_counter()
        clip_base = str(self.config.validate_clip_base(event.clip_base))
        bundle = discover_bundle(clip_base)
        if bundle.avi is None:
            raise InvalidMediaError("AVI is required for temporal fireball features")
        source_identity = bundle.source_identity()

        # Keep v1 artifacts intact while v2 rebuilds an event with the same
        # stable ID. Schema-versioned result directories are never migrated or
        # overwritten in place.
        output_directory = self.config.state_root / "results" / "v2" / event.event_id
        result_path = output_directory / "result.json"
        cached = read_committed_result(
            result_path,
            event_id=event.event_id,
            clip_base=clip_base,
            model_version=self.model.manifest.model_version,
            model_sha256=self.model.manifest.model_sha256,
            model_manifest_sha256=self.model.manifest.manifest_sha256,
            source_identity=source_identity,
        )
        if cached is not None:
            return cached

        # Candidate locations and all temporal values come solely from the
        # AVI.  Classifier pixels come from a valid P stack, or a maximum AVI
        # composite at runtime if the stack is unavailable.  In particular,
        # M.bmp is not opened anywhere in the scoring path.
        extraction = extract_candidate(bundle)
        stack = read_stack_or_avi_composite(bundle)
        representative = stack.image
        height, width = representative.shape[:2]
        scored_candidates: list[dict[str, Any]] = []
        for region in extraction.regions:
            temporal = measure_temporal_features(bundle.avi, region)
            input_tensor = prepare_roi(
                representative, region, size=self.model.manifest.image_size
            )
            calibrated_score, roi_logit, calibration_values = self.model.score(
                input_tensor, region, temporal, width, height
            )
            scored_candidates.append(
                {
                    "calibrated_score": calibrated_score,
                    "roi_model_logit": roi_logit,
                    "roi": region.as_dict(),
                    "temporal_features": temporal.as_dict(),
                    "calibration_features": calibration_values,
                    "_region": region,
                }
            )
        primary = max(scored_candidates, key=lambda item: item["calibrated_score"])
        region = primary.pop("_region")
        for candidate in scored_candidates:
            candidate.pop("_region", None)
        calibrated_score = float(primary["calibrated_score"])
        decision = self.model.decision(calibrated_score)

        annotated_path = output_directory / "annotated.jpg"
        write_image_atomic(
            annotated_path,
            lambda temporary: write_annotated_image(
                representative, region, decision, calibrated_score, temporary
            ),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        capture_to_result_ms = max(0.0, (time.time() - event.created_at) * 1000.0)
        result: dict[str, Any] = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "event_id": event.event_id,
            "clip_base": clip_base,
            "decision": decision,
            "calibrated_score": calibrated_score,
            "roi_model_logit": primary["roi_model_logit"],
            "roi": region.as_dict(),
            "temporal_features": primary["temporal_features"],
            "calibration_features": primary["calibration_features"],
            "candidates": scored_candidates,
            "candidate_extraction": {
                "source": region.source,
                "candidate_count": len(extraction.regions),
            },
            "model_version": self.model.manifest.model_version,
            "model_sha256": self.model.manifest.model_sha256,
            "model_manifest_sha256": self.model.manifest.manifest_sha256,
            "quantization": self.model.manifest.quantization,
            "candidate_extractor": CANDIDATE_EXTRACTOR,
            "source_identity": source_identity,
            # Source provenance lists the complete capture bundle.  Scoring
            # provenance is intentionally narrower: AVI plus a selected P
            # stack only.  When the stack is reconstructed, AVI is the sole
            # scored sidecar and appears once.
            "source_provenance": bundle.source_files(),
            "scoring_sidecars": list(
                dict.fromkeys(
                    str(path)
                    for path in (bundle.avi, stack.selected_path)
                    if path is not None
                )
            ),
            "stack_image_source": stack.source,
            "star_mask_role": "provenance_only" if bundle.star_mask else "absent",
            "xml_role": "validation_only" if bundle.xml is not None else "absent",
            "xml_validation": validate_xml_provenance(bundle.xml),
            "processing_time_ms": elapsed_ms,
            "capture_to_result_ms": capture_to_result_ms,
            "scientific_status": "uncalibrated",
            "annotated_image": str(annotated_path),
        }
        write_json_atomic(result_path, result)
        return result
