from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fireball_edge.artifacts import read_committed_result, write_json_atomic
from fireball_edge.config import EdgeConfig
from fireball_edge.contracts import CANDIDATE_EXTRACTOR, SCHEMA_VERSION
from fireball_edge.inference import (
    ModelManifest,
    ModelPackageError,
    OnnxCandidateModel,
    calibration_features,
    sha256_file,
)
from fireball_edge.offline.manifest import read_manifest
from fireball_edge.offline.model_tools import event_max_score, write_model_manifest
from fireball_edge.offline.orchestrator import run_training
from fireball_edge.offline.training import CachedRoiDataset
from fireball_edge.vision import CandidateRegion, TemporalFeatures

try:
    import cv2
    import numpy as np
    import onnx
    import onnxruntime
    import torch
except ImportError:  # pragma: no cover - base runtime install omits training extras
    cv2 = np = onnx = onnxruntime = torch = None


class V1BoundaryRejectionTests(unittest.TestCase):
    def test_v1_manifest_and_cache_require_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"schema_version": 1, "records": []}))
            with self.assertRaisesRegex(ValueError, "rebuild"):
                read_manifest(manifest)

            cache = root / "cache.json"
            cache.write_text(json.dumps({"schema_version": 1, "records": []}))
            with self.assertRaisesRegex(ValueError, "rebuild"):
                CachedRoiDataset(cache, {"fold_1"})

    def test_v1_model_and_cached_result_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model = root / "candidate.onnx"
            model.write_bytes(b"synthetic")
            manifest = root / "model-manifest.json"
            write_model_manifest(
                manifest,
                model_path=model,
                model_version="v2-test",
                feature_order=["roi_logit", "duration_seconds"],
                coefficients=[1.0, 0.0],
                intercept=0.0,
                possible_threshold=0.2,
                probable_threshold=0.8,
                quantization="fp32",
            )
            document = json.loads(manifest.read_text(encoding="utf-8"))
            document["schema_version"] = 1
            manifest.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ModelPackageError, "schema_version"):
                ModelManifest.load(manifest)

            annotation = root / "annotated.jpg"
            annotation.write_bytes(b"image")
            result_path = root / "result.json"
            cached_result = {
                "schema_version": 1,
                "candidate_extractor": CANDIDATE_EXTRACTOR,
                "event_id": "event",
                "clip_base": "/capture/event",
                "model_version": "model",
                "model_sha256": "model-hash",
                "model_manifest_sha256": "manifest-hash",
                "source_identity": {},
                "annotated_image": str(annotation),
            }
            write_json_atomic(result_path, cached_result)
            self.assertIsNone(
                read_committed_result(
                    result_path,
                    event_id="event",
                    clip_base="/capture/event",
                    model_version="model",
                    model_sha256="model-hash",
                    model_manifest_sha256="manifest-hash",
                    source_identity={},
                )
            )

    def test_v2_cache_rejects_a_tampered_roi_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            roi = root / "roi.npy"
            roi.write_bytes(b"before")
            cache = root / "index.json"
            cache.write_text(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "candidate_extractor": CANDIDATE_EXTRACTOR,
                        "manifest_sha256": "11" * 32,
                        "records": [
                            {
                                "partition": "fold_1",
                                "candidates": [
                                    {
                                        "roi_npy": str(roi),
                                        "roi_sha256": sha256_file(roi),
                                        "roi": {},
                                        "image_geometry": {},
                                        "temporal_features": {},
                                        "candidate_source": "avi_sequential_difference",
                                        "candidate_extractor": CANDIDATE_EXTRACTOR,
                                        "source_identity": {
                                            "avi": {},
                                            "stack_image": {},
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            roi.write_bytes(b"after")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                CachedRoiDataset(cache, {"fold_1"})


class FeatureAndAggregationTests(unittest.TestCase):
    def test_governance_metadata_and_star_mask_fields_are_not_features(self) -> None:
        region = CandidateRegion(1, 2, 30, 10, 42, 100.0, "avi_sequential_difference")
        temporal = TemporalFeatures(
            8, 25.0, 4, 0.16, 12.0, 0.9, 0.01, 4.0, 8.0, 0.2, 0.5
        )
        features = calibration_features(0.3, region, temporal, 1920, 1080)
        forbidden = {
            "station",
            "camera",
            "night",
            "camera_night",
            "countrycode",
            "lid",
            "sid",
            "cam",
            "lens",
            "latitude",
            "longitude",
            "xml",
            "star_mask",
            "map_background_brightness",
            "map_brightness_above_background",
        }
        self.assertTrue(forbidden.isdisjoint(features))
        self.assertEqual(15, len(features))

    def test_event_score_is_maximum_candidate_score(self) -> None:
        score = event_max_score([[0.0, 100.0], [2.0, -100.0]], 0.0, [1.0, 0.0])
        self.assertGreater(score, 0.88)

    def test_shared_contract_is_v2(self) -> None:
        self.assertEqual(2, SCHEMA_VERSION)
        self.assertEqual("avi-diff-stack-v2", CANDIDATE_EXTRACTOR)


@unittest.skipIf(
    any(item is None for item in (cv2, np, onnx, onnxruntime, torch)),
    "training and ONNX dependencies are not installed",
)
class TrainingOrchestratorSmokeTests(unittest.TestCase):
    @staticmethod
    def _model_factory():
        return torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(3, 1),
        )

    def _write_bundle(self, source: Path, index: int, *, locked: bool) -> Path:
        base = source / f"M20260903_01020{index}_CAM01"
        width, height = 64, 48
        writer = cv2.VideoWriter(
            str(base.with_suffix(".avi")),
            cv2.VideoWriter_fourcc(*"MJPG"),
            10.0,
            (width, height),
        )
        if not writer.isOpened():
            self.skipTest("OpenCV MJPG writer is unavailable")
        stack = np.zeros((height, width, 3), dtype=np.uint8)
        for frame_index in range(5):
            frame = np.zeros_like(stack)
            start = 6 + frame_index * 5
            cv2.rectangle(frame, (start, 18), (start + 12, 24), (255, 255, 255), -1)
            stack = np.maximum(stack, frame)
            writer.write(frame)
        writer.release()
        self.assertTrue(cv2.imwrite(str(source / f"{base.name}P.bmp"), stack))
        star_mask = np.zeros_like(stack)
        star_mask[:, :, 0] = (index + 1) * 10
        self.assertTrue(cv2.imwrite(str(source / f"{base.name}M.bmp"), star_mask))
        day = 4 if locked else 3
        base.with_suffix(".xml").write_text(
            f'<ufocapture_record y="2026" mo="9" d="{day}" h="1" m="2" '
            's="03.125" tz="10800" cx="64" cy="48" fps="10" frames="5" '
            'fourcc="MJPG" drop="0" countrycode="PL" lid="1" sid="2" '
            'cam="3" lens="4"/>',
            encoding="utf-8",
        )
        return base

    def test_miniature_grouped_training_exports_a_runtime_loadable_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "captures"
            source.mkdir()
            rows = []
            labels = ("fireball", "fireball", "non_fireball", "non_fireball", "fireball", "non_fireball")
            for index, label in enumerate(labels):
                base = self._write_bundle(source, index, locked=index >= 4)
                rows.append(f"{base},{label},physical-{index},")
            labels_path = root / "expert.csv"
            labels_path.write_text(
                "clip_base,label,physical_event_id,nuisance_tags\n"
                + "\n".join(rows)
                + "\n",
                encoding="utf-8",
            )
            state = root / "state"
            active_manifest = state / "models" / "active" / "model-manifest.json"
            config = EdgeConfig(
                state_root=state,
                monitored_roots=(source,),
                model_manifest=active_manifest,
            )
            report_path = run_training(
                config,
                labels_path=labels_path,
                dataset_name="mini",
                model_name="tiny",
                locked_nights={"2026-09-03"},
                locked_cameras=set(),
                folds=2,
                epochs=1,
                batch_size=2,
                learning_rate=1e-3,
                seed=7,
                model_factory=self._model_factory,
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(report["activation"]["performed"])
            self.assertFalse(active_manifest.exists())
            self.assertTrue(report["source_immutability"]["verified"])

            oof_path = state / "training" / "mini" / "tiny" / "oof-predictions-v2.json"
            oof = json.loads(oof_path.read_text(encoding="utf-8"))
            self.assertEqual("maximum_calibrated_candidate_score", oof["event_aggregation"])
            physical_partitions: dict[str, set[str]] = {}
            for prediction in oof["predictions"]:
                physical_partitions.setdefault(prediction["physical_event_id"], set()).add(
                    prediction["fold"]
                )
            self.assertTrue(all(len(partitions) == 1 for partitions in physical_partitions.values()))

            cache_path = state / "datasets" / "mini" / "roi-cache-v2" / "index.json"
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            candidate = cache["records"][0]["candidates"][0]
            self.assertIn("roi_sha256", candidate)
            self.assertIn("image_geometry", candidate)
            self.assertIn("temporal_features", candidate)
            self.assertEqual("avi_sequential_difference", candidate["candidate_source"])

            manifest = ModelManifest.load(report["model_manifest"])
            runtime = OnnxCandidateModel(manifest, max_threads=1)
            self.assertEqual(["CPUExecutionProvider"], runtime.session.get_providers())
            model_manifest_mtime = manifest.manifest_path.stat().st_mtime_ns
            resumed_report = run_training(
                config,
                labels_path=labels_path,
                dataset_name="mini",
                model_name="tiny",
                locked_nights={"2026-09-03"},
                locked_cameras=set(),
                resume=True,
                folds=2,
                epochs=1,
                batch_size=2,
                learning_rate=1e-3,
                seed=7,
                model_factory=self._model_factory,
            )
            self.assertEqual(report_path, resumed_report)
            self.assertEqual(model_manifest_mtime, manifest.manifest_path.stat().st_mtime_ns)


if __name__ == "__main__":
    unittest.main()
