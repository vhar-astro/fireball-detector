from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - base environment intentionally stays light
    cv2 = np = None

try:
    import onnx
    import onnxruntime
except ImportError:  # pragma: no cover - runtime dependencies are optional for base tests
    onnx = onnxruntime = None

from fireball_edge.bundles import discover_bundle
from fireball_edge.config import EdgeConfig
from fireball_edge.offline.manifest import compare_snapshots, snapshot_tree
from fireball_edge.pipeline import EventProcessor
from fireball_edge.inference import ModelManifest, OnnxCandidateModel
from fireball_edge.queue import EventQueue
from fireball_edge.worker import EdgeWorker
from fireball_edge.vision import (
    CandidateRegion,
    InvalidMediaError,
    candidate_from_change_map,
    extract_candidate,
    measure_temporal_features,
    prepare_roi,
    read_peak_or_avi_frame,
    TemporalFeatures,
)
from fireball_edge.offline.model_tools import quantize_static_qdq, write_model_manifest
from fireball_edge.offline.cache import build_roi_cache


@unittest.skipIf(cv2 is None or np is None, "vision dependencies are not installed")
class VisionIntegrationTests(unittest.TestCase):
    class FakeModel:
        def __init__(self) -> None:
            self.manifest = SimpleNamespace(
                model_version="synthetic-v1",
                model_sha256="synthetic-hash",
                manifest_sha256="synthetic-manifest-hash",
                image_size=224,
                quantization="fp32",
            )

        def score(self, tensor, region, temporal, width, height):
            assert tensor.shape == (1, 3, 224, 224)
            return 0.70, 1.25, {"roi_logit": 1.25}

        def decision(self, score: float) -> str:
            return "possible_fireball" if score >= 0.5 else "no_alert"

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.source = self.root / "readonly-captures"
        self.source.mkdir()
        self.base = self.source / "M20260902_010203_CAM01"
        self._write_bundle()

    def _write_bundle(self) -> None:
        width, height = 96, 64
        writer = cv2.VideoWriter(
            str(self.base.with_suffix(".avi")),
            cv2.VideoWriter_fourcc(*"MJPG"),
            10.0,
            (width, height),
        )
        if not writer.isOpened():
            self.skipTest("OpenCV MJPG writer is unavailable")
        peak = np.zeros((height, width, 3), dtype=np.uint8)
        for frame_index in range(8):
            frame = np.zeros_like(peak)
            start_x = 16 + frame_index * 3
            cv2.line(frame, (start_x, 30), (start_x + 18, 32), (255, 255, 255), 2)
            peak = np.maximum(peak, frame)
            writer.write(frame)
        writer.release()
        self.assertTrue(cv2.imwrite(str(self.source / f"{self.base.name}P.bmp"), peak))
        change_map = np.zeros_like(peak)
        cv2.line(change_map, (14, 30), (58, 32), (0, 0, 255), 2)
        self.assertTrue(cv2.imwrite(str(self.source / f"{self.base.name}M.bmp"), change_map))
        self.base.with_suffix(".xml").write_text("<ufo><station>test</station></ufo>")

    def test_red_channel_candidate_temporal_features_and_letterbox(self) -> None:
        bundle = discover_bundle(self.base)
        extraction = extract_candidate(bundle)
        self.assertTrue(extraction.used_change_map)
        self.assertEqual("change_map_red_channel", extraction.region.source)
        self.assertGreater(extraction.region.changed_pixels, 12)
        temporal = measure_temporal_features(bundle.avi, extraction.region)
        self.assertEqual(8, temporal.frame_count)
        self.assertGreater(temporal.duration_seconds, 0)
        self.assertGreater(temporal.motion_pixels, 0)
        image = read_peak_or_avi_frame(bundle)
        tensor = prepare_roi(image, extraction.region)
        self.assertEqual((1, 3, 224, 224), tensor.shape)
        self.assertEqual(np.float32, tensor.dtype)

    def test_blue_channel_is_not_mistaken_for_change(self) -> None:
        blue_only = np.zeros((32, 32, 3), dtype=np.uint8)
        cv2.line(blue_only, (2, 16), (28, 16), (255, 0, 0), 2)
        path = self.root / "blue.bmp"
        cv2.imwrite(str(path), blue_only)
        with self.assertRaises(InvalidMediaError):
            candidate_from_change_map(path)

    def test_blue_mask_does_not_replace_green_background_channel(self) -> None:
        mapped = np.zeros((32, 32, 3), dtype=np.uint8)
        cv2.line(mapped, (2, 16), (28, 16), (255, 0, 255), 2)
        path = self.root / "red-with-blue-mask.bmp"
        cv2.imwrite(str(path), mapped)
        self.assertGreater(candidate_from_change_map(path).changed_pixels, 12)

    def test_green_background_is_measured_without_suppressing_red_detection(self) -> None:
        mapped = np.zeros((32, 32, 3), dtype=np.uint8)
        cv2.line(mapped, (2, 16), (28, 16), (0, 190, 200), 2)
        path = self.root / "bright-background.bmp"
        cv2.imwrite(str(path), mapped)
        region = candidate_from_change_map(path)
        self.assertGreater(region.map_background_brightness, 0)
        self.assertGreater(region.map_brightness_above_background, 0)

    def test_static_bright_object_is_not_temporal_activity_without_map(self) -> None:
        path = self.root / "static-moon.avi"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48)
        )
        if not writer.isOpened():
            self.skipTest("OpenCV MJPG writer is unavailable")
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        cv2.circle(frame, (32, 24), 8, (255, 255, 255), -1)
        for _ in range(6):
            writer.write(frame)
        writer.release()
        region = CandidateRegion(20, 12, 24, 24, 100, 100.0, "test")
        temporal = measure_temporal_features(path, region)
        self.assertEqual(0, temporal.active_frame_count)

    def test_wrong_size_map_falls_back_to_streamed_avi_difference(self) -> None:
        wrong = np.zeros((16, 16, 3), dtype=np.uint8)
        wrong[:, :, 2] = 255
        cv2.imwrite(str(self.source / f"{self.base.name}M.bmp"), wrong)
        extraction = extract_candidate(discover_bundle(self.base))
        self.assertFalse(extraction.used_change_map)
        self.assertEqual("avi_frame_difference", extraction.region.source)
        self.assertIn("do not match AVI", extraction.fallback_reason)

    def test_missing_map_peak_and_xml_use_avi_without_inventing_provenance(self) -> None:
        (self.source / f"{self.base.name}M.bmp").unlink()
        (self.source / f"{self.base.name}P.bmp").unlink()
        self.base.with_suffix(".xml").unlink()
        bundle = discover_bundle(self.base)
        extraction = extract_candidate(bundle)
        self.assertFalse(extraction.used_change_map)
        self.assertEqual("change map is missing", extraction.fallback_reason)
        self.assertEqual((64, 96), read_peak_or_avi_frame(bundle).shape[:2])

        state = self.root / "external-state"
        config = EdgeConfig(
            state_root=state,
            monitored_roots=(self.source,),
            model_manifest=state / "models" / "active" / "model-manifest.json",
        )
        queue = EventQueue(state)
        claimed = queue.claim_next()
        if claimed is None:
            queue.enqueue(self.base)
            claimed = queue.claim_next()
        result = EventProcessor(config, model=self.FakeModel())(claimed)
        self.assertEqual("absent", result["xml_role"])
        self.assertNotIn(".xml", " ".join(result["sidecars_used"]))

    def test_wrong_size_peak_falls_back_to_streamed_maximum_composite(self) -> None:
        wrong_peak = np.zeros((16, 16, 3), dtype=np.uint8)
        cv2.imwrite(str(self.source / f"{self.base.name}P.bmp"), wrong_peak)
        representative = read_peak_or_avi_frame(discover_bundle(self.base))
        self.assertEqual((64, 96), representative.shape[:2])
        self.assertGreater(int((representative > 0).any(axis=2).sum()), 40)

    def test_positive_event_cache_retains_all_candidates_for_multi_instance_training(self) -> None:
        mapped = np.zeros((64, 96, 3), dtype=np.uint8)
        cv2.line(mapped, (8, 12), (30, 12), (0, 0, 255), 2)
        cv2.line(mapped, (60, 48), (84, 48), (0, 0, 255), 2)
        map_path = self.source / f"{self.base.name}M.bmp"
        cv2.imwrite(str(map_path), mapped)
        bundle = discover_bundle(self.base)
        before = snapshot_tree(self.source)
        index_path = build_roi_cache(
            [
                {
                    "event_id": "obs-test",
                    "clip_base": str(bundle.clip_base),
                    "physical_event_id": "physical-test",
                    "station": "station",
                    "camera": "camera",
                    "night": "2026-09-02",
                    "label": "fireball",
                    "nuisance_tags": ["fireball_with_moon"],
                    "partition": "fold_1",
                    "avi": str(bundle.avi),
                    "peak": str(bundle.peak),
                    "change_map": str(bundle.change_map),
                    "xml": str(bundle.xml),
                }
            ],
            self.root / "external-cache",
        )
        with index_path.open("r", encoding="utf-8") as source:
            cached = json.load(source)["records"][0]
        self.assertGreaterEqual(len(cached["candidates"]), 2)
        self.assertEqual("multi_instance_event_label", cached["training_objective"])
        self.assertEqual([], compare_snapshots(before, snapshot_tree(self.source)))

    def test_corrupt_avi_retries_then_fails_without_touching_source(self) -> None:
        self.base.with_suffix(".avi").write_bytes(b"not an AVI")
        before = snapshot_tree(self.source)
        state = self.root / "external-state"
        config = EdgeConfig(
            state_root=state,
            monitored_roots=(self.source,),
            model_manifest=state / "models" / "active" / "model-manifest.json",
            max_attempts=2,
            retry_initial_seconds=0,
        )
        queue = EventQueue(state)
        event = queue.enqueue(self.base)
        worker = EdgeWorker(queue, config)
        processor = EventProcessor(config, model=self.FakeModel())
        worker.run(processor, once=True)
        self.assertEqual("retry", queue.get(event.event_id).state)
        worker.run(processor, once=True)
        self.assertEqual("failed", queue.get(event.event_id).state)
        self.assertEqual([], compare_snapshots(before, snapshot_tree(self.source)))

    def test_end_to_end_pipeline_keeps_source_tree_byte_identical(self) -> None:
        before = snapshot_tree(self.source)
        state = self.root / "external-state"
        model_manifest = state / "models" / "active" / "model-manifest.json"
        config = EdgeConfig(
            state_root=state,
            monitored_roots=(self.source,),
            model_manifest=model_manifest,
        )
        queue = EventQueue(state)
        queued = queue.enqueue(self.base)
        claimed = queue.claim_next()
        assert claimed is not None
        result = EventProcessor(config, model=self.FakeModel())(claimed)
        queue.complete(claimed.event_id, result)

        self.assertEqual("possible_fireball", result["decision"])
        self.assertEqual("uncalibrated", result["scientific_status"])
        self.assertEqual("provenance_only", result["xml_role"])
        self.assertNotIn(str(self.base.with_suffix(".xml")), result["scoring_sidecars"])
        self.assertTrue(Path(result["annotated_image"]).is_file())
        result_path = state / "results" / queued.event_id / "result.json"
        self.assertTrue(result_path.is_file())
        self.assertEqual([], compare_snapshots(before, snapshot_tree(self.source)))


@unittest.skipIf(onnx is None or onnxruntime is None or np is None, "ONNX dependencies are not installed")
class OnnxRuntimeIntegrationTests(unittest.TestCase):
    def test_cpu_provider_thread_limits_and_real_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            package = Path(temporary_directory)
            model_path = package / "mean.onnx"
            helper = onnx.helper
            graph = helper.make_graph(
                [helper.make_node("ReduceMean", ["image"], ["logit"], axes=[1, 2, 3], keepdims=0)],
                "mean-logit",
                [helper.make_tensor_value_info("image", onnx.TensorProto.FLOAT, [1, 3, 224, 224])],
                [helper.make_tensor_value_info("logit", onnx.TensorProto.FLOAT, [1])],
            )
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
            model.ir_version = 9
            onnx.save(model, model_path)
            manifest_path = package / "model-manifest.json"
            write_model_manifest(
                manifest_path,
                model_path=model_path,
                model_version="ort-test-v1",
                feature_order=["roi_logit", "duration_seconds"],
                coefficients=[1.0, 0.0],
                intercept=0.0,
                possible_threshold=0.4,
                probable_threshold=0.8,
                quantization="fp32",
            )
            runtime = OnnxCandidateModel(ModelManifest.load(manifest_path), max_threads=2)
            options = runtime.session.get_session_options()
            self.assertEqual(2, options.intra_op_num_threads)
            self.assertEqual(1, options.inter_op_num_threads)
            self.assertEqual(
                onnxruntime.ExecutionMode.ORT_SEQUENTIAL, options.execution_mode
            )
            self.assertEqual(["CPUExecutionProvider"], runtime.session.get_providers())
            region = CandidateRegion(0, 0, 10, 2, 20, 100.0, "test")
            temporal = TemporalFeatures(1, 25.0, 1, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            score, logit, values = runtime.score(
                np.zeros((1, 3, 224, 224), dtype=np.float32),
                region,
                temporal,
                224,
                224,
            )
            self.assertEqual(0.0, logit)
            self.assertEqual(0.5, score)
            self.assertEqual({"roi_logit": 0.0, "duration_seconds": 0.04}, values)
            self.assertEqual("possible_fireball", runtime.decision(score))

    def test_static_qdq_quantization_uses_real_calibration_reader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fp32 = root / "conv.onnx"
            int8 = root / "conv-int8.onnx"
            helper = onnx.helper
            weight = helper.make_tensor(
                "weight", onnx.TensorProto.FLOAT, [1, 3, 1, 1], [0.2, 0.3, 0.5]
            )
            bias = helper.make_tensor("bias", onnx.TensorProto.FLOAT, [1], [0.0])
            graph = helper.make_graph(
                [
                    helper.make_node("Conv", ["image", "weight", "bias"], ["features"]),
                    helper.make_node("GlobalAveragePool", ["features"], ["pooled"]),
                    helper.make_node("Flatten", ["pooled"], ["logit"], axis=1),
                ],
                "quantization-test",
                [helper.make_tensor_value_info("image", onnx.TensorProto.FLOAT, [1, 3, 8, 8])],
                [helper.make_tensor_value_info("logit", onnx.TensorProto.FLOAT, [1, 1])],
                [weight, bias],
            )
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
            model.ir_version = 9
            onnx.save(model, fp32)
            calibration = root / "calibration.npy"
            np.save(calibration, np.ones((1, 3, 8, 8), dtype=np.float32))
            quantize_static_qdq(fp32, int8, [calibration])
            self.assertTrue(int8.is_file())
            session = onnxruntime.InferenceSession(
                str(int8), providers=["CPUExecutionProvider"]
            )
            output = session.run(None, {"image": np.ones((1, 3, 8, 8), dtype=np.float32)})[0]
            self.assertEqual((1, 1), output.shape)


if __name__ == "__main__":
    unittest.main()
