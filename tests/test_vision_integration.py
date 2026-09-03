from __future__ import annotations

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
    extract_candidate,
    measure_temporal_features,
    prepare_roi,
    read_stack_or_avi_composite,
    TemporalFeatures,
)
from fireball_edge.offline.model_tools import quantize_static_qdq, write_model_manifest


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
            roi_logit = float(tensor.mean())
            score = max(0.5, min(0.9, 0.7 + roi_logit * 0.01))
            return score, roi_logit, {"roi_logit": roi_logit}

        def decision(self, score: float) -> str:
            return "possible_fireball" if score >= 0.5 else "no_alert"

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.source = self.root / "readonly-captures"
        self.source.mkdir()
        self.base = self.source / "M20260902_010203_CAM01"
        self._process_count = 0
        self._write_bundle()

    def _write_bundle(self) -> None:
        # Use the real UFOCapture HD geometry: this catches accidental
        # coordinate/shape assumptions that tiny synthetic frames conceal.
        width, height = 1920, 1080
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
            start_x = 320 + frame_index * 30
            cv2.line(frame, (start_x, 540), (start_x + 180, 560), (255, 255, 255), 4)
            peak = np.maximum(peak, frame)
            writer.write(frame)
        writer.release()
        self.assertTrue(cv2.imwrite(str(self.source / f"{self.base.name}P.bmp"), peak))
        mask = np.zeros_like(peak)
        cv2.line(mask, (300, 540), (800, 560), (0, 0, 255), 4)
        self.assertTrue(cv2.imwrite(str(self.source / f"{self.base.name}M.bmp"), mask))
        self.base.with_suffix(".xml").write_text(
            '<ufocapture_record timestamp="2026-09-02T01:02:03.125" tz="10800" '
            'width="1920" height="1080" fps="10" framecount="8" codec="MJPG" '
            'droppedframes="0" countrycode="PL" lid="1" sid="2" cam="3" lens="4"/>'
        )

    def _process(self) -> dict[str, object]:
        self._process_count += 1
        state = self.root / f"external-state-{self._process_count}"
        config = EdgeConfig(
            state_root=state,
            monitored_roots=(self.source,),
            model_manifest=state / "models" / "active" / "model-manifest.json",
        )
        queue = EventQueue(state)
        queue.enqueue(self.base)
        claimed = queue.claim_next()
        assert claimed is not None
        return EventProcessor(config, model=self.FakeModel())(claimed)

    def _assert_same_path(self, expected: str | Path, actual: str | Path) -> None:
        self.assertTrue(
            Path(expected).samefile(Path(actual)),
            f"paths do not identify the same file: {expected!s} != {actual!s}",
        )

    def _assert_same_paths(
        self, expected: list[str | Path], actual: list[str | Path]
    ) -> None:
        self.assertEqual(len(expected), len(actual))
        for expected_path, actual_path in zip(expected, actual, strict=True):
            self._assert_same_path(expected_path, actual_path)

    def test_avi_candidate_temporal_features_and_stack_roi(self) -> None:
        bundle = discover_bundle(self.base)
        extraction = extract_candidate(bundle)
        self.assertEqual("avi_sequential_difference", extraction.region.source)
        self.assertGreater(extraction.region.changed_pixels, 12)
        temporal = measure_temporal_features(bundle.avi, extraction.region)
        self.assertEqual(8, temporal.frame_count)
        self.assertGreater(temporal.duration_seconds, 0)
        self.assertGreater(temporal.motion_pixels, 0)
        stack = read_stack_or_avi_composite(bundle)
        self._assert_same_path(self.source / f"{self.base.name}P.bmp", stack.selected_path)
        tensor = prepare_roi(stack.image, extraction.region)
        self.assertEqual((1, 3, 224, 224), tensor.shape)
        self.assertEqual(np.float32, tensor.dtype)

    def test_star_mask_change_corruption_and_removal_do_not_affect_scoring(self) -> None:
        bundle = discover_bundle(self.base)
        baseline_regions = tuple(region.as_dict() for region in extract_candidate(bundle).regions)
        baseline_temporal = measure_temporal_features(bundle.avi, extract_candidate(bundle).region)
        baseline = self._process()

        mask_path = self.source / f"{self.base.name}M.bmp"
        mask_path.write_bytes(b"not an image")
        corrupt_bundle = discover_bundle(self.base)
        self.assertEqual(baseline_regions, tuple(region.as_dict() for region in extract_candidate(corrupt_bundle).regions))
        self.assertEqual(baseline_temporal.as_dict(), measure_temporal_features(corrupt_bundle.avi, extract_candidate(corrupt_bundle).region).as_dict())
        corrupt = self._process()
        self.assertEqual(baseline["calibrated_score"], corrupt["calibrated_score"])
        self.assertEqual(baseline["candidates"], corrupt["candidates"])
        self.assertFalse(
            any(mask_path.samefile(path) for path in corrupt["scoring_sidecars"])
        )
        self.assertEqual("provenance_only", corrupt["star_mask_role"])

        mask_path.unlink()
        missing_bundle = discover_bundle(self.base)
        self.assertEqual(baseline_regions, tuple(region.as_dict() for region in extract_candidate(missing_bundle).regions))
        self.assertEqual("absent", self._process()["star_mask_role"])

    def test_static_bright_object_is_not_temporal_activity(self) -> None:
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

    def test_missing_stack_uses_avi_composite_and_missing_xml_is_nonfatal(self) -> None:
        (self.source / f"{self.base.name}M.bmp").unlink()
        (self.source / f"{self.base.name}P.bmp").unlink()
        self.base.with_suffix(".xml").unlink()
        bundle = discover_bundle(self.base)
        stack = read_stack_or_avi_composite(bundle)
        self.assertIsNone(stack.selected_path)
        self.assertEqual("avi_maximum_composite", stack.source)
        self.assertEqual((1080, 1920), stack.image.shape[:2])
        result = self._process()
        self.assertEqual("absent", result["xml_role"])
        self.assertEqual("absent", result["xml_validation"])
        self._assert_same_paths([bundle.avi], result["scoring_sidecars"])

    def test_bmp_is_preferred_and_jpg_is_fallback_after_bmp_corruption(self) -> None:
        bmp_path = self.source / f"{self.base.name}P.bmp"
        jpg_path = self.source / f"{self.base.name}P.jpg"
        self.assertTrue(cv2.imwrite(str(jpg_path), np.full((1080, 1920, 3), 17, dtype=np.uint8)))
        self._assert_same_path(
            bmp_path,
            read_stack_or_avi_composite(discover_bundle(self.base)).selected_path,
        )
        bmp_path.write_bytes(b"corrupt bmp")
        bundle = discover_bundle(self.base)
        self._assert_same_path(jpg_path, read_stack_or_avi_composite(bundle).selected_path)
        self.assertTrue(any(bmp_path.samefile(path) for path in bundle.source_files()))
        self.assertTrue(any(jpg_path.samefile(path) for path in bundle.source_files()))

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
        self.assertEqual("validation_only", result["xml_role"])
        self.assertEqual("valid", result["xml_validation"])
        self.assertEqual("provenance_only", result["star_mask_role"])
        self._assert_same_paths(
            [self.base.with_suffix(".avi"), self.source / f"{self.base.name}P.bmp"],
            result["scoring_sidecars"],
        )
        mask_path = self.source / f"{self.base.name}M.bmp"
        self.assertTrue(
            any(mask_path.samefile(path) for path in result["source_provenance"])
        )
        self.assertFalse(
            any(mask_path.samefile(path) for path in result["scoring_sidecars"])
        )
        self.assertTrue(Path(result["annotated_image"]).is_file())
        result_path = state / "results" / "v2" / queued.event_id / "result.json"
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
