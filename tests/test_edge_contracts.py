from __future__ import annotations

import json
import contextlib
import io
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fireball_edge import artifacts
from fireball_edge.artifacts import read_committed_result, write_json_atomic
from fireball_edge.__main__ import main
from fireball_edge.bundles import discover_bundle
from fireball_edge.contracts import CANDIDATE_EXTRACTOR, SCHEMA_VERSION
from fireball_edge.config import EdgeConfig, StateRootError, load_config
from fireball_edge.event_id import event_id_for_clip_base
from fireball_edge.inference import (
    ModelManifest,
    ModelPackageError,
    OnnxCandidateModel,
    sha256_file,
)
from fireball_edge.notifications import OutboxDispatcher
from fireball_edge.offline.evaluation import (
    Prediction,
    locked_report,
    quantization_gate,
    select_possible_threshold,
    select_probable_threshold,
    write_quantization_report,
)
from fireball_edge.offline.manifest import compare_snapshots, snapshot_tree
from fireball_edge.offline.model_tools import write_model_manifest
from fireball_edge.offline.splits import assign_grouped_partitions
from fireball_edge.offline.manifest import ManifestRecord
from fireball_edge.queue import EventQueue
from fireball_edge.process_lock import process_lock
from fireball_edge.queue import WorkerAlreadyRunningError
from fireball_edge.worker import EdgeWorker


class BundleTests(unittest.TestCase):
    def test_bundle_discovery_and_event_id_accept_each_sidecar_form(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            base = root / "M20260902_010203_CAM01"
            for suffix in (".AVI", "P.BMP", "M.bmp", ".Xml"):
                (root / f"{base.name}{suffix}").write_bytes(b"x")
            bundle = discover_bundle(root / f"{base.name}P.BMP")
            self.assertEqual(base, bundle.clip_base)
            self.assertIsNotNone(bundle.avi)
            self.assertIsNotNone(bundle.stack_image)
            self.assertIsNotNone(bundle.star_mask)
            self.assertIsNotNone(bundle.xml)
            ids = {
                event_id_for_clip_base(base),
                event_id_for_clip_base(root / f"{base.name}.AVI"),
                event_id_for_clip_base(root / f"{base.name}P.BMP"),
                event_id_for_clip_base(root / f"{base.name}M.bmp"),
            }
            self.assertEqual(1, len(ids))


class SourceGuardTests(unittest.TestCase):
    def test_clip_parent_overlap_is_rejected_before_queue_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            capture = Path(temporary_directory) / "capture"
            capture.mkdir()
            state = capture / "state"
            config_path = Path(temporary_directory) / "edge.json"
            config_path.write_text(json.dumps({"state_root": str(state)}))
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(
                    1,
                    main(
                        [
                            "enqueue",
                            "--clip-base",
                            str(capture / "event"),
                            "--config",
                            str(config_path),
                        ]
                    ),
                )
            self.assertFalse(state.exists())


class ModelManifestTests(unittest.TestCase):
    def test_non_finite_runtime_output_and_json_are_rejected(self) -> None:
        class FakeArray:
            def reshape(self, *_):
                return [float("nan")]

        with self.assertRaisesRegex(ModelPackageError, "non-finite"):
            OnnxCandidateModel._roi_logit(FakeArray())
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "result.json"
            with self.assertRaises(ValueError):
                write_json_atomic(destination, {"score": float("nan")})
            self.assertFalse(destination.exists())

    def test_hash_schema_and_calibration_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            package = Path(temporary_directory)
            model = package / "candidate.onnx"
            model.write_bytes(b"synthetic graph bytes")
            manifest = package / "model-manifest.json"
            write_model_manifest(
                manifest,
                model_path=model,
                model_version="synthetic-v1",
                feature_order=["roi_logit", "duration_seconds"],
                coefficients=[1.0, 0.0],
                intercept=0.0,
                possible_threshold=0.2,
                probable_threshold=0.8,
                quantization="fp32",
            )
            loaded = ModelManifest.load(manifest)
            self.assertEqual("synthetic-v1", loaded.model_version)

            model.write_bytes(b"tampered")
            with self.assertRaisesRegex(ModelPackageError, "model hash mismatch"):
                ModelManifest.load(manifest)

    def test_int8_manifest_requires_and_rechecks_target_hardware_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            package = Path(temporary_directory)
            model = package / "candidate.onnx"
            model.write_bytes(b"int8 graph")
            manifest = package / "model-manifest.json"
            with self.assertRaisesRegex(ValueError, "requires passed"):
                write_model_manifest(
                    manifest,
                    model_path=model,
                    model_version="int8-v1",
                    feature_order=["roi_logit", "duration_seconds"],
                    coefficients=[1.0, 0.0],
                    intercept=0.0,
                    possible_threshold=0.2,
                    probable_threshold=0.8,
                    quantization="qdq_int8",
                )
            evidence = {
                "locked_report_file": "locked-report.json",
                "target_cpu": "Intel Core i7-4500U",
                "fp32_recall": 0.96,
                "int8_recall": 0.94,
                "fp32_p95_ms": 100.0,
                "int8_p95_ms": 80.0,
                "ship_int8": True,
            }
            report_path = package / evidence["locked_report_file"]
            write_quantization_report(
                report_path,
                target_cpu=evidence["target_cpu"],
                locked_predictions_sha256="11" * 32,
                fp32_recall=evidence["fp32_recall"],
                int8_recall=evidence["int8_recall"],
                fp32_p95_ms=evidence["fp32_p95_ms"],
                int8_p95_ms=evidence["int8_p95_ms"],
            )
            evidence["locked_report_sha256"] = sha256_file(report_path)
            write_model_manifest(
                manifest,
                model_path=model,
                model_version="int8-v1",
                feature_order=["roi_logit", "duration_seconds"],
                coefficients=[1.0, 0.0],
                intercept=0.0,
                possible_threshold=0.2,
                probable_threshold=0.8,
                quantization="qdq_int8",
                quantization_evidence=evidence,
            )
            with self.assertRaisesRegex(ModelPackageError, "release gate"):
                ModelManifest.load(manifest)
            evidence["int8_recall"] = 0.95
            write_quantization_report(
                report_path,
                target_cpu=evidence["target_cpu"],
                locked_predictions_sha256="11" * 32,
                fp32_recall=evidence["fp32_recall"],
                int8_recall=evidence["int8_recall"],
                fp32_p95_ms=evidence["fp32_p95_ms"],
                int8_p95_ms=evidence["int8_p95_ms"],
            )
            evidence["locked_report_sha256"] = sha256_file(report_path)
            write_model_manifest(
                manifest,
                model_path=model,
                model_version="int8-v1",
                feature_order=["roi_logit", "duration_seconds"],
                coefficients=[1.0, 0.0],
                intercept=0.0,
                possible_threshold=0.2,
                probable_threshold=0.8,
                quantization="qdq_int8",
                quantization_evidence=evidence,
            )
            self.assertEqual("qdq_int8", ModelManifest.load(manifest).quantization)

    def test_config_keeps_model_package_below_external_models_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "edge.json"
            config_path.write_text(
                json.dumps(
                    {
                        "state_root": str(root / "state"),
                        "model_manifest": str(root / "outside" / "model.json"),
                    }
                )
            )
            with self.assertRaises(StateRootError):
                load_config(config_path)


class ArtifactTests(unittest.TestCase):
    def test_binary_fsync_reopens_file_with_windows_writable_descriptor(self) -> None:
        file_handle = mock.MagicMock()
        opened_handle = file_handle.__enter__.return_value
        opened_handle.fileno.return_value = 42
        with (
            mock.patch.object(Path, "open", return_value=file_handle) as open_file,
            mock.patch.object(artifacts.os, "fsync") as fsync,
        ):
            artifacts._fsync_file(Path("annotation.jpg"))
        open_file.assert_called_once_with("r+b")
        fsync.assert_called_once_with(42)

    def test_result_json_is_a_reconcilable_commit_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            annotation = root / "annotated.jpg"
            annotation.write_bytes(b"image")
            result_path = root / "result.json"
            document = {
                "schema_version": SCHEMA_VERSION,
                "event_id": "evt_1",
                "clip_base": "/capture/event",
                "model_version": "v1",
                "model_sha256": "model-hash",
                "model_manifest_sha256": "manifest-hash",
                "source_identity": {
                    "avi": {"path": "/capture/event.avi", "size": 1, "mtime_ns": 2}
                },
                "candidate_extractor": CANDIDATE_EXTRACTOR,
                "annotated_image": str(annotation),
            }
            write_json_atomic(result_path, document)
            self.assertEqual(
                document,
                read_committed_result(
                    result_path,
                    event_id="evt_1",
                    clip_base="/capture/event",
                    model_version="v1",
                    model_sha256="model-hash",
                    model_manifest_sha256="manifest-hash",
                    source_identity=document["source_identity"],
                ),
            )
            self.assertIsNone(
                read_committed_result(
                    result_path,
                    event_id="evt_1",
                    clip_base="/capture/event",
                    model_version="v2",
                    model_sha256="model-hash",
                    model_manifest_sha256="manifest-hash",
                    source_identity=document["source_identity"],
                )
            )

    def test_v1_complete_queue_row_is_archived_requeued_and_not_overwritten(self) -> None:
        for legacy_notification_state in ("pending", "sent"):
            with self.subTest(legacy_notification_state=legacy_notification_state):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    state = root / "state"
                    queue = EventQueue(state)
                    clip_base = root / "captures" / "event"
                    queued = queue.enqueue(clip_base)
                    claimed = queue.claim_next()
                    assert claimed is not None
                    old_image = state / "results" / queued.event_id / "old.jpg"
                    old_image.parent.mkdir(parents=True)
                    old_image.write_bytes(b"legacy image")
                    queue.complete(
                        claimed.event_id,
                        {"schema_version": 1, "candidate_extractor": "change-map-v1"},
                        notification={
                            "destination": "TEST_CHAT_ID",
                            "image_path": str(old_image),
                            "caption": "legacy caption",
                        },
                    )
                    if legacy_notification_state == "sent":
                        with queue._connect() as connection:
                            connection.execute(
                                "UPDATE notification_outbox SET state = 'sent' WHERE event_id = ?",
                                (queued.event_id,),
                            )
                    legacy_result = state / "results" / queued.event_id / "result.json"
                    legacy_result.write_text('{"schema_version": 1}', encoding="utf-8")

                    rebuilt = queue.enqueue(
                        clip_base,
                        required_schema_version=SCHEMA_VERSION,
                        candidate_extractor=CANDIDATE_EXTRACTOR,
                    )
                    self.assertEqual("queued", rebuilt.state)
                    self.assertTrue(legacy_result.is_file())
                    with queue._connect() as connection:
                        archived = connection.execute(
                            "SELECT result_json FROM legacy_event_results WHERE event_id = ?",
                            (queued.event_id,),
                        ).fetchone()
                        archived_notification = connection.execute(
                            """SELECT payload_json FROM legacy_notification_outbox
                               WHERE event_id = ?""",
                            (queued.event_id,),
                        ).fetchone()
                        self.assertIsNone(
                            connection.execute(
                                "SELECT 1 FROM notification_outbox WHERE event_id = ?",
                                (queued.event_id,),
                            ).fetchone()
                        )
                    self.assertIsNotNone(archived)
                    self.assertIsNotNone(archived_notification)
                    self.assertIn("legacy caption", archived_notification["payload_json"])

                    new_image = state / "results" / "v2" / queued.event_id / "new.jpg"
                    new_image.parent.mkdir(parents=True)
                    new_image.write_bytes(b"v2 image")
                    config = EdgeConfig(
                        state_root=state,
                        monitored_roots=(root / "captures",),
                        telegram_enabled=True,
                    )
                    EdgeWorker(queue, config).run(
                        lambda _: {
                            "schema_version": SCHEMA_VERSION,
                            "candidate_extractor": CANDIDATE_EXTRACTOR,
                            "decision": "possible_fireball",
                            "calibrated_score": 0.8,
                            "annotated_image": str(new_image),
                        },
                        once=True,
                    )
                    final = queue.get(queued.event_id)
                    assert final is not None and final.result is not None
                    self.assertEqual(SCHEMA_VERSION, final.result["schema_version"])
                    with queue._connect() as connection:
                        live_notification = connection.execute(
                            """SELECT image_path, caption, state FROM notification_outbox
                               WHERE event_id = ?""",
                            (queued.event_id,),
                        ).fetchone()
                    self.assertEqual(str(new_image), live_notification["image_path"])
                    self.assertNotEqual("legacy caption", live_notification["caption"])
                    self.assertEqual("pending", live_notification["state"])
                    self.assertTrue(legacy_result.is_file())


class NotificationTests(unittest.TestCase):
    class FakeTransport:
        def __init__(self) -> None:
            self.calls: list[dict[str, str]] = []

        def send_photo(self, **kwargs: str) -> str:
            self.calls.append(kwargs)
            return "42"

    def test_completion_and_outbox_are_atomic_and_duplicate_suppressed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            state = root / "state"
            image = state / "results" / "annotation.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"image")
            config = EdgeConfig(
                state_root=state,
                monitored_roots=(root / "clips",),
                telegram_enabled=True,
            )
            queue = EventQueue(state)
            event = queue.enqueue(root / "clips" / "event")
            worker = EdgeWorker(queue, config)
            worker.run(
                lambda _: {
                    "decision": "possible_fireball",
                    "calibrated_score": 0.7,
                    "annotated_image": str(image),
                },
                once=True,
            )
            duplicate = queue.enqueue(root / "clips" / "event")
            self.assertEqual("complete", duplicate.state)

            transport = self.FakeTransport()
            dispatcher = OutboxDispatcher(
                queue,
                config,
                transport=transport,
                environ={
                    config.telegram_token_env: "test-token",
                    config.telegram_chat_id_env: "test-chat",
                },
            )
            self.assertTrue(dispatcher.dispatch_once())
            self.assertFalse(dispatcher.dispatch_once())
            self.assertEqual(1, len(transport.calls))
            self.assertIn(event.event_id, transport.calls[0]["caption"])

    def test_unavailable_telegram_does_not_change_completed_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            state = root / "state"
            image = state / "annotation.jpg"
            state.mkdir()
            image.write_bytes(b"image")
            config = EdgeConfig(
                state_root=state,
                monitored_roots=(root / "clips",),
                telegram_enabled=True,
                telegram_max_attempts=2,
            )
            queue = EventQueue(state)
            event = queue.enqueue(root / "clips" / "event")
            EdgeWorker(queue, config).run(
                lambda _: {
                    "decision": "probable_fireball",
                    "calibrated_score": 0.95,
                    "annotated_image": str(image),
                },
                once=True,
            )
            self.assertEqual("complete", queue.get(event.event_id).state)
            dispatcher = OutboxDispatcher(queue, config, environ={})
            self.assertTrue(dispatcher.dispatch_once())
            self.assertEqual("complete", queue.get(event.event_id).state)
            with queue._connect() as connection:
                outbox = connection.execute(
                    "SELECT state, attempts FROM notification_outbox WHERE event_id = ?",
                    (event.event_id,),
                ).fetchone()
            self.assertEqual(("retry", 1), (outbox["state"], outbox["attempts"]))


class OfflineValidationTests(unittest.TestCase):
    def _predictions(self) -> list[Prediction]:
        return [
            Prediction("p1", "g1", "f1", 1, 0.99, "c1", "n1", ("fireball_with_moon",)),
            Prediction("p2", "g2", "f1", 1, 0.55, "c1", "n1"),
            Prediction("p3", "g3", "f2", 1, 0.60, "c2", "n2", ("saturation",)),
            Prediction("p4", "g4", "f2", 1, 0.70, "c2", "n2"),
            Prediction("n1", "g5", "f1", 0, 0.40, "c1", "n1", ("moon_only",)),
            Prediction("n2", "g6", "f2", 0, 0.10, "c2", "n2", ("ordinary_meteor",)),
        ]

    def test_thresholds_report_and_quantization_release_gate(self) -> None:
        rows = self._predictions()
        possible = select_possible_threshold(
            rows, target_recall=1.0, minimum_fold_recall=1.0
        )
        self.assertEqual(0.55, possible)
        self.assertGreaterEqual(select_probable_threshold(rows), possible)
        report = locked_report(rows, possible)
        self.assertEqual(1.0, report["possible_fireball_recall"])
        self.assertFalse(
            quantization_gate(
                fp32_recall=0.96,
                int8_recall=0.94,
                fp32_p95_ms=100,
                int8_p95_ms=80,
            )["ship_int8"]
        )
        self.assertTrue(
            quantization_gate(
                fp32_recall=0.96,
                int8_recall=0.95,
                fp32_p95_ms=100,
                int8_p95_ms=84,
            )["ship_int8"]
        )

    def test_physical_event_never_crosses_a_partition(self) -> None:
        records: list[ManifestRecord] = []
        for index in range(8):
            records.append(
                ManifestRecord(
                    event_id=f"e{index}",
                    clip_base=f"/source/e{index}",
                    physical_event_id=f"g{index // 2}",
                    station=f"s{index}",
                    camera="locked" if index < 2 else f"c{index}",
                    night=f"n{index}",
                    label="fireball" if index % 2 else "non_fireball",
                    nuisance_tags=(),
                    avi=f"/source/e{index}.avi",
                    stack_image=f"/source/e{index}P.bmp",
                    star_mask=None,
                    xml=f"/source/e{index}.xml",
                    star_mask_role="provenance_only",
                    xml_validation="valid",
                    capture_metadata={},
                    metadata_warnings=(),
                )
            )
        assigned = assign_grouped_partitions(
            records,
            locked_nights=set(),
            locked_cameras={"locked"},
            fold_count=3,
        )
        partitions: dict[str, set[str]] = {}
        for item in assigned:
            partitions.setdefault(str(item["physical_event_id"]), set()).add(str(item["partition"]))
        self.assertTrue(all(len(value) == 1 for value in partitions.values()))

    def test_source_snapshot_detects_no_mutation_and_reports_a_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "capture"
            source.mkdir()
            clip = source / "event.avi"
            clip.write_bytes(b"read only bytes")
            before = snapshot_tree(source)
            self.assertEqual([], compare_snapshots(before, snapshot_tree(source)))
            clip.write_bytes(b"changed")
            self.assertEqual(["changed: event.avi"], compare_snapshots(before, snapshot_tree(source)))


class ConcurrencyTests(unittest.TestCase):
    def test_concurrent_duplicate_enqueue_creates_one_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            queue = EventQueue(root / "state")
            clip = root / "clips" / "event"
            results = []

            def enqueue() -> None:
                results.append(queue.enqueue(clip).event_id)

            threads = [threading.Thread(target=enqueue) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(1, len(set(results)))
            claimed = queue.claim_next()
            self.assertIsNotNone(claimed)
            self.assertIsNone(queue.claim_next())

    def test_retry_backoff_and_os_process_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            queue = EventQueue(root / "state")
            event = queue.enqueue(root / "clips" / "event")
            claimed = queue.claim_next()
            assert claimed is not None
            queue.retry(event.event_id, "temporarily locked", delay_seconds=60)
            self.assertIsNone(queue.claim_next())
            with process_lock(queue.state_root / "worker.lock"):
                with self.assertRaises(WorkerAlreadyRunningError):
                    with process_lock(queue.state_root / "worker.lock"):
                        pass

    def test_slow_notification_dispatch_does_not_block_event_inference(self) -> None:
        class SlowDispatcher:
            def __init__(self) -> None:
                self.started = threading.Event()
                self.release = threading.Event()

            def dispatch_once(self) -> bool:
                self.started.set()
                self.release.wait(2)
                return False

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = EdgeConfig(
                state_root=root / "state",
                monitored_roots=(root / "clips",),
                poll_interval_seconds=0.01,
            )
            queue = EventQueue(config.state_root)
            queue.enqueue(root / "clips" / "first")
            queue.enqueue(root / "clips" / "second")
            dispatcher = SlowDispatcher()
            stop = threading.Event()
            processed: list[str] = []

            def processor(event):
                processed.append(event.event_id)
                if len(processed) == 2:
                    stop.set()
                return {"decision": "no_alert"}

            worker_thread = threading.Thread(
                target=lambda: EdgeWorker(queue, config, dispatcher).run(
                    processor, stop_event=stop
                )
            )
            worker_thread.start()
            self.assertTrue(dispatcher.started.wait(1))
            for _ in range(100):
                if len(processed) == 2:
                    break
                threading.Event().wait(0.01)
            self.assertEqual(2, len(processed))
            dispatcher.release.set()
            worker_thread.join(2)
            self.assertFalse(worker_thread.is_alive())


if __name__ == "__main__":
    unittest.main()
