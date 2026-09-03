"""Focused contract tests for the dependency-free edge queue scaffold."""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fireball_edge.__main__ import main
from fireball_edge import config as edge_config
from fireball_edge.config import EdgeConfig, StateRootError, default_state_root, load_config
from fireball_edge.event_id import event_id_for_clip_base, normalize_clip_base
from fireball_edge.queue import EventQueue, WorkerAlreadyRunningError
from fireball_edge.worker import EdgeWorker


class ConfigurationTests(unittest.TestCase):
    def test_windows_default_uses_localappdata(self) -> None:
        root = default_state_root(
            environ={"LOCALAPPDATA": r"C:\Users\observer\AppData\Local"}, platform="win32"
        )
        self.assertEqual(
            str(root).replace("\\", "/"),
            "C:/Users/observer/AppData/Local/FireballDetector",
        )

    def test_config_rejects_state_root_inside_monitored_tree_before_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            monitored = temporary / "ufocapture"
            proposed_state = monitored / "FireballDetector"
            config_path = temporary / "edge.json"
            config_path.write_text(
                json.dumps(
                    {
                        "state_root": str(proposed_state),
                        "monitored_roots": [str(monitored)],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(StateRootError):
                load_config(config_path)
            self.assertFalse(proposed_state.exists())

    def test_config_rejects_monitored_tree_inside_state_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            config_path = temporary / "edge.json"
            config_path.write_text(
                json.dumps(
                    {
                        "state_root": str(temporary / "state"),
                        "monitored_roots": [str(temporary / "state" / "clips")],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(StateRootError):
                load_config(config_path)

    def test_windows_containment_is_case_insensitive(self) -> None:
        child = Path(os.sep) / "Temp" / "Capture" / "event"
        parent = Path(os.sep) / "TEMP" / "CAPTURE"
        with mock.patch.object(edge_config.os, "name", "nt"):
            self.assertTrue(edge_config._is_within(child, parent))

    def test_direct_config_canonicalizes_all_runtime_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            capture = temporary / "captures"
            capture.mkdir()
            state = temporary / "state"
            config = EdgeConfig(
                state_root=state / ".." / "state",
                monitored_roots=(capture / ".." / "captures",),
                model_manifest=state / "models" / "active" / "model-manifest.json",
            )
            self.assertEqual(state.resolve(strict=False), config.state_root)
            self.assertEqual(
                (capture.resolve(strict=False),), config.monitored_roots
            )
            self.assertEqual(
                (state / "models" / "active" / "model-manifest.json").resolve(
                    strict=False
                ),
                config.model_manifest,
            )


class EventIdentityTests(unittest.TestCase):
    def test_normalized_clip_base_gives_the_same_id_for_equivalent_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            first = temporary / "night" / ".." / "night" / "20260902_010203"
            second = temporary / "night" / "20260902_010203"
            self.assertEqual(normalize_clip_base(first), normalize_clip_base(second))
            self.assertEqual(event_id_for_clip_base(first), event_id_for_clip_base(second))
            self.assertTrue(event_id_for_clip_base(first).startswith("evt_"))


class QueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.queue = EventQueue(self.root / "external-state")

    def test_enqueue_is_idempotent_and_never_changes_a_completed_event(self) -> None:
        clip = self.root / "readonly-source" / "event"
        first = self.queue.enqueue(clip)
        duplicate = self.queue.enqueue(clip.parent / "." / clip.name)
        self.assertEqual(first.event_id, duplicate.event_id)
        self.assertEqual("queued", duplicate.state)

        claimed = self.queue.claim_next()
        assert claimed is not None
        completed = self.queue.complete(claimed.event_id, {"label": "possible_fireball"})
        again = self.queue.enqueue(clip)
        self.assertEqual("complete", again.state)
        self.assertEqual(completed.result, again.result)

    def test_claim_retry_fail_and_complete_state_transitions(self) -> None:
        retry_event = self.queue.enqueue(self.root / "clips" / "retry")
        claimed = self.queue.claim_next()
        assert claimed is not None
        self.assertEqual(("processing", 1), (claimed.state, claimed.attempts))
        retried = self.queue.retry(retry_event.event_id, "corrupt AVI")
        self.assertEqual("retry", retried.state)
        claimed_again = self.queue.claim_next()
        assert claimed_again is not None
        self.assertEqual(2, claimed_again.attempts)
        failed = self.queue.fail(retry_event.event_id, "still corrupt")
        self.assertEqual("failed", failed.state)

        complete_event = self.queue.enqueue(self.root / "clips" / "complete")
        complete_claim = self.queue.claim_next()
        assert complete_claim is not None
        finished = self.queue.complete(complete_event.event_id, {"score": 0.99})
        self.assertEqual("complete", finished.state)
        self.assertEqual({"score": 0.99}, finished.result)

    def test_restart_recovery_requeues_processing_rows(self) -> None:
        event = self.queue.enqueue(self.root / "clips" / "interrupted")
        claimed = self.queue.claim_next()
        assert claimed is not None
        self.assertEqual("processing", claimed.state)
        self.assertEqual(1, self.queue.recover_processing())
        recovered = self.queue.get(event.event_id)
        assert recovered is not None
        self.assertEqual("retry", recovered.state)
        self.assertEqual("recovered after worker restart", recovered.last_error)

    def test_only_one_worker_can_hold_a_live_lease(self) -> None:
        with self.queue.worker_lock(60):
            with self.assertRaises(WorkerAlreadyRunningError):
                self.queue.acquire_worker_lock("second-worker", 60)


class WorkerAndCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.config_path = self.root / "edge.json"
        self.state_root = self.root / "external-state"
        self.config_path.write_text(
            json.dumps({"state_root": str(self.state_root), "monitored_roots": [str(self.root / "clips")]}),
            encoding="utf-8",
        )

    def test_worker_recovers_then_processes_with_an_injected_processor(self) -> None:
        config = load_config(self.config_path)
        queue = EventQueue(config.state_root)
        queued = queue.enqueue(self.root / "clips" / "event")
        assert queue.claim_next() is not None  # Simulate a process that stopped mid-event.
        worker = EdgeWorker(queue, config)
        processed = worker.run(lambda event: {"event": event.event_id}, once=True)
        self.assertEqual(1, processed)
        final = queue.get(queued.event_id)
        assert final is not None
        self.assertEqual("complete", final.state)
        self.assertEqual(2, final.attempts)

    def test_os_lock_owner_can_replace_a_crashed_sqlite_lease_immediately(self) -> None:
        config = load_config(self.config_path)
        queue = EventQueue(config.state_root)
        queued = queue.enqueue(self.root / "clips" / "event")
        queue.acquire_worker_lock("dead-process", 3600)
        processed = EdgeWorker(queue, config).run(
            lambda event: {"event": event.event_id}, once=True
        )
        self.assertEqual(1, processed)
        self.assertEqual("complete", queue.get(queued.event_id).state)

    def test_cli_enqueue_and_worker_once(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(
                0,
                main(
                    [
                        "enqueue",
                        "--clip-base",
                        str(self.root / "clips" / "event"),
                        "--config",
                        str(self.config_path),
                    ]
                ),
            )
        document = json.loads(stdout.getvalue())
        self.assertEqual("queued", document["state"])

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(1, main(["worker", "--config", str(self.config_path), "--once"]))
        self.assertIn("model_manifest is required", stderr.getvalue())
        self.assertFalse(
            any(
                getattr(handler, "_fireball_edge_owned", False)
                for handler in logging.getLogger("fireball_edge").handlers
            )
        )


if __name__ == "__main__":
    unittest.main()
