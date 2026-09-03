"""Command line entry point used by UFOCapture's capture-end action."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .config import EdgeConfig, load_config
from .contracts import CANDIDATE_EXTRACTOR, SCHEMA_VERSION
from .queue import EventQueue, QueueEvent, WorkerAlreadyRunningError
from .notifications import OutboxDispatcher
from .pipeline import EventProcessor
from .worker import EdgeWorker
from .logging_setup import close_logging, configure_logging


def _event_document(event: QueueEvent) -> dict[str, object]:
    return {
        "event_id": event.event_id,
        "clip_base": event.clip_base,
        "state": event.state,
        "attempts": event.attempts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fireball-edge")
    commands = parser.add_subparsers(dest="command", required=True)

    enqueue = commands.add_parser("enqueue", help="queue one UFOCapture clip base")
    enqueue.add_argument("--clip-base", required=True, type=Path)
    enqueue.add_argument("--config", type=Path)

    worker = commands.add_parser("worker", help="run the single-instance edge worker")
    worker.add_argument("--config", type=Path)
    worker.add_argument(
        "--once",
        action="store_true",
        help="process at most one queue item (for service checks and tests)",
    )
    return parser


def _execute(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config: EdgeConfig = load_config(getattr(args, "config", None))
    clip_base = None
    if args.command == "enqueue":
        # This check happens before EventQueue creates state_root.
        clip_base = config.validate_clip_base(args.clip_base)
    # Re-check after creating the directory to account for links/junctions that
    # did not exist during initial configuration parsing.
    queue = EventQueue(config.state_root)
    from .config import validate_state_root

    validate_state_root(queue.state_root, config.monitored_roots)
    if args.command == "enqueue":
        assert clip_base is not None
        clip_base = config.validate_clip_base(clip_base)
        event = queue.enqueue(
            clip_base,
            required_schema_version=SCHEMA_VERSION,
            candidate_extractor=CANDIDATE_EXTRACTOR,
        )
        print(json.dumps(_event_document(event), sort_keys=True))
        return 0

    try:
        # Native runtimes may create diagnostics/telemetry cache files in the
        # process working directory during import. Move to external state before
        # ONNX Runtime is imported so UFOCapture's tree remains immutable.
        previous_directory = Path.cwd()
        os.chdir(config.state_root)
        try:
            configure_logging(config.state_root)
            processor = EventProcessor(config)
            dispatcher = OutboxDispatcher(queue, config) if config.telegram_enabled else None
            processed = EdgeWorker(queue, config, dispatcher).run(processor, once=args.once)
        finally:
            close_logging()
            os.chdir(previous_directory)
    except WorkerAlreadyRunningError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps({"processed": processed, "state_root": str(config.state_root)}))
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return _execute(argv)
    except WorkerAlreadyRunningError as error:
        print(str(error), file=sys.stderr)
        return 2
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
