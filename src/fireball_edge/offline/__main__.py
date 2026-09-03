"""Offline dataset and release-evidence commands (not bundled for edge use)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..artifacts import write_json_atomic
from ..config import load_config
from ..contracts import CANDIDATE_EXTRACTOR, SCHEMA_VERSION
from .evaluation import (
    load_predictions,
    locked_report,
    quantization_gate,
    select_possible_threshold,
    select_probable_threshold,
)
from .manifest import (
    build_records,
    load_expert_labels,
    read_manifest,
    snapshot_tree,
    write_manifest,
)
from .orchestrator import run_training
from .splits import assign_grouped_partitions


def _external_output(config_path: Path, relative: Path) -> tuple[object, Path]:
    config = load_config(config_path)
    destination = (config.state_root / relative).resolve(strict=False)
    destination.relative_to(config.state_root)
    return config, destination


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python -m fireball_edge.offline")
    commands = root.add_subparsers(dest="command", required=True)

    manifest = commands.add_parser("build-manifest")
    manifest.add_argument("--config", required=True, type=Path)
    manifest.add_argument("--labels", required=True, type=Path)
    manifest.add_argument("--name", required=True)

    preflight = commands.add_parser("preflight")
    preflight.add_argument("--config", required=True, type=Path)
    preflight.add_argument("--labels", required=True, type=Path)
    preflight.add_argument("--name", required=True)

    train = commands.add_parser("train")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--labels", required=True, type=Path)
    train.add_argument("--name", required=True)
    train.add_argument("--model-name", required=True)
    train.add_argument("--locked-night", action="append", default=[])
    train.add_argument("--locked-camera", action="append", default=[])
    train.add_argument("--resume", action="store_true")
    train.add_argument("--folds", type=int, default=5)
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--seed", type=int, default=1729)

    snapshot = commands.add_parser("snapshot-sources")
    snapshot.add_argument("--config", required=True, type=Path)
    snapshot.add_argument("--root", required=True, type=Path)
    snapshot.add_argument("--name", required=True)

    split = commands.add_parser("split-manifest")
    split.add_argument("--config", required=True, type=Path)
    split.add_argument("--manifest", required=True, type=Path)
    split.add_argument("--name", required=True)
    split.add_argument("--locked-night", action="append", default=[])
    split.add_argument("--locked-camera", action="append", default=[])
    split.add_argument("--folds", type=int, default=5)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--config", required=True, type=Path)
    evaluate.add_argument("--oof", required=True, type=Path)
    evaluate.add_argument("--locked", required=True, type=Path)
    evaluate.add_argument("--name", required=True)

    gate = commands.add_parser("quantization-gate")
    gate.add_argument("--fp32-recall", required=True, type=float)
    gate.add_argument("--int8-recall", required=True, type=float)
    gate.add_argument("--fp32-p95-ms", required=True, type=float)
    gate.add_argument("--int8-p95-ms", required=True, type=float)
    return root


def execute(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "quantization-gate":
        print(
            json.dumps(
                quantization_gate(
                    fp32_recall=args.fp32_recall,
                    int8_recall=args.int8_recall,
                    fp32_p95_ms=args.fp32_p95_ms,
                    int8_p95_ms=args.int8_p95_ms,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "train":
        config = load_config(args.config)
        output = run_training(
            config,
            labels_path=args.labels,
            dataset_name=args.name,
            model_name=args.model_name,
            locked_nights=set(args.locked_night),
            locked_cameras=set(args.locked_camera),
            resume=args.resume,
            folds=args.folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
    elif args.command in {"build-manifest", "preflight"}:
        relative = (
            Path("datasets") / args.name / "manifest-v2.json"
            if args.command == "build-manifest"
            else Path("validation") / f"{args.name}-preflight-v2.json"
        )
        config, output = _external_output(
            args.config, relative
        )
        records = build_records(load_expert_labels(args.labels))
        for record in records:
            config.validate_clip_base(record.clip_base)  # type: ignore[attr-defined]
        if args.command == "build-manifest":
            write_manifest(output, records)
        else:
            write_json_atomic(
                output,
                {
                    "schema_version": SCHEMA_VERSION,
                    "candidate_extractor": CANDIDATE_EXTRACTOR,
                    "status": "passed",
                    "records": [
                        {
                            "event_id": record.event_id,
                            "clip_base": record.clip_base,
                            "station": record.station,
                            "camera": record.camera,
                            "night": record.night,
                            "stack_image": record.stack_image,
                            "star_mask_role": record.star_mask_role,
                            "xml_validation": record.xml_validation,
                            "warnings": list(record.metadata_warnings),
                        }
                        for record in records
                    ],
                },
            )
    elif args.command == "snapshot-sources":
        config, output = _external_output(
            args.config, Path("validation") / f"{args.name}-source-snapshot.json"
        )
        config.validate_source(args.root)  # type: ignore[attr-defined]
        write_json_atomic(output, snapshot_tree(args.root))
    elif args.command == "split-manifest":
        _, output = _external_output(
            args.config, Path("datasets") / args.name / "partitioned-manifest-v2.json"
        )
        records = read_manifest(args.manifest)
        assigned = assign_grouped_partitions(
            records,
            locked_nights=set(args.locked_night),
            locked_cameras=set(args.locked_camera),
            fold_count=args.folds,
        )
        write_json_atomic(
            output,
            {
                "schema_version": SCHEMA_VERSION,
                "candidate_extractor": CANDIDATE_EXTRACTOR,
                "grouping_key": "physical_event_id",
                "locked_selection": {
                    "nights": args.locked_night,
                    "cameras": args.locked_camera,
                },
                "records": assigned,
            },
        )
    else:
        _, output = _external_output(
            args.config, Path("validation") / f"{args.name}-evaluation-v2.json"
        )
        oof = load_predictions(args.oof)
        possible = select_possible_threshold(oof)
        probable = select_probable_threshold(oof)
        if probable < possible:
            probable = possible
        locked = load_predictions(args.locked)
        report = {
            "schema_version": SCHEMA_VERSION,
            "candidate_extractor": CANDIDATE_EXTRACTOR,
            "possible_threshold": possible,
            "probable_threshold": probable,
            "selection": {
                "possible": "at least 98% OOF recall and every fold at least 95%",
                "probable": "maximum OOF F2, clamped not below possible",
            },
            "locked_test": locked_report(locked, possible),
            "rollout_gate": {
                "minimum_possible_recall": 0.95,
                "passed": locked_report(locked, possible)["possible_fireball_recall"] >= 0.95,
            },
        }
        write_json_atomic(output, report)
    print(str(output))
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return execute(argv)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
