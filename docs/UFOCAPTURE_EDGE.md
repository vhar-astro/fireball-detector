# UFOCapture Edge Fireball Triage

This package is a separate production path from the legacy whole-frame CNN and
Faster R-CNN. It never imports or invokes either model. It implements a local,
network-independent inference cascade and queues notification delivery only
after an inference result is complete.

## Runtime data flow

1. UFOCapture's capture-end action calls `fireball-edge enqueue --clip-base ...`.
2. Enqueue normalizes the clip base and idempotently inserts it in an external
   SQLite queue. It does not decode or modify source media.
3. The single-instance worker streams sequential AVI frame differences as the
   sole candidate source and retains at most three regions. `*M.bmp` is an
   optional star-mask provenance file and is never opened for scoring.
4. The worker measures duration, motion, linearity, saturation, brightness over
   local background, halo growth, and temporal peak from sequential AVI frames.
5. Candidate ROIs are cropped from a valid `*P.bmp` stack, or `*P.jpg` when the
   BMP is absent or unreadable. Runtime reconstructs a maximum composite from
   AVI when neither stack is usable. ROIs are aspect-fit and median-padded to
   the model input, then an ONNX MobileNetV3-Small score is combined with named
   AVI-temporal and geometric features by the manifest-bound calibrator.
6. The highest candidate score produces `no_alert`, `possible_fireball`, or
   `probable_fireball`. The latter two enter the Telegram outbox.
7. The annotated image is atomically published under external state, followed
   by `result.json` as the commit marker under `results/v2/<event-id>/`, then
   the SQLite event is completed. Legacy complete queue rows are archived and
   requeued; their v1 result files are left untouched.

UFOCapture documents the capture-end action as receiving the completed clip name:
<https://sonotaco.com/soft/UFO2/help/english/3-3.html>.

All discovered bundle files are listed in `source_provenance`. Only AVI and the
selected P stack are listed in `scoring_sidecars`; a reconstructed stack lists
AVI once. XML is parsed for capture validation and grouping metadata, while its
values are excluded from every scoring interface. The result records
`star_mask_role: "provenance_only"`, XML validation status, and
`scientific_status: "uncalibrated"`. The score is a triage probability, not a
physical brightness measurement. The IAU's fireball definition depends on
absolute visual magnitude as seen from 100 km, which this edge camera alone does
not establish: <https://iauarchive.eso.org/public/themes/meteors_and_meteorites/>.

## Install and configure

For a Python development install:

```bash
python -m pip install -e .
```

Copy `edge-config.example.json` to
`%LOCALAPPDATA%\FireballDetector\edge-config.json`. Omitting `state_root` uses
`%LOCALAPPDATA%\FireballDetector`. `model_manifest` may be relative to that
root, but it must remain under `state_root\models`. Set every capture directory
in `monitored_roots`; overlap in either direction with state is rejected before
the queue is created. Keep `state_root` on a local disk; SQLite WAL and the OS
worker lock are not supported on an SMB/network share.

Model and threshold data are intentionally not included. Schema-v1 manifests,
caches, model packages, and cached results are rejected and must be rebuilt;
existing external files are not migrated or deleted. The worker fails
closed until a model package contains a verified ONNX hash, verified calibrator
hash, preprocessing schema, extractor version, feature order, quantization
mode, model version, and OOF-selected thresholds.
An INT8 manifest is additionally rejected unless it carries a locked report,
verifies its SHA-256, and records passing recall/target-i7-4500U latency evidence.

Telegram secrets are never stored in the JSON configuration. When delivery is
enabled, define the environment variables named by `token_env` and
`chat_id_env`. Inference completion is independent of Telegram availability;
failed delivery stays in the external outbox with exponential delay, bounded
attempts, and a unique event/destination key. Telegram has no idempotency key,
so a process crash after the server accepts a photo but before local success is
recorded leaves a small duplicate-delivery window. Each caption contains the
event ID so duplicates are recognizable.

## UFOCapture action and worker

UFOCapture appends the clip name to the selected command. Copy and adjust
`ufocapture-action.example.cmd`, then select that command file as **Action at
Capture End**. It invokes:

```text
fireball-edge.exe enqueue --clip-base "<completed clip>" --config "<edge-config.json>"
```

Run one background worker (for example through Windows Task Scheduler):

```text
fireball-edge.exe worker --config "%LOCALAPPDATA%\FireballDetector\edge-config.json"
```

For a service check, `worker --once` processes at most one inference event.
The normal long-running worker drains the network outbox on a separate thread,
so Telegram timeouts cannot delay later inference claims. Event states are `queued`, `processing`, `complete`,
`retry`, and `failed`. Duplicate action calls return the existing event and do
not reset completed work. Interrupted processing rows and outbox sends are
recovered when the next singleton worker starts. A crash-released OS lock fences
the renewable SQLite lease, and processing retries use bounded exponential
backoff rather than immediately exhausting all attempts.

## Windows build

Build on Windows; PyInstaller does not cross-build native OpenCV or ONNX Runtime
artifacts:

```powershell
py -m pip install ".[build]"
py -m PyInstaller --clean --noconfirm fireball-edge.spec
```

The output is an `onedir` application at
`dist\fireball-edge\fireball-edge.exe`. It is one executable interface with the
`enqueue` and `worker` subcommands. `onedir` is deliberate: a one-file bundle
would unpack large native libraries for every capture-end enqueue call.

## Offline dataset and model tools

Training code is under `fireball_edge.offline`; Torch and torchvision remain
outside the edge import path and PyInstaller package. Expert labels are CSV
rows with:

```text
clip_base,label,physical_event_id,nuisance_tags
```

Labels must be `fireball` or `non_fireball`, physical event IDs are mandatory,
and nuisance tags are semicolon separated. Optional `station`, `camera`, and
`night` columns are assertions against XML-derived values; they never override
XML. Use tags such as `moon_only`, `fireball_with_moon`,
`ordinary_meteor`, `aircraft`, `cloud_glare`, and `sensor_artifact`. Moon is a
hard-negative/slice tag, never a rejection rule.

The available commands keep outputs below external state:

```bash
python -m fireball_edge.offline preflight --config edge.json --labels expert.csv --name dataset-v2
python -m fireball_edge.offline train --config edge.json --labels expert.csv --name dataset-v2 --model-name model-v2 --locked-night 2026-09-01 --locked-camera PL:station:camera:lens --resume
python -m fireball_edge.offline build-manifest --config edge.json --labels expert.csv --name dataset-v2
python -m fireball_edge.offline split-manifest --config edge.json --manifest manifest-v2.json --name dataset-v2 --locked-night 2026-09-01 --locked-camera PL:station:camera:lens
python -m fireball_edge.offline snapshot-sources --config edge.json --root D:\UFOCapture\captures --name before
python -m fireball_edge.offline evaluate --config edge.json --oof oof.csv --locked locked.csv --name dataset-v2
python -m fireball_edge.offline quantization-gate --fp32-recall 0.97 --int8-recall 0.96 --fp32-p95-ms 100 --int8-p95-ms 84
```

`train` performs strict preflight, immutable manifest creation, physical-event
grouped folds, candidate ROI/geometry/temporal caching, fold training, grouped
OOF prediction, event-maximum multi-instance calibration, threshold selection,
final FP32 training, ONNX export, and locked-set evaluation. Defaults are five
folds, 12 epochs, batch size 32, learning rate `3e-4`, deterministic seeding,
and zero data-loader workers. It publishes a candidate package but never
replaces the configured active model. Activation is eligible only when locked
possible-fireball recall reaches 95%.

Static QDQ INT8 generation remains a subsequent stage with `QUInt8` activations, `QInt8`
weights, and reduced range. `ship_int8` is true only when locked recall falls no
more than one percentage point and target-machine p95 latency improves by at
least 15%.

## Verification and rollout status

Automated tests use generated 1920x1080 AVI, P BMP/JPG stacks, colored `M.bmp`,
and XML artifacts. They cover M corruption/removal invariance, AVI-only
candidate and temporal semantics, BMP preference, JPG and runtime AVI-stack
fallback, strict training preflight, missing sidecars,
corrupt AVI retry-to-failed behavior, duplicate/concurrent enqueue, restart
recovery, external atomic outputs, source snapshot equality, Telegram outage,
v1 manifest/cache/model/result rejection, grouped threshold selection,
event-maximum calibration, and a real two-thread CPU ONNX Runtime session.

Run:

```bash
python -m unittest discover -s tests -v
```

These tests do not establish model quality or deployment readiness. The seven
still images currently available are illustrative and are not used as accuracy
evidence. Before operational rollout, all of the following remain required:

- Full expert-labeled AVI/P/XML bundles (with optional M provenance) grouped by
  physical event.
- Grouped OOF selection: possible threshold at least 98% recall, no fold below
  95%; probable threshold by maximum F2.
- A naturally imbalanced locked set of unseen nights/cameras with at least 95%
  possible-fireball recall, PR-AUC, calibration, false alerts per camera-night,
  and all requested nuisance slices.
- FP32 versus static-QDQ-INT8 locked-set and latency comparison. Do not ship INT8
  unless both gates pass.
- On the actual i7-4500U: capture-completion-to-result p95 at most 30 seconds,
  peak memory below 1 GB, at most two inference threads, and a soak test showing
  no additional UFOCapture dropped frames or missed triggers.
- Several nights shadow-running to a test Telegram chat before operational
  notification is enabled.

Gemma has no production import or runtime role. If used offline for nuisance-tag
suggestions or uncertain-sample prioritization, a human must confirm its output,
and it must never see or label the locked test set.
