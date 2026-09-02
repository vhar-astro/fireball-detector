# UFOCapture Edge Fireball Triage

This package is a separate production path from the legacy whole-frame CNN and
Faster R-CNN. It never imports or invokes either model. It implements a local,
network-independent inference cascade and queues notification delivery only
after an inference result is complete.

## Runtime data flow

1. UFOCapture's capture-end action calls `fireball-edge enqueue --clip-base ...`.
2. Enqueue normalizes the clip base and idempotently inserts it in an external
   SQLite queue. It does not decode or modify source media.
3. The single-instance worker prefers the red detected-pixel channel of
   `*M.bmp`. The green long-term average is retained as background evidence,
   while the blue area/scintillation mask is not treated as brightness. A missing, corrupt,
   empty, or dimensionally invalid map triggers streamed AVI frame differencing.
4. The worker measures duration, motion, linearity, saturation, brightness over
   local background, halo growth, and temporal peak from sequential AVI frames.
5. Up to three retained candidate ROIs are aspect-fit and median-padded to the
   model input. An ONNX MobileNetV3-Small score is combined with named temporal
   and geometric features by the manifest-bound calibrator.
6. The highest candidate score produces `no_alert`, `possible_fireball`, or
   `probable_fireball`. The latter two enter the Telegram outbox.
7. The annotated image is atomically published under external state, followed
   by `result.json` as the commit marker, then the SQLite event is completed.

UFOCapture documents `*M.bmp` as R=detected change, G=long-term average
brightness, and B=area/scintillation mask, and documents the capture-end action
as receiving the completed clip name:
<https://sonotaco.com/soft/UFO2/help/english/3-3.html>.

XML is listed in `sidecars_used` for provenance only and is excluded from
`scoring_sidecars` and every scoring interface. The result always has
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

Model and threshold data are intentionally not included. The worker fails
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

Training code is under `fireball_edge.offline` and is not imported or packaged
by the edge worker. Expert labels are CSV rows with:

```text
clip_base,label,physical_event_id,station,camera,night,nuisance_tags
```

Labels must be `fireball` or `non_fireball`; nuisance tags are semicolon
separated. Use tags such as `moon_only`, `fireball_with_moon`,
`ordinary_meteor`, `aircraft`, `cloud_glare`, and `sensor_artifact`. Moon is a
hard-negative/slice tag, never a rejection rule.

The available commands keep outputs below external state:

```bash
python -m fireball_edge.offline build-manifest --config edge.json --labels expert.csv --name v1
python -m fireball_edge.offline split-manifest --config edge.json --manifest manifest.json --name v1 --locked-night 2026-09-01 --locked-camera CAM-NEW
python -m fireball_edge.offline snapshot-sources --config edge.json --root D:\UFOCapture\captures --name before
python -m fireball_edge.offline evaluate --config edge.json --oof oof.csv --locked locked.csv --name v1
python -m fireball_edge.offline quantization-gate --fp32-recall 0.97 --int8-recall 0.96 --fp32-p95-ms 100 --int8-p95-ms 84
```

The code provides ImageNet-pretrained MobileNetV3-Small construction, external
multi-candidate ROI caches with a multi-instance event-label objective (so a
Moon candidate cannot steal a positive event label), grouped fold assignment, OOF logistic calibration, fixed-shape ONNX
export, and static QDQ INT8 generation with `QUInt8` activations, `QInt8`
weights, and reduced range. `ship_int8` is true only when locked recall falls no
more than one percentage point and target-machine p95 latency improves by at
least 15%.

## Verification and rollout status

Automated tests use generated AVI, `M.bmp`, peak-hold, and XML artifacts. They
cover red/green/blue channel interpretation, AVI fallback, missing sidecars,
corrupt AVI retry-to-failed behavior, duplicate/concurrent enqueue, restart
recovery, external atomic outputs, source snapshot equality, Telegram outage,
manifest/hash rejection, grouped threshold selection, and a real two-thread CPU
ONNX Runtime session.

Run:

```bash
python -m unittest discover -s tests -v
```

These tests do not establish model quality or deployment readiness. The seven
still images currently available are illustrative and are not used as accuracy
evidence. Before operational rollout, all of the following remain required:

- Full expert-labeled AVI/P/M/XML bundles grouped by physical event.
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
