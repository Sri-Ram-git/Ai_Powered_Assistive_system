# AI-Powered Assistive Vision System

**Version 1.0** — see [VERSION](VERSION)

A modular computer-vision system for an assistive device that helps
visually impaired people navigate their surroundings:

```
Camera → Object Detection → Tracking → OCR → Safety → Response Planner → Speech Output
```

The pipeline runs as an **asynchronous core engine** (`src/core`) with a
web dashboard + JSON API, and is being productised through a phased
roadmap (`docs/productization/`).

> Version 1.0 marks the audited, tested release state (159/159 tests).
> The productisation branch adds async architecture, evaluation, safety,
> modes, observability, CI, containerization, and documentation.
> See [REPORT.md](REPORT.md) for the full technical audit.

## Quickstart

```bash
pip install -r requirements.txt

# Download the detection model (yolov8s, ~22 MB, git-ignored).
# It identifies more everyday objects reliably than the smaller yolov8n.
curl -L -o models/yolov8s.onnx ^
  https://github.com/CVHub520/X-AnyLabeling/releases/download/v0.1.0/yolov8s.onnx

# Run the assistive desktop app (camera + detection + OCR + speech)
python src/assist/assist_app.py --config configs/assist_config.yaml

# Run the test suite (hardware-free, coverage-gated)
python -m pytest -q
```

The desktop app is the primary interface — no web browser or server
required.

## Modules

| Module | Package | Purpose |
|---|---|---|
| Core Engine | `src/core` | Async pipeline (grab/detect/OCR threads), config, latest-results |
| Camera System | `src/camera` | Webcam init/selection, mirror feed, FPS, threaded recording, draggable HUD |
| Object Detection | `src/detection` | YOLOv8 ONNX via OpenCV DNN — 80 COCO classes, CPU-only |
| Object Tracking | `src/tracking` | IoU tracker + guidance monitor (stable IDs, distance re-announce) |
| OCR | `src/ocr` | RapidOCR + async worker (never blocks the loop) |
| Navigation | `src/navigation` | Direction, pinhole distance, FOV calibration |
| Decision Engine | `src/decision` | Rule-based prioritised cues with cooldown |
| Safety Engine | `src/safety` | Deterministic hazard assessment (never an LLM) |
| Scene Context | `src/vision` | World model + optional VLM (offline fallback) |
| Response Planner | `src/response` | Priority / dedup / cooldown over all spoken output |
| Product Modes | `src/modes` | Object / Reading / Navigation / Scene / Voice presets |
| Speech | `src/audio`, `src/speech` | OS-native TTS + deterministic command parser/STT |
| Depth | `src/depth` | Optional monocular depth (model-free synthetic default) |
| API Layer | `src/api` | JSON API (health/state/config/command/mode/metrics) |
| Web UI | `src/ui` | Dark, camera-dominant dashboard (template + static) |
| Evaluation | `src/evaluation` | Detection/OCR/assistive/distance metrics |
| Metrics | `src/metrics` | Prometheus-style counters/gauges/histograms |
| Assist App | `src/assist` | End-to-end desktop pipeline |

## Productisation roadmap

Phases 0–25 are tracked in [`docs/productization/`](docs/productization/):

| Phase | Deliverable | Doc |
|---|---|---|
| 0 | Baseline audit + performance baseline | `baseline.md` |
| 1–2 | Async pipeline + non-blocking OCR | `phases_01_02_async_pipeline_ocr.md` |
| 3 | Distance calibration | — |
| 4 | AI evaluation system | `model_evaluation.md` |
| 5 | Speech input (commands/STT) | — |
| 6 | Optional depth | — |
| 7–8 | Scene context + safety engine | — |
| 9–10 | Optional VLM + response planner | — |
| 11–12 | Benchmark suite + model optimisation | `performance.md` |
| 13–14 | API/UI split + professional dashboard | — |
| 15 | Observability | `observability.md` |
| 16 | Security & privacy | `security_privacy.md` |
| 17 | Testing (coverage ≥80%) | — |
| 18 | CI/CD | `ci_cd.md` |
| 19 | Containerization | `containerization.md` |
| 20 | Model management | `models/manifest.yaml` |
| 21 | Product modes | `modes.md` |
| 22–24 | Offline-first / mobile / cloud strategy | `offline_first.md`, `mobile_path.md`, `cloud_devops.md` |
| 25 | Documentation + README overhaul | this file |

## Perception hardening (26-phase plan)

The basic vision pipeline (camera → detection → label → tracking →
smooth display → intelligent speech) is being hardened phase by phase,
recorded in [`docs/perception/`](docs/perception/):

| Doc | Covers |
|---|---|
| `baseline.md` | Phase 1 BEFORE numbers (3.2 fps blocking loop → async) |
| `detection.md` | Phases 3-5 preprocessing audit, threshold sweep, model choice |
| `tracking.md` | Phases 8-13 class-consistent + smoothed + voted tracking |
| `performance.md` | Phases 6-7, 21-23 async pipeline, targets, debug mode |
| `evaluation.md` | Phases 14-17 quality metrics + training decision |
| `FINAL_RESULTS.md` | Phases 24-26 checklist + honest BEFORE/AFTER |

Tooling: `scripts/benchmark/perception_benchmark.py`,
`scripts/debug/detection_visualizer.py`,
`scripts/benchmark/object_detection_metrics.py`, `configs/object_priority.yaml`.

### Object vocabulary (1000+ words)

The speaking engine only repeats words the model can detect.  The
vocabulary ([`docs/vocabulary/`](docs/vocabulary/)) gives the system a
**1551-word list** where every word is a real category in a labelled
image dataset (LVIS 1203 + OpenImages 601 + COCO 80):

| Item | Path |
|---|---|
| Runtime manifest | `data/vocabulary/object_vocabulary.yaml` |
| Plain word list | `data/vocabulary/words.txt` |
| Builder | `scripts/vocabulary/build_vocabulary.py` |
| Label-image downloader | `scripts/vocabulary/download_labeled_dataset.py` |
| Teach your own objects | `scripts/training/teach_objects.py` |
| Guide | `docs/vocabulary/vocabulary.md`, `docs/vocabulary/training.md` |

Words carry a safety tier (56 critical / 112 high / 1333 normal /
50 low) that raises speech priority, and the app varies repeated
phrasing so announcements are not one fixed sentence.

## Architecture

```
Camera → YOLO detection (yolov8s) → IoU tracking → distance/guidance
       → OCR (RapidOCR, throttled) → decision engine → speech (TTS)
```

The desktop app (`src/assist/assist_app.py`) is the **primary
interface** — it runs the whole pipeline in one window with a draggable
HUD, stable track IDs, per-object distances, and spoken guidance.
Safety decisions are deterministic and never pass through an LLM.  The
device is offline-first by design.

(The async engine in `src/core`, JSON API in `src/api`, and dashboard in
`src/ui` exist for headless/remote operation and evaluation tooling —
they are optional and not needed to use the device.)

## Product modes

Set the starting mode via `app.mode` in `configs/assist_config.yaml`
(optional; the desktop app runs object+reading+voice control out of the
box):

- `object` — detect objects, guide by proximity
- `reading` — OCR-focused: read text aloud
- `navigation` — safe navigation focus
- `scene` — higher-level scene description
- `voice` — command-first, quiet otherwise

**Safety is never mode-dependent.**

## Repository layout

```
├── src/
│   ├── core/            # async pipeline, config, latest-results
│   ├── api/             # JSON API blueprint
│   ├── ui/              # dashboard blueprint (html/css/js)
│   ├── detection/ ocr/ tracking/ navigation/ decision/
│   ├── safety/ vision/ response/ modes/ speech/ depth/
│   ├── evaluation/      # AI metrics
│   ├── metrics/         # Prometheus registry
│   ├── camera/ image_fundamentals/ image_processing/ morphology/
│   ├── assist/ server/ playground/
│   ├── audio/            # SpeechOutput, SpeechQueue, speech variety
│   ├── vocabulary/       # 1000+ word object vocabulary runtime
│   └── utils/
├── tests/               # hardware-free suite (coverage-gated ≥80%)
├── performance/         # benchmark suite + results
├── scripts/             # calibration, benchmarks, audits, optimization
│   ├── vocabulary/      # build manifest + download labelled images
│   └── training/        # teach-your-own-objects capture tool
├── data/vocabulary/     # manifest, word list, dataset class sources
├── docs/                # per-module + productization docs
├── configs/             # YAML configuration
├── models/              # manifest.yaml + weights (git-ignored)
├── evaluation/          # datasets, runner, reports
├── assets/              # local media (never pushed — see .gitignore)
└── logs/                # runtime logs
```

## Security & privacy

The device is offline-first and private: frames stay in memory, personal
media and model weights are git-ignored, and the API exposes only
whitelisted non-sensitive config.  Run:

```bash
python scripts/audit/security_scan.py   # secrets / personal-media scan
```

See `docs/productization/security_privacy.md` and `offline_first.md`.

## CI / container

```bash
docker build -t assistive-vision .      # CPU-only image (honest)
docker run --rm -p 5000:5000 assistive-vision
```

CI (`.github/workflows/ci.yml`) runs tests with the coverage gate,
security scan, and ruff on every push.

## Coding standards

- Python 3.11+, OpenCV, NumPy, Pillow, pytest, PyYAML, onnxruntime,
  rapidocr-onnxruntime, pyttsx3, Flask, psutil
- Type hints, docstrings, logging, explicit exception handling
- No global mutable state; stateless functions; PEP 8
- Safety-critical path stays deterministic (no LLM)

## Documentation

- `docs/architecture.md` — system design and module map
- `docs/productization/` — phased roadmap, performance, observability,
  security, CI/CD, containerization, modes, strategy docs
- Per-module docs: `docs/camera.md`, `docs/detection.md`, `docs/ocr.md`,
  `docs/decision.md`, `docs/audio.md`, `docs/navigation.md`,
  `docs/tracking.md`, `docs/assist.md`, `docs/server.md`, etc.

## License

MIT — see [LICENSE](LICENSE).