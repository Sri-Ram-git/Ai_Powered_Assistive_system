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

# Download the detection model (12 MB, git-ignored)
curl -L -o models/yolov8n.onnx ^
  https://github.com/CVHub520/X-AnyLabeling/releases/download/v0.1.0/yolov8n.onnx

# Run the live web dashboard (browser UI with camera + AI guidance)
python src/server/app.py --config configs/assist_config.yaml --port 5000
#   -> dashboard:      http://127.0.0.1:5000/
#   -> JSON API:       http://127.0.0.1:5000/api/health
#   -> metrics:        http://127.0.0.1:5000/api/metrics

# Run the desktop assist app (detection + OCR + speech)
python src/assist/assist_app.py

# Run the test suite (hardware-free, coverage-gated)
python -m pytest -q
```

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

## Architecture

```
┌──────────────  src/core (engine, no Flask)  ──────────────┐
│ grab thread → FrameManager → JPEG (UI)                     │
│ detect thread → YOLO → tracker → scene → safety → planner  │
│ OCR worker   → latest OCR result (non-blocking)            │
└──────────────────────────┬─────────────────────────────────┘
                           │ state / results / jpeg
        ┌──────────────────┴───────────────────┐
   src/api (JSON)                     src/ui (dashboard)
   /api/health /api/state /api/mode  /  + /video_feed
   /api/command /api/config /api/metrics
```

The engine has no Flask dependency; the server is a thin composition
root.  Safety decisions are deterministic and never pass through an LLM.
The device is offline-first by design.

## Product modes

Switch modes via the dashboard, `POST /api/mode`, or `app.mode` config:

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
│   └── utils/
├── tests/               # hardware-free suite (coverage-gated ≥80%)
├── performance/         # benchmark suite + results
├── scripts/             # calibration, benchmarks, audits, optimization
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