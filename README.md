# AI-Powered Assistive Vision System

**Version 1.0** — see [VERSION](VERSION)

A modular computer-vision system for an assistive device that helps
visually impaired people navigate their surroundings:

```
Camera → Object Detection → Tracking → OCR → Decision Engine → Speech Output
```

**Weeks 1-3 are complete** — from webcam foundation through live AI
assistance.  The full pipeline runs in `src/assist/assist_app.py`, and a
live **web dashboard** (`src/server`) streams the annotated feed and AI
guidance to any browser.

> Version 1.0 marks the audited, tested release state: 159/159 tests passing,
> TTS one-shot bug fixed, and de-duplicated continuous speech. See
> [REPORT.md](REPORT.md) for the full technical audit.

## Modules

| Module | Package | Purpose |
|---|---|---|
| Camera System | `src/camera` | Webcam init/selection, mirror feed, FPS, screenshots, threaded recording, professional draggable HUD |
| Image Fundamentals | `src/image_fundamentals` | Read/save, resize, crop, rotate, flip, colour-space conversions, pixel inspection, histograms |
| Image Processing | `src/image_processing` | Blur, threshold, Canny/Sobel/Laplacian edges, sharpen, brightness/contrast, noise add/remove |
| Morphology & Shapes | `src/morphology` | Erode/dilate/open/close, contours, geometry metrics, circle/rectangle/triangle detection |
| Vision Playground | `src/playground` | Live fullscreen webcam app with filter switching, toggles, recording, draggable HUD |
| **Object Detection** | `src/detection` | YOLOv8 ONNX via OpenCV DNN — 80 COCO classes, CPU-only |
| **Object Tracking** | `src/tracking` | IoU-based multi-object tracker + guidance monitor (stable IDs, distance re-announcement) |
| **OCR** | `src/ocr` | RapidOCR text recognition (CPU ONNX, no Paddle/PyTorch) |
| **Decision Engine** | `src/decision` | Rule-based prioritised cues with cooldown/rate limiting |
| **Speech Output** | `src/audio` | OS-native TTS (pyttsx3 / SAPI5), non-blocking worker |
| **Navigation** | `src/navigation` | Direction, distance heuristic, crosswalk & traffic-sign cues |
| **Assist App** | `src/assist` | End-to-end live pipeline: Camera → Detection → OCR → Decision → Speech |
| **Web Dashboard** | `src/server` | Flask server: MJPEG camera feed + live detections/distance/AI guidance in the browser |
| Infrastructure | — | `tests/`, `docs/`, `configs/`, `assets/`, `logs/` |

## Quickstart

```bash
pip install -r requirements.txt

# Download the detection model (12 MB, git-ignored)
curl -L -o models/yolov8n.onnx ^
  https://github.com/CVHub520/X-AnyLabeling/releases/download/v0.1.0/yolov8n.onnx

# Run the full assistive pipeline (detection + OCR + speech)
python src/assist/assist_app.py

# Run the live web dashboard (browser UI with camera + AI guidance)
python src/server/app.py --config configs/assist_config.yaml --port 5000

# Test the camera (mirror feed, fullscreen, draggable HUD)
python src/camera/camera_test.py

# Playground: live filters + toggles + recording
python src/playground/playground.py

# Demos (no webcam needed)
python src/image_fundamentals/image_demo.py
python src/image_processing/processing_demo.py
python src/morphology/demo.py

# Run the test suite
python -m pytest tests -q
```

## Distance accuracy

Distances use a pinhole model: `distance = reference_height × focal / box_height`.
Tune `configs/assist_config.yaml` → `navigation`:

- `vertical_fov` (degrees) — the biggest lever. Typical laptop webcams are
  ~50-60° vertical; measure yours or adjust until a known object reads
  correct. Wider FOV → shorter distances.
- `reference_heights` — override per-class real-world heights (metres),
  e.g. `person: 1.75`.

## Assist app keys

| Key | Action |
|---|---|
| `m` | Mute / unmute speech |
| `t` | Toggle OCR mode: ask-before-read vs auto-read |
| `r` | Read the most recently detected text aloud |
| `s` | Save annotated screenshot |
| `space` | Reset the decision cooldown |
| `q` | Quit |

OCR mode (default `ask_before_reading: false` in the config): text is
read aloud automatically as soon as it is detected ("Text says, ...").
Set `ask_before_reading: true` to make it announce "Text detected. Press
R to hear it read." and only read on `r` instead.

## Repository layout

```
├── src/
│   ├── camera/                # camera.py, camera_manager.py,
│   │                          # camera_utils.py, hud.py, camera_test.py
│   ├── image_fundamentals/    # image_utils.py, image_demo.py
│   ├── image_processing/      # processing.py, demos
│   ├── morphology/            # contour_utils.py, shape_detector.py
│   ├── detection/             # detector.py (YOLOv8 ONNX)
│   ├── ocr/                   # ocr_engine.py (RapidOCR)
│   ├── decision/              # engine.py (rule-based + cooldown)
│   ├── audio/                 # tts.py (pyttsx3)
│   ├── navigation/            # guidance.py (direction, distance, cues)
│   ├── tracking/              # tracker.py (IoU), monitor.py (guidance)
│   ├── assist/                # assist_app.py (full pipeline)
│   ├── server/                # app.py, pipeline.py (web dashboard)
│   ├── playground/            # playground.py
│   └── utils/                 # logger.py, exceptions.py
├── tests/                     # pytest suite (hardware-free)
├── docs/                      # per-module + architecture docs
├── configs/                   # YAML configuration
├── models/                    # yolov8n.onnx (git-ignored, re-download)
├── assets/                    # local media (never pushed — see .gitignore)
└── logs/                      # runtime logs
```

## Security & privacy

The `.gitignore` is configured to **never push personal media**: the
`assets/` tree (screenshots, recordings, photos) is blocked except for
placeholder `.gitkeep` files, and `models/*.onnx` weights are ignored.
Only the synthetic, generated test scene under
`src/image_fundamentals/sample_images/` is versioned.

## Coding standards

- Python 3.11+, OpenCV, NumPy, Pillow (HUD text), pytest, PyYAML,
  onnxruntime, rapidocr-onnxruntime, pyttsx3
- Type hints, docstrings, logging, explicit exception handling
- No global variables; stateless functions; PEP 8 style
- Per-module docs include architecture, function reference, dependencies,
  limitations, and future extensions

## Documentation

- `docs/architecture.md` — system design and module map
- `docs/camera.md`, `docs/image_fundamentals.md`,
  `docs/image_processing.md`, `docs/morphology.md`, `docs/playground.md`
- `docs/detection.md`, `docs/ocr.md`, `docs/decision.md`,
  `docs/audio.md`, `docs/navigation.md`, `docs/assist.md`,
  `docs/tracking.md`, `docs/server.md`

## License

MIT — see [LICENSE](LICENSE).
