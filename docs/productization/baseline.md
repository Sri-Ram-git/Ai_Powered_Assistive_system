# Productization Baseline — VERSION 1.0 MVP

**Branch:** `feature/ai-productization` (created from `master` @ `755e6e7`)
**Baseline recorded:** 16 Aug 2026
**Purpose:** Freeze the state of the working MVP before productization begins, so
progress on every subsequent phase can be measured against a documented baseline.
Per the project working rule, **no baseline code has been modified** to produce this
document.

---

## 1. Existing Architecture

### 1.1 High-level data flow

```
                    ┌─────────────────────────────────────────────┐
                    │              src/assist/assist_app.py       │
                    │         (single synchronous loop, GUI)      │
                    └─────────────────────────────────────────────┘
                                      │
┌──────────┐   ┌───────────┐   ┌────────────┐   ┌───────────┐   ┌──────────┐
│ Camera   │──▶│ Detect    │──▶│ Track      │──▶│ Monitor   │──▶│ Decision │
│ (OpenCV) │   │ YOLOv8n   │   │ IoUTracker │   │ + OCR     │   │ Engine   │
│ CAP_DSHOW│   │ 640×640   │   │ IoU match  │   │ (RapidOCR)│   │ priority │
└──────────┘   └───────────┘   └────────────┘   └───────────┘   └────┬─────┘
                                                                     │
                                                                     ▼
                                                              ┌────────────┐
                                                              │ SpeechOutput│
                                                              │ pyttsx3     │
                                                              │ SAPI5 worker│
                                                              └────────────┘
```

### 1.2 Two runtimes share the same modules

**A. Desktop app — `src/assist/assist_app.py`**
- One loop per frame: read → (every 2 frames) detect + track → (every 10 frames)
  OCR → `TrackingMonitor.events()` → `DecisionEngine.decide()` → `tts.speak()` →
  draw tracks/HUD → `imshow`.
- OCR runs **in-band**, so OCR latency stalls the whole loop.
- Keyboard: m (mute), t (OCR mode), r (read text), s (screenshot), space (reset),
  q (quit). Draggable HUD.

**B. Web dashboard — `src/server/app.py` + `src/server/pipeline.py`**
- `PipelineServer` runs **three threads**:
  - *coordinator* (`_run`): boots camera, spawns grab + infer, waits on stop.
  - *grab thread* (`_grab_loop`): reads frames, annotates with latest AI results,
    JPEG-encodes → `latest_jpeg` for `/video_feed`. **Never runs AI.** Reports feed
    FPS.
  - *inference thread* (`_infer_loop`): detect/track/OCR/decision on the latest
    shared frame; publishes JSON state (`detections`, `ocr_text`, `guidance`,
    `latency_ms`) and calls `speech_callback`.
- Flask endpoints: `/` (dashboard HTML), `/video_feed` (MJPEG), `/api/state` (JSON).
- Grab and inference are decoupled, so the feed stays responsive while inference is
  slow.

### 1.3 Module map (`src/`)

| Package | Key types | Notes |
|---|---|---|
| `src/camera` | `Camera`, `CameraManager`, `CameraInfo`, `HUD`, `Canvas`, `FontManager`, `VideoRecorder`, utils | CAP_DSHOW, mirror default; PIL-drawn HUD |
| `src/image_fundamentals` | stateless fns in `image_utils.py` | I/O, transforms, colour, stats, hist |
| `src/image_processing` | stateless fns in `processing.py` | blur/threshold/edges/enhance/noise |
| `src/morphology` | `contour_utils`, `ShapeDetector` | erode/dilate/open/close, shapes |
| `src/detection` | `YoloDetector`, `DetectionResult` | ONNX via cv2.dnn; 80 COCO classes |
| `src/tracking` | `IoUTracker`, `TrackedObject`, `TrackingMonitor` | pure-NumPy IoU; change-based phrases |
| `src/ocr` | `OcrEngine`, `OcrResult` | RapidOCR CPU (ONNX) |
| `src/navigation` | stateless fns in `guidance.py` | direction, pinhole distance, cues |
| `src/decision` | `DecisionEngine`, `evaluate`, `FrameSummary`, `Decision`, `cue_identity` | priority + cooldown + dedup |
| `src/audio` | `SpeechOutput` | pyttsx3 worker, startLoop(False)+iterate pump |
| `src/assist` | `main()` | desktop end-to-end |
| `src/server` | `PipelineServer`, `PipelineConfig`, `create_app` | Flask + threaded pipeline |
| `src/playground` | `main()` | Week-1 live filter app |
| `src/utils` | `setup_logger`, exception hierarchy | logging to `logs/app.log` |

### 1.4 Config flow

- `configs/assist_config.yaml` → `assist_app.main()` (per-section dicts) and
  `PipelineConfig.from_yaml()` (server).
- `assist_app` reads `detection.conf_threshold` (default 0.35) and
  `navigation.vertical_fov` (default 55.0).
- `PipelineConfig.from_yaml` pushes optional `navigation.reference_heights` into
  `guidance._REFERENCE_HEIGHTS`.
- `camera_config.yaml` and `logging_config.yaml` exist but are **not consumed** by
  code (constructor defaults / hardcoded logger settings used).

---

## 2. Existing Features

| Feature | Status | Where |
|---|---|---|
| Camera capture, mirror, FPS, screenshots, threaded recording | Working | `src/camera` |
| Draggable monochrome HUD (menu bar, dashboard, REC, toasts) | Working | `src/camera/hud.py` |
| Image fundamentals (I/O, transforms, colour, histograms) | Working | `src/image_fundamentals` |
| Image processing (blur, threshold, edges, enhance, noise) | Working | `src/image_processing` |
| Morphology + shape detection (circle/rect/triangle) | Working | `src/morphology` |
| YOLOv8n object detection (ONNX via OpenCV DNN, CPU) | Working | `src/detection` |
| IoU multi-object tracking with stable IDs | Working | `src/tracking` |
| Tracking guidance monitor (speech on change) | Working | `src/tracking/monitor.py` |
| RapidOCR text recognition | Working | `src/ocr` |
| Navigation cues (direction, distance, crosswalk/traffic) | Working | `src/navigation` |
| Rule-based decision engine (cooldown + identity dedup) | Working | `src/decision` |
| OS-native TTS (pyttsx3/SAPI5), non-blocking worker | Working | `src/audio` |
| End-to-end Assist App | Working | `src/assist/assist_app.py` |
| Flask web dashboard (MJPEG + JSON state) | Working | `src/server` |
| Config system (YAML) | Working (minor gaps) | `configs/` |
| Logging to `logs/app.log` | Working | `src/utils/logger.py` |
| Voice commands / STT | **NOT IMPLEMENTED** | — |
| Depth estimation | **NOT IMPLEMENTED** | — |
| Scene context layer | **NOT IMPLEMENTED** | — |
| Safety engine | **NOT IMPLEMENTED** | — |
| Formal AI evaluation | **NOT IMPLEMENTED** | — |

---

## 3. Existing Tests

- **Command:** `python -m pytest tests -q`
- **Baseline result: 159 passed, 0 failed** (fresh run 29.44 s on the new branch).

| Test file | Tests | Area |
|---|---|---|
| tests/test_audio.py | 7 | TTS SpeechOutput (fake engine) |
| tests/test_camera.py | 14 | Camera lifecycle, manager, utils, HUD |
| tests/test_decision.py | 22 | DecisionEngine, cue_identity, cooldown, priorities |
| tests/test_detection.py | 15 | YoloDetector parsing, NMS, letterbox |
| tests/test_image_utils.py | 21 | Image fundamentals I/O, transforms, stats |
| tests/test_morphology.py | 14 | Erode/dilate/open/close, contours, shapes |
| tests/test_navigation.py | 17 | direction, distance, scene_cues, obstacles |
| tests/test_ocr.py | 9 | RapidOCR wrapper, OcrResult |
| tests/test_pipeline_e2e.py | 2 | Real ONNX model + stub camera end-to-end |
| tests/test_processing.py | 17 | Filters, thresholds, edges, noise |
| tests/test_server.py | 5 | Flask endpoints, PipelineConfig, helpers |
| tests/test_tracking.py | 16 | IoU association, monitor events |

**Coverage:** **NOT MEASURED** — `pytest-cov` is declared in `requirements.txt` but
not installed; no coverage config exists.

---

## 4. Existing Performance (baseline numbers)

Measured on this machine (CPU-only, no GPU), synthetic inputs unless noted.

| Metric | Baseline value | Notes |
|---|---|---|
| YOLO detect, 720p → 640×640, steady-state avg | ~50–76 ms | high variance (46–330 ms across runs) |
| YOLO detect throughput | ~13–20 inference/s | at 720p |
| RapidOCR steady-state | ~2.8–4.7 s per frame | **high run-to-run variance** (2.1–4.7 s) |
| RapidOCR first call | ~0.6–4.5 s | includes model load |
| `direction_of` / `distance_estimate` | < 0.2 ms | negligible |
| `DecisionEngine.decide` | ~0.01 ms | negligible |
| `IoUTracker.update` / `TrackingMonitor.events` | < 0.1 ms | negligible |
| End-to-end loop (stub camera, no HW) | ~1.4–2.4 loop fps | **bounded by OCR** |
| Web feed, real camera 1280×720 | ~5–9 fps | grab thread decoupled from inference |
| `/api/state` reported latency (idle scene) | ~0.1 ms | AI-loop tick, no detections |

**Key interpretation:** OCR is the dominant CPU bottleneck (≈ 2.8–4.7 s/frame). The
server's grab/infer thread split keeps the live feed usable, but the single-threaded
`assist_app.py` inherits the OCR-bound cadence (~1.4–2.4 fps).

---

## 5. Existing Limitations

1. **OCR latency** — ~4–5 s per CPU call (varies 2.1–4.7 s); blocks the synchronous
   app loop.
2. **Synchronous end-to-end pipeline** — ~1.37–2.44 fps because OCR blocks
   inference/UI/speech.
3. **Distance estimation is heuristic** — pinhole model with assumed 55° vertical FOV
   and per-class reference heights; **never calibrated against ground truth**.
4. **AI accuracy never formally measured** — no evaluation dataset, no precision /
   recall / mAP / CER / WER numbers.
5. **Code coverage not measured** — pytest-cov absent.
6. **No speech input** — STT / command parser / voice mode missing.
7. **Detection + OCR + rules only** — no higher-level scene understanding.
8. **No depth estimation.**
9. **No formal AI evaluation dataset.**
10. **No production deployment architecture** (no Docker/CI/CD/observability).
11. **Config inconsistencies** — `server/pipeline.py` hardcodes conf/iou thresholds;
    `camera_config.yaml` and `logging_config.yaml` not consumed by code.
12. **`VideoRecorder.start()`** doesn't validate the camera is running (writes an
    empty file with warning spam).
13. **Declared-but-missing deps** — `matplotlib` and `pytest-cov` in
    `requirements.txt` are not installed.
14. **No model manifest** — the ONNX file is git-ignored with no versioning/checksum
    metadata.
15. **Dashboard binds localhost only** (by design); no API auth, polling-based UI.

---

## 6. Existing Dependencies

Declared in `requirements.txt`; installed versions verified on this machine:

| Package | requirements.txt | Installed |
|---|---|---|
| opencv-python | >=4.9.0 | 5.0.0.93 |
| numpy | >=1.24.0 | 2.5.1 |
| Pillow | >=10.0.0 | 12.2.0 |
| matplotlib | >=3.7.0 | **not installed** (unused in src) |
| pytest | >=8.0.0 | 9.1.1 |
| pytest-cov | >=4.1.0 | **not installed** |
| PyYAML | >=6.0 | 6.0.3 |
| onnxruntime | >=1.18.0 | 1.28.0 |
| rapidocr-onnxruntime | >=1.2.0 | 1.2.3 |
| pyttsx3 | >=2.90 | 2.99 |
| Flask | >=3.0.0 | 3.1.3 |
| flask-cors | >=4.0.0 | 6.0.5 |

Python 3.13.14, Windows (win32). Model: `models/yolov8n.onnx` (12.8 MB, git-ignored,
input 640×640, 80 COCO classes, ONNX format, downloaded from the CVHub520
X-AnyLabeling releases).

---

## 7. Git / Repo State at Baseline

- Branch `feature/ai-productization` branched from `master` @ `755e6e7`
  ("release: mark project as VERSION 1.0").
- `master` is up to date with `origin/master`; `v1.0` tag exists.
- Working tree clean at the time of this document.
- Audit artifacts committed on master: `REPORT.md`, `TEST_RESULTS.md`,
  `PERFORMANCE_RESULTS.md`, `architecture-current.md`, `scripts/audit/*`,
  `VERSION` (1.0).

---

## 8. Definition of Done for Productization

The MVP becomes a **"credible AI-powered assistive vision prototype with a
production-oriented architecture"** when, at minimum:

- Existing 159 tests still pass.
- Coverage is measured (≥ 80% meaningful target per the roadmap).
- OCR performance improved or a justified alternative selected.
- Camera remains responsive.
- AI accuracy has measurable evaluation results.
- Distance estimation has calibration results.
- Speech input works.
- Safety engine works independently of any LLM.
- Optional VLM has offline fallback.
- No API key is hardcoded; no personal media committed.
- Performance metrics, failure modes, CI, README, and architecture docs reflect the
  actual implementation.
- All model licenses documented.