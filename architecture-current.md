# architecture-current.md — Current System Architecture (as audited)

**Date:** 16 Aug 2026
**Source:** verified code inspection (`src/`), configs, tests, and live execution.
This is a factual description of the architecture **as it exists today**.

## 1. High-level data flow

```
                    ┌─────────────────────────────────────────────┐
                    │                 src/assist/assist_app.py    │
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

## 2. Two runtimes share the same modules

### A. Desktop app — `src/assist/assist_app.py`
- One loop per frame: read → (every 2 frames) detect+track → (every 10 frames) OCR →
  monitor.events() → decision.decide() → tts.speak() → draw tracks/HUD → imshow.
- OCR runs in-band, so OCR latency stalls the whole loop (~1.4 fps measured).
- Keyboard: mute, OCR mode, read-text, screenshot, reset, quit. Draggable HUD.

### B. Web dashboard — `src/server/app.py` + `src/server/pipeline.py`
- `PipelineServer` runs **three threads**:
  - *coordinator* (`_run`): boots camera, spawns grab + infer, waits on stop event.
  - *grab thread* (`_grab_loop`): reads frames, annotates with latest AI results,
    JPEG-encodes → `latest_jpeg` for `/video_feed`. Never runs AI. Reports feed FPS.
  - *inference thread* (`_infer_loop`): detect/track/OCR/decision on the latest
    shared frame; publishes JSON state (`detections`, `ocr_text`, `guidance`,
    `latency_ms`) and invokes `speech_callback` (TTS).
- Flask endpoints: `/` (dashboard HTML), `/video_feed` (MJPEG), `/api/state` (JSON).
- Live feed stays responsive (~9 fps) because grab and inference are decoupled.

## 3. Module map (source of truth: `src/`)

| Package | Key types | Notes |
|---|---|---|
| `src/camera` | `Camera`, `CameraManager`, `CameraInfo`, `HUD`, `Canvas`, `FontManager`, `VideoRecorder`, utils | CAP_DSHOW, mirror default; PIL-drawn HUD |
| `src/image_fundamentals` | stateless fns in `image_utils.py` | I/O, transforms, colour, stats, hist |
| `src/image_processing` | stateless fns in `processing.py` | blur/threshold/edges/enhance/noise |
| `src/morphology` | `contour_utils`, `ShapeDetector` | erode/dilate/open/close, shapes |
| `src/detection` | `YoloDetector`, `DetectionResult` | ONNX via cv2.dnn; 80 COCO classes |
| `src/tracking` | `IoUTracker`, `TrackedObject`, `TrackingMonitor` | pure-NumPy IoU; change-based phrases |
| `src/ocr` | `OcrEngine`, `OcrResult` | RapidOCR CPU |
| `src/navigation` | stateless fns in `guidance.py` | direction, pinhole distance, cues |
| `src/decision` | `DecisionEngine`, `evaluate`, `FrameSummary`, `Decision`, `cue_identity` | priority + cooldown + dedup |
| `src/audio` | `SpeechOutput` | pyttsx3 worker, startLoop(False)+iterate pump |
| `src/assist` | `main()` | desktop end-to-end |
| `src/server` | `PipelineServer`, `PipelineConfig`, `create_app` | Flask + threaded pipeline |
| `src/playground` | `main()` | Week-1 live filter app |
| `src/utils` | `setup_logger`, exception hierarchy | logging to `logs/app.log` |

## 4. Configuration flow

- `configs/assist_config.yaml` → `assist_app.main()` (per-section dicts) and
  `PipelineConfig.from_yaml()` (server).
- `assist_app` reads `detection.conf_threshold` (default 0.35 if absent) and
  `navigation.vertical_fov` (default 55.0).
- `PipelineConfig.from_yaml` also pushes optional `navigation.reference_heights`
  into `guidance._REFERENCE_HEIGHTS`.
- `camera_config.yaml` and `logging_config.yaml` exist but are **not consumed** by
  code (constructor defaults / hardcoded logger settings used instead).

## 5. Speech guidance path (the recent fixes)

1. `TrackingMonitor.events()` returns phrases for changed objects
   ("Person ahead, about 3 metres").
2. `DecisionEngine.decide(summary, already_spoken=phrases)`:
   - `evaluate()` builds cues for **all** detections (incl. generic objects like
     "cell phone") via `scene_cues`.
   - dedups by `cue_identity()` (strips "about N metres"/"far away"/… suffixes) so
     distance jitter never re-triggers the same phrase.
   - skips any decision whose identity matches `already_spoken` (one narration per
     object per frame).
   - applies global `cooldown_seconds` and `min_priority` gating.
3. Result goes to `SpeechOutput.speak()` → worker thread → SAPI5.

## 6. Data & persistence

- No database, no on-disk state. Everything is in-memory (tracks, memory, cooldowns).
- Screenshots/recordings are written to `assets/` (git-ignored).
- Logs to `logs/app.log`.

## 7. Security & privacy posture

- `.gitignore` blocks personal media (`assets/*`), ONNX weights, and logs.
- Only the synthetic sample scene is versioned.
- Dashboard binds to `127.0.0.1` by default; no auth — appropriate for localhost only.
- TTS/OCR/detection all run locally (no cloud calls observed).

## 8. Deployment model

- **Not packaged**: no `setup.py`/`pyproject.toml`, no installer, no Dockerfile.
- Runs from source: `pip install -r requirements.txt`, download
  `models/yolov8n.onnx`, then `python src/assist/assist_app.py` or
  `python src/server/app.py --config configs/assist_config.yaml --port 5000`.
- The Flask server is the natural API surface for a mobile client (Flutter/RN):
  `/api/state` (JSON) and `/video_feed` (MJPEG) are already the needed endpoints.