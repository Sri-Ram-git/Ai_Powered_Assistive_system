# Architecture

## System goal (Week 3)

```
Camera → Object Detection → Tracking → OCR → Decision Engine → Speech Output
```

Week 1 delivers the vision foundation; Weeks 2-3 add the AI modules.
The full pipeline is integrated in `src/assist/assist_app.py` and served
to a browser dashboard by `src/server/`.

## Week 1 module map

```
┌─────────────────────────────────────────────────────────────────┐
│                        assistive-vision-system                   │
├───────────────┬───────────────┬───────────────┬─────────────────┤
│  src/camera   │ image_        │ image_        │  morphology/    │
│               │ fundamentals  │ processing    │                 │
│  acquisition  │  utilities    │  filters/      │  shapes &       │
│  HUD, record  │  I/O, colour  │  edges/noise   │  contours       │
├───────────────┴───────────────┴───────────────┴─────────────────┤
│                      src/playground (Week 1 integration)         │
│            src/utils (logger, exceptions) — shared               │
├───────────────┬───────────────┬───────────────┬─────────────────┤
│  detection/   │  ocr/         │  decision/    │  audio/         │
│  YOLOv8 ONNX  │  RapidOCR     │  rule engine  │  TTS (pyttsx3)  │
├───────────────┼───────────────┼───────────────┴─────────────────┤
│  tracking/    │  navigation/  │  guidance + continuous speech   │
│  IoU tracker  │  cues         │                                  │
├───────────────┴───────────────┴─────────────────────────────────┤
│  src/assist (desktop app)      src/server (web dashboard)        │
│  tests/ | docs/ | configs/ | assets/ | logs/                     │
└─────────────────────────────────────────────────────────────────┘
```

## Layer rules

- **Stateless functions** in `image_fundamentals`, `image_processing`,
  and `navigation`: input → output, no side effects, no global state.
- **Stateful objects** only where required: `Camera` (device handle),
  `HUD` (presentation state), `VideoRecorder` (background thread),
  `ShapeDetector` (tuning parameters), `YoloDetector` (loaded model),
  `OcrEngine` (model), `SpeechOutput` (worker thread), `DecisionEngine`
  (cooldown state).
- **No cross-module circular imports.** Dependency direction:
  `utils ← camera`, `utils ← image_fundamentals ← image_processing ←
  morphology ← playground`, and `utils ← detection / ocr / audio ←
  navigation ← decision ← assist`.
- **Presentation vs logic:** HUD only draws; it never touches the
  camera or pipeline.  The playground / assist app orchestrates.

## Data flow (live feed)

```
Camera.read() ──▶ [filter] ─▶ [gray] ─▶ [edge] ─▶ [thresh]
                                    │
                    scale_to_fit ───┘
                                    │
                              HUD.render() ──▶ cv2.imshow
```

## Week 2-3 pipeline (assist app)

```
Camera.read() ──▶ YoloDetector.detect (every N frames) ──┐
      │                                                 │
      └──▶ OcrEngine.read_text (every M frames) ──▶ draw_text_boxes
                          │                             │
          IoUTracker.update ─▶ TrackingMonitor.events ──┤
                          │                             │
          FrameSummary(tracks, ocr, w, h)               │
                          │                             │
                    DecisionEngine.decide                │
                          │                             │
                    SpeechOutput.speak                   │
                          │                             │
               HUD.render ──▶ cv2.imshow ◀──────────────┘
```

## Web dashboard (src/server)

The same pipeline runs in a background thread (`PipelineServer`) and is
exposed to the browser:

```
Camera.read() ──▶ detect ─▶ track ─▶ monitor ─▶ decision ─▶ speech
      │                                                   │
      └── annotate ─▶ JPEG ─▶ /video_feed (MJPEG)         │
                                └▶ state dict ─▶ /api/state
                                                  │
                                        dashboard HTML (polls /api/state)
```

- `PipelineServer` (background thread): camera → detection → tracking →
  OCR → decision; stores the latest annotated JPEG and a JSON state
  snapshot under locks.
- Flask routes in `src/server/app.py`: `/` (dashboard), `/video_feed`
  (MJPEG stream), `/api/state` (JSON with detections, distances,
  guidance, FPS, latency).
- The dashboard is a dark-theme single page: live camera feed with
  REC indicator, AI Guidance card (spoken phrase), Current Detections
  list (track ID, label, confidence, distance), and a footer with
  FPS / resolution / AI status / latency.

## Module map (Weeks 2-3)

| Package | Slot | Feeds |
|---|---|---|
| `src/detection` | YOLOv8 object detection (OpenCV DNN + ONNX) | tracking, decision |
| `src/tracking` | IoU multi-object tracking + change monitor | decision, speech |
| `src/ocr` | text recognition (RapidOCR, CPU ONNX) | decision |
| `src/decision` | rule-based decisions with cooldown | speech |
| `src/audio` | TTS / speech synthesis (pyttsx3) | user |
| `src/navigation` | guidance cues (direction, distance, signs) | tracking, decision |
| `src/assist` | end-to-end integration app (desktop) | user |
| `src/server` | web dashboard (Flask: MJPEG + state API) | user |

## Configuration & observability

- All settings are externalised to `configs/*.yaml`
  (`camera_config.yaml`, `logging_config.yaml`, `assist_config.yaml`).
- All modules log through `src.utils.logger.setup_logger` to console +
  `logs/app.log`.
- Custom exception hierarchy in `src/utils/exceptions.py`.
- Model weights (`models/*.onnx`) are git-ignored and re-downloaded.
