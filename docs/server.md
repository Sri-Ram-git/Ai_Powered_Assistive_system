# Web Dashboard Module (`src/server`)

## Overview

A Flask server that runs the full AI vision pipeline in a **background
thread** and exposes it to a browser dashboard.  The user sees the live
annotated camera feed (MJPEG) plus real-time AI state: tracked objects
with distances, the spoken guidance phrase, OCR text, FPS, and latency.

```
python src/server/app.py --config configs/assist_config.yaml --port 5000
```

Open <http://127.0.0.1:5000/>.

## Architecture

```
┌───────────────────────────── PipelineServer thread ─────────────────────────┐
│  Camera.read()                                                              │
│    └─▶ YoloDetector.detect  (every `detect_every` frames)                   │
│    └─▶ IoUTracker.update     (stable track IDs)                             │
│    └─▶ OcrEngine.read_text  (every `ocr_every` frames)                      │
│    └─▶ TrackingMonitor.events + DecisionEngine.decide  → guidance phrase    │
│    └─▶ annotate(frame, tracks, ocr) → JPEG  (latest_jpeg)                   │
│    └─▶ state dict (detections, guidance, fps, latency, resolution)          │
└──────────────────────────────────────────────────────────────────────────────┘
                              │                        │
              /video_feed (MJPEG)            /api/state (JSON)
                              └─────────▶ dashboard HTML/JS
```

## Files

| File | Purpose |
|---|---|
| `pipeline.py` | `PipelineConfig` (YAML-driven) + `PipelineServer` (the background loop, thread-safe `latest_jpeg` / `state_snapshot`) |
| `app.py` | Flask app: `/`, `/video_feed`, `/api/state`, CLI entry point; wires TTS |

## Endpoints

| Route | Returns |
|---|---|
| `GET /` | The dashboard page (embedded HTML/CSS/JS) |
| `GET /video_feed` | MJPEG stream — multipart JPEG frames of the annotated feed |
| `GET /api/state` | JSON: `running`, `fps`, `resolution`, `latency_ms`, `error`, `detections[]`, `ocr_text`, `guidance` |

### `detections[]` item

```json
{
  "track_id": 0,
  "label": "person",
  "confidence": 0.91,
  "distance": 3.4,
  "direction": "ahead"
}
```

## Dashboard UI

Single dark-theme page (no build step):

- **Camera feed** (75%) with REC indicator and `object-fit: contain`.
- **AI Guidance card** (25%): the current spoken phrase with a live
  speaker animation.
- **Current Detections list**: track tag, label, direction, confidence,
  distance in metres.
- **Footer**: FPS, resolution, AI status, and latency in ms.

The JS polls `/api/state` every ~700 ms and updates the feed image from
the MJPEG stream.

## Configuration

`PipelineConfig` is populated from `configs/assist_config.yaml` — the
same file the desktop app uses.  Notable keys:

| Section | Keys |
|---|---|
| `detection` | `model_path`, `every_n_frames` |
| `tracking` | `iou_threshold`, `max_missed`, `distance_change_metres`, `min_announce_interval` |
| `ocr` | `every_n_frames`, `min_confidence` |
| `decision` | `cooldown_seconds`, `min_priority`, `speak_ocr_text` |
| `camera` | `id`, `resolution` |
| `app` | `jpeg_width`, `jpeg_quality` (MJPEG stream size/quality) |

Model path is resolved relative to the project root, so
`models/yolov8n.onnx` in the config works from any working directory.

## Thread-safety

The pipeline thread writes the annotated JPEG and state under
`threading.Lock`s; Flask readers call `latest_jpeg` / `state_snapshot`
which return copies.  This keeps the camera model untouched by HTTP
traffic — one slow client cannot block detection.

## Dependencies

- Flask (web server)
- OpenCV + NumPy (frames, annotation, JPEG encoding)
- YOLOv8 ONNX model (`models/yolov8n.onnx`, git-ignored — download via
  the README curl command)
- Optional: pyttsx3 (TTS — the dashboard speaks the guidance too)

## Tests

`tests/test_server.py` (endpoints + config/helpers with a stub pipeline)
and `tests/test_pipeline_e2e.py` (full pipeline loop with a stub camera
and the real ONNX model — no webcam required).

## Future extensions

- WebSocket push instead of MJPEG + polling for lower latency.
- Confidence/distance charts over time.
- Manual multi-camera selection and per-camera config.
