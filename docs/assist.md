# Assistive Vision App (`src/assist`)

## Overview

The Week 2-3 end-to-end integration. It wires the full pipeline:

```
Camera → Object Detection (YOLOv8 ONNX) → Tracking (IoU)
       → OCR (RapidOCR) → Decision Engine → Speech Output
```

Detections are drawn as boxes with **stable track IDs and live distance**,
OCR text is overlaid, and the decision engine + tracking monitor speak
guidance (including re-announced distances as objects move) through the
OS speech engine.
A draggable monochrome HUD (reused from `src/camera`) shows the mode,
object/text counts, and live FPS.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        assist/                              │
│  assist_app.py   main()                                     │
│                                                             │
│  Camera ──▶ YoloDetector.detect ──▶ IoUTracker.update       │
│      │                                   │                 │
│      └──▶ OcrEngine.read_text ──▶ TrackingMonitor.events   │
│      │          │                        │                 │
│  FrameSummary(tracks, ocr, w, h) ◀───────┘                 │
│        │                                                   │
│  DecisionEngine.decide ──▶ SpeechOutput.speak             │
│        │                                                   │
│  HUD.render ──▶ cv2.imshow ◀── track/OCR boxes             │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

All knobs live in `configs/assist_config.yaml`:

| Section | Keys |
|---|---|
| `detection` | `model_path`, `input_size`, `conf_threshold`, `iou_threshold`, `every_n_frames` |
| `tracking` | `iou_threshold`, `max_missed`, `distance_change_metres`, `min_announce_interval` |
| `ocr` | `min_confidence`, `max_boxes`, `every_n_frames`, `ask_before_reading` |
| `navigation` | `vertical_fov`, `reference_heights` |
| `speech` | `rate`, `volume` |
| `decision` | `cooldown_seconds`, `min_priority`, `speak_ocr_text`, `max_ocr_chars` |
| `camera` | `id`, `resolution` |
| `app` | `scale_display_to`, `jpeg_width`, `jpeg_quality` |

## Performance tuning

- `camera.resolution: [1280, 720]` keeps a sharp live feed; YOLO and OCR
  run on the raw frame but are throttled below.
- `detection.every_n_frames` throttles YOLO inference (every N frames).
- `ocr.every_n_frames` throttles OCR (the slowest stage).
- `tracking.*` controls how often distance re-announcements happen.
- `app.scale_display_to` caps the *display* resolution; the AI always
  runs on the raw camera frame.

## Usage

```bash
python src/assist/assist_app.py [--camera 0] [--config configs/assist_config.yaml] [--model models/yolov8n.onnx]
```

## Keys

| Key | Action |
|---|---|
| `m` | Mute / unmute speech |
| `t` | Toggle OCR mode: ask-before-read vs auto-read |
| `r` | Read the most recently detected text aloud |
| `s` | Save annotated screenshot to `assets/screenshots/assist/` |
| `space` | Reset the decision cooldown + tracking |
| `q` | Quit |

## OCR mode

Default `ocr.ask_before_reading: false`: as soon as the camera sees text,
the app reads it aloud automatically ("Text says, ..."). Set
`ask_before_reading: true` to instead announce "Text detected. Press R to
hear it read." and only speak the text when you press `r`.

## Distance accuracy

`navigation.vertical_fov` (degrees) drives the pinhole distance model —
set it to your webcam's real vertical FOV for correct metres. Wider FOV
gives shorter estimates. Optional `navigation.reference_heights` override
per-class heights, e.g.:

```yaml
navigation:
  vertical_fov: 55
  reference_heights:
    person: 1.75
```

## Dependencies

- All Week 1 modules (`camera`, `hud`)
- `detection`, `tracking`, `ocr`, `decision`, `audio`, `navigation`
- `models/yolov8n.onnx` (git-ignored; download via the URL in the config)
- `PyYAML`

## Limitations

- CPU inference limits FPS (detection on every N frames, OCR every Mth).
- Speech rate is limited by the decision cooldown + tracking monitor
  interval to avoid spam.
- IoU tracking can merge identical objects that fully overlap for long
  stretches (no appearance features).
- Requires a working webcam and a system TTS voice.

## Future extensions

- Appearance-based re-identification to survive full occlusions.
- Region-of-interest scanning (read only one zone at a time)
- Haptic / vibration feedback for critical alerts
