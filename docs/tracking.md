# Tracking Module (`src/tracking`)

## Overview

Two small, dependency-light pieces that give the pipeline a sense of
**object identity and change over time**:

- `tracker.py` — `IoUTracker`: associates detections frame-to-frame so
  each object keeps a **stable ID**.
- `monitor.py` — `TrackingMonitor`: turns tracked objects into **spoken
  guidance phrases** whenever something changes (new object, distance
  moved, direction changed).

No ML tracking models or extra dependencies — just NumPy + the IoU of
bounding boxes.  Detections arrive every few frames (pipeline throttling)
so the IoU threshold is deliberately modest.

## Architecture

```
┌────────────────────────────────────────────────┐
│               tracking/                        │
│  tracker.py  IoUTracker                        │
│    update(detections) → List[TrackedObject]    │
│    active_tracks / all_tracks / reset          │
│                                                │
│  monitor.py  TrackingMonitor                   │
│    events(tracks, w, h, now) → List[str]       │
│    reset                                       │
└────────────────────────────────────────────────┘
        │ feeds                              │
        ▼                                    ▼
  decision engine                     speech / dashboard
```

## IoUTracker

### Association

Each frame, detections are matched to existing tracks **greedily**:

1. Detections sorted by confidence (highest first).
2. Each detection picks the track with the best IoU **above
   `iou_threshold`** (default `0.3`).
3. A matched track inherits the detection's box, confidence, and centre.
4. Unmatched detections spawn new tracks; unmatched tracks increment
   their `missed` counter.
5. Tracks with `missed > max_missed` (default `8`) are dropped.

| Parameter | Default | Meaning |
|---|---|---|
| `iou_threshold` | `0.3` | Min IoU to associate a detection with a track |
| `max_missed` | `8` | Drop a track after this many unmatched frames |

### TrackedObject

| Field | Meaning |
|---|---|
| `track_id` | Stable integer identity (monotonic) |
| `label` | Class label (person, car, …) |
| `box` | `(x, y, w, h)` in frame coords |
| `confidence` | Latest detection confidence |
| `age` | Frames the track has lived |
| `missed` | Consecutive frames without a match |
| `alive` | `True` when `missed == 0` |

## TrackingMonitor

Emits phrases **only on change**, so the user gets continuous (not
one-off) guidance without chatter:

| Event | Phrase |
|---|---|
| New object | `Person ahead, about 3 metres` |
| Distance moved ≥ `distance_change_metres` | `Person now ahead, about 2 metres` |
| Direction changed | `Person now left, about 2 metres` |

Per-track throttling: a track will not re-announce within
`min_announce_interval` seconds, and small distance jitter is absorbed.

| Parameter | Default | Meaning |
|---|---|---|
| `distance_change_metres` | `1.0` | Min distance change before re-announcing |
| `min_announce_interval` | `3.0` | Min seconds between phrases for one track |

Distance uses the pinhole model from `src.navigation` (label-specific
reference heights); direction uses the frame thirds.

## Usage

```python
from src.tracking import IoUTracker, TrackingMonitor
from src.detection import YoloDetector

detector = YoloDetector("models/yolov8n.onnx")
tracker = IoUTracker()
monitor = TrackingMonitor()

for frame in frames:
    detections = detector.detect(frame)
    tracks = tracker.update(detections)
    phrases = monitor.events(tracks, frame.shape[1], frame.shape[0])
    for p in phrases:
        print(p)          # -> speech.speak(p)
```

## Configuration

All knobs live in `configs/assist_config.yaml` under `tracking:` and are
picked up by both the desktop app and the web dashboard.

## Dependencies & limitations

- Depends on: `src.detection` (DetectionResult), `src.navigation`
  (distance/direction), `src.utils.logger`.
- Pure IoU — IDs can swap if objects visually collide and overlap, and
  throttled detection means fast motion may briefly miss a track.
- No appearance features (colour/histogram) are used, so identical
  objects that fully overlap for several frames may be merged.

## Tests

`tests/test_tracking.py` — association, persistence, missing/drop
behaviour, monitor phrase generation, throttling, and reset.

## Future extensions

- Appearance-based re-identification (colour histograms) to survive
  full occlusions.
- Kalman-filtered box smoothing for less jittery distances.
- Track lifetime statistics (e.g. "the car stopped" via speed estimate).
