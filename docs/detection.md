# Object Detection Module (`src/detection`)

## Overview

Real-time object detection using YOLOv8 exported to ONNX and executed
with OpenCV's DNN module. No PyTorch, CUDA, or GPU is required — the
model runs on CPU via OpenCV.

## Architecture

```
┌──────────────────────────────────────────────┐
│              detection/                      │
│  detector.py                                 │
│    YoloDetector                              │
│      detect(frame) → List[DetectionResult]   │
│    label_detections(frame, results)          │
└──────────────────────────────────────────────┘
```

The detector owns the model, the inference loop, and the decode step:
letterbox → forward pass → confidence filter → NMS → scale-back.

## Detection algorithm

1. **Letterbox** the frame to the model's square input (640×640) while
   preserving aspect ratio and padding with 114 (grey).
2. **Forward pass** — a single ONNX inference producing `[1, 84, 8400]`
   (84 = 4 box coords + 80 COCO class probabilities per anchor).
3. **Decode** — transpose to `[8400, 84]`, take the argmax class and
   its score per anchor.
4. **Confidence filter** — drop anchors below `conf_threshold`.  Some
   classes (`laptop`, `tv`, `book`, ...) get a higher per-class bar
   (`_HIGH_CONF_CLASSES`) because the model tends to fire them on plain
   rectangles.
5. **NMS** — `cv2.dnn.NMSBoxes` with `iou_threshold` to suppress
   overlapping boxes.
6. **False-laptop filter** — a "laptop" box that is much taller than
   wide (aspect h > 1.15·w) is dropped; that shape is a door/wall, not a
   laptop.
7. **Scale back** — convert model-space (cx, cy, w, h) boxes back to
   original-frame (x, y, w, h), undoing the letterbox offset.

## Data model

`DetectionResult`:

| Field | Description |
|---|---|
| `label` | COCO class name (e.g. "person", "bus") |
| `confidence` | Class probability after NMS |
| `box` | `(x, y, w, h)` in original frame coordinates |
| `category` | *Property* — coarse group for the decision engine: `person`, `vehicle`, `obstacle`, `traffic signal`, or `object` |
| `center` | *Property* — box centre `(cx, cy)` |
| `area` | *Property* — `w * h` |

## Function reference

| Member | Description |
|---|---|
| `YoloDetector(model_path, input_size=640, conf_threshold=0.4, iou_threshold=0.45)` | Loads the ONNX model |
| `detect(frame)` | → `List[DetectionResult]` sorted by confidence |
| `class_names` | The 80 COCO class labels |
| `label_detections(frame, results)` | Annotated copy with boxes + labels |

## Usage

```python
from src.detection import YoloDetector, label_detections

detector = YoloDetector("models/yolov8n.onnx")
results = detector.detect(frame)          # List[DetectionResult]
display = label_detections(frame, results)
```

## Execution flow

```
python src/assist/assist_app.py    # live assist pipeline (includes detection)
```

## Dependencies

- Python 3.11+, OpenCV (`cv2.dnn`), NumPy
- `models/yolov8n.onnx` — download via the URL in `configs/assist_config.yaml`
  (git-ignored; re-downloaded, never committed)

## Limitations

- CPU inference only — frame rate depends on the machine (~15-30 FPS
  at 640×640 for YOLOv8n on a modern laptop).
- Detection quality is capped by the YOLOv8n model and COCO classes;
  the model does not know custom classes.
- The distance estimate in `navigation` is a heuristic, not calibrated.

## Future extensions

- TRT / OpenVINO backends for higher FPS
- YOLOv8 segmentation (`-seg`) for pixel-level obstacles
- Custom fine-tuned model for assistive-device-specific classes
