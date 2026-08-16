# Detection preprocessing audit (Phases 3-5)

## What the detector does today

`src/detection/detector.py` (`YoloDetector`, OpenCV DNN on an ONNX
export):

1. **Letterbox** — resize keeping aspect ratio to a 640x640 canvas filled
   with grey 114, padding centred.  `ratio` + `pad_x/pad_y` are carried
   so boxes map back to original pixels.
2. **Blob** — `cv2.dnn.blobFromImage(canvas, 1/255, (640,640),
   swapRB=True)` → BGR frame becomes RGB, normalised to [0,1], NCHW.
3. **Forward** — single-pass 640x640 inference.
4. **Decode** — the `[1, 84, 8400]` output is transposed; centre-xywh →
   corner rects; per-class confidence filter; `cv2.dnn.NMSBoxes`.
5. **Scale back** — subtract letterbox pads, divide by `ratio`, clamp to
   frame bounds.

### Audit result

The preprocessing chain is **correct** for YOLOv8 ONNX exports
(letterbox aspect-preserving, RGB swap, 1/255 normalisation, symmetric
padding, NMS, coordinate scale-back).  The visualiser
(`scripts/debug/detection_visualizer.py`) renders the original frame
next to the *actual* model input, so any future preprocessing bug is
visually obvious instead of hiding inside a "no detections" mystery.

### Issues fixed

| Issue | Before | After |
|---|---|---|
| Per-class confidence floors hardcoded | `laptop/tv/book/...` needed 0.5-0.55, silently dropping real objects | configurable `conf_overrides` (default empty); nothing silently discarded |
| Tall-laptop drop heuristic | always active — could delete real (open) laptops | `filter_tall_laptops: false` default; opt-in only |
| Config ignored by engine | engine hardcoded `conf=0.35, iou=0.45` | `conf_threshold`, `iou_threshold`, `conf_overrides`, `filter_tall_laptops` parsed from YAML and used |

## Phase 4 — confidence threshold sweep

`scripts/benchmark/perception_benchmark.py --mode sweep` re-parses the
same live frames at conf 0.20 → 0.50 and reports mean/max detections and
mean confidence per threshold.

Result on this machine (16 Aug): the current scene contained **0
detected objects at every threshold** — the room had nothing the model
recognises.  The sweep tool works; the *default* cannot honestly be
chosen until a scene with objects is measured.  The default remains
**conf 0.40** (documented, not claimed optimal).

## Phase 5 — model size trade-off (evidence so far)

| Model | Latency (live frames, uncontended) | Note |
|---|---|---|
| yolov8n | not re-measured this session | previously ~48-70 ms on synthetic 1280x720 |
| yolov8s | **mean ~84 ms** live, 1280x720 camera | 80-class COCO, better everyday-object coverage |

Both models are installed and recorded in `models/manifest.yaml` with
sha256 checksums.  On this CPU, yolov8s at ~84 ms inference still lets
the detection stage run at >10 Hz, so it remains the default; a head-to-
head mAP comparison belongs to the labelled evaluation set
(`docs/perception/evaluation.md`), not to guesswork.