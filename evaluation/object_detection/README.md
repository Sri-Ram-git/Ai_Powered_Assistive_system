# Detection quality evaluation (Phases 14-17)

## Purpose

Measure the *real-world* detection quality of the pipeline and decide —
on evidence — whether training is needed.  The rule for this whole
hardening effort: **software tests prove code correctness; real labelled
scenes prove perception quality.  Never invent accuracy numbers.**

## Dataset structure

```
evaluation/object_detection/
    images/            *.jpg   — raw, unedited camera frames
    annotations/       *.json  — ground truth per image
    raw/               *._vis.png (diagnostic outputs from the visualiser)
    results/           metrics.json (written by the metrics script)
```

Annotation format (COCO-ish, pixel boxes `[x, y, w, h]` in frame
coordinates):

```json
[
  {
    "image": "scene_001.jpg",
    "objects": [
      { "label": "person", "box": [210, 90, 120, 300] },
      { "label": "chair",  "box": [520, 300, 90, 110] }
    ]
  }
]
```

## Collecting scenes

1. Capture frames with the live visualiser:
   `python scripts/debug/detection_visualizer.py --camera 0`
   (press `s` to save a frame to `assets/debug/`).
2. Copy useful frames into `evaluation/object_detection/images/`.
3. Label every object you care about.  Aim for a spread of lighting,
   distances, and clutter.  ~50+ labelled images is a meaningful start.

## Running the metrics

```
python scripts/benchmark/object_detection_metrics.py --conf 0.40
```

Reports per-class precision / recall / F1 / FP / FN and mAP@0.5 (IoU
match threshold 0.5, 11-point interpolated AP), plus the aggregate.
When no ground truth exists the script says so instead of guessing.

## Phase 16 — diagnose, don't retrain

Before any fine-tuning, classify the failure patterns on the real
scenes:

| Symptom | Likely cause | Action before training |
|---|---|---|
| Object not detected at all | small / far / heavy blur | check letterbox scaling; test conf 0.2-0.5 sweep |
| Detected but wrong class | fine-grained confusion | label voting absorbs it; note for future fine-tune |
| Box covers only part | occlusion / camera angle | check if it matters for guidance |
| Flicker / ID swap | tracking association | already addressed (class-consistent + smoothing) |
| Detections flood | threshold too low | raise conf; use conf_overrides per class |

Only when a *specific, recurring* class is systematically missed by
YOLOv8s should fine-tuning be considered (Phase 17).

## Phase 17 — fine-tuning policy

- Never train from scratch; start from `yolov8n/s.onnx` pretrained
  weights and do transfer learning on your own labelled set.
- Only classes with enough labels (hundreds of boxes each) are worth
  tuning.
- Retain the ability to fall back: keep both the stock model and the
  tuned export, and compare mAP + latency on the same evaluation set
  before switching the default.

## Current status (honest)

- **No ground truth has been labelled yet** in this repository, so no
  precision/recall/F1/mAP numbers are claimed.
- The tooling to produce them exists and is exercised (it correctly
  refuses to fabricate results without GT).
- The live camera scene captured during this hardening
  (`evaluation/object_detection/raw/scene_vis.png`) contains **0
  detected objects** at conf 0.40 — the room genuinely had none visible.
- Training is therefore **not currently justified**; the correct next
  step is collecting and labelling real scenes.