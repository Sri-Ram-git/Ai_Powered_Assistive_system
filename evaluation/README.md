# AI Evaluation

This directory holds the formal AI-evaluation system for the assistive
vision product.  Its job is to produce **honest, reproducible** numbers
about model performance — not to manufacture impressive-sounding metrics.

```
evaluation/
├── datasets/       # ground-truth data (COCO-style JSON annotations)
├── annotations/    # (future) raw annotation files
├── scripts/        # runnable evaluation pipeline
├── results/        # generated metric reports (committed for transparency)
└── reports/        # human-readable evaluation reports
```

## What is measured

| Task                  | Metrics                                                         |
|-----------------------|-----------------------------------------------------------------|
| Object detection      | Precision, Recall, mAP@50, mAP@50:95, false positives, false negatives |
| OCR                   | Character Error Rate (CER), Word Error Rate (WER), detection success rate |
| Distance estimation   | MAE, RMSE, relative error (see `tools/calibrate_camera.py`)       |
| End-to-end assistive  | Correct/incorrect guidance, missed object, false warning, response latency |

The metric implementations live in `src/evaluation/` and are covered by
unit tests in `tests/evaluation/`.

## How to run

```bash
# Full report (uses the synthetic dataset, writes evaluation/results/report.json)
python evaluation/scripts/run_evaluation.py
```

## Dataset status

The current dataset (`datasets/synthetic.json`) is **synthetic and tiny**
(6 images).  It exists to:

1. Exercise the evaluation pipeline end-to-end in CI / on any machine.
2. Provide a schema others can follow when adding real data.

Because it is synthetic and small, **no statistically meaningful accuracy
claim may be derived from it.**  Every report it produces includes an
explicit caveat to that effect.  Real evaluation requires:

* captured frames from the target camera (indoor + outdoor);
* manually verified ground-truth boxes and text;
* a dataset of hundreds of images for meaningful mAP/CER numbers;
* measured distances for distance-accuracy evaluation.

## Annotation schema (COCO-style, JSON)

```json
{
  "id": "image_001",
  "ground_truth": [
    {"label": "person", "box": [400, 200, 60, 220]}
  ],
  "predictions": [
    {"label": "person", "confidence": 0.87, "box": [402, 202, 58, 216]}
  ],
  "reference_text": ["EXIT"],
  "recognised_text": ["EXIT"]
}
```

Boxes are `[x, y, w, h]` in image pixels.

## Ground rules

* Never modify `results/` numbers by hand — regenerate them.
* Never claim model accuracy that isn't backed by this evaluation.
* State dataset size when reporting metrics (small datasets => caveats).