# Model Evaluation

## Purpose

This document records how model performance is measured for the
AI-powered assistive vision system, what the current (synthetic)
baseline is, and what is required before any accuracy claim can be
treated as meaningful.

All metrics are computed by `src/evaluation/` and reported through
`evaluation/scripts/run_evaluation.py`.

## Object detection (YOLOv8n ONNX, 80 COCO classes)

Metrics computed per image then averaged:

* **Precision** — TP / (TP + FP) at IoU 0.50 (greedy confidence-ranked
  matching, class must match).
* **Recall** — TP / (TP + FN) at IoU 0.50.
* **mAP@50** — mean Average Precision over classes at IoU 0.50.
* **mAP@50:95** — mean AP over IoU thresholds 0.50 → 0.95 step 0.05,
  averaged over classes.
* **False positives / false negatives** — counts.

### Current (synthetic) baseline

Dataset: `evaluation/datasets/synthetic.json` (6 synthetic images).

| Metric       | Value |
|--------------|-------|
| Precision    | 0.806 |
| Recall       | 1.000 |
| mAP@50       | 0.944 |
| mAP@50:95    | 0.903 |
| False pos.   | 3     |
| False neg.   | 0     |

**Caveat:** synthetic ground truth with near-perfect box overlap — these
numbers validate the *measurement pipeline*, not real-world YOLO
accuracy.  Real-world mAP requires a captured, manually-annotated
dataset (see `evaluation/README.md`).

## OCR (RapidOCR ONNX)

Metrics computed per image:

* **CER** — Levenshtein edit distance / reference character count.
* **WER** — edit distance / reference word count.
* **Detection success rate** — fraction of reference strings detected
  with ≥ 50% word overlap.

### Current (synthetic) baseline

| Metric                  | Value |
|-------------------------|-------|
| Character Error Rate    | 0.000 |
| Word Error Rate         | 0.000 |
| Detection success rate  | 1.000 |

The synthetic data feeds clean, rendered text — real-world OCR on noisy
signage/camera motion will be substantially worse.  Latency benchmark:
see `PERFORMANCE_RESULTS.md` and `docs/productization/performance.md`.

## Distance estimation

Metrics: **MAE**, **RMSE**, **relative error**, computed by
`src/navigation/calibration.py` and driven by `tools/calibrate_camera.py`
against `data/calibration.csv`.

The calibration machinery is validated (recovers the pinhole model to
MAE < 0.05 m on synthetic data).  **No real-world distance accuracy is
claimed yet** — the current CSV rows are synthetic.  Live calibration
must be run against a tape measure before reporting real numbers.

## End-to-end assistive behaviour

A small deterministic evaluation (`AssistiveCase` list) checks whether
the guidance system produced the expected *kind* of output.  Current
synthetic baseline: accuracy 0.60 (5 cases).  These cases are authored
scenarios, not measured real sessions.

## What is NOT claimed

* No production-grade accuracy claim is made for any perception model.
* Synthetic datasets are explicitly flagged in every report.
* "Trained" is never used for pretrained models (YOLOv8n, RapidOCR,
  and any future depth/VLM model are all pretrained).

## Required before accuracy can be trusted

1. Captured real frames (indoor + outdoor) from the target camera.
2. Manually verified ground-truth boxes + text + measured distances.
3. Hundreds of images (per class) for statistically meaningful mAP/CER.
4. Re-run `run_evaluation.py` and commit the updated `results/`.
5. Cross-check distance MAE/RMSE against live `calibrate_camera.py` runs.