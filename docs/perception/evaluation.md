# Evaluation — detection quality (Phases 14-17)

Full workflow, dataset structure, and labelling instructions live in
`evaluation/object_detection/README.md`.  This page summarises the
state and decisions.

## What is measured

`scripts/benchmark/object_detection_metrics.py` computes per-class and
aggregate **precision / recall / F1 / FP / FN / mAP@0.5** from real
detections vs. ground truth, writing `evaluation/object_detection/
results/metrics.json`.  Detections must match a same-class GT box at
IoU ≥ 0.5; AP uses the 11-point interpolated curve.

## Phase 16 — failure diagnosis (before training)

The current scene on this machine yielded **0 detections** at conf 0.40
(empty room) — a pipeline-working / scene-empty result, not a claim of
accuracy.  The diagnosis table in `evaluation/object_detection/README.md`
maps symptom → cause → pre-training action (threshold sweep, conf
overrides, label voting).

## Phase 17 — training decision

**No fine-tuning has been performed.**  Reasons, on evidence:

1. No labelled dataset exists to train against or to validate against.
2. No systematic, recurring class-miss has been observed — the only
   "0 detections" observation is an empty scene.
3. The pipeline fixes (threshold sweep, per-class conf_overrides, label
   voting) address the likely failure modes without retraining.

Policy when it becomes necessary: transfer-learn from stock pretrained
weights, never train from scratch; tune only classes with hundreds of
labels; keep the stock model as a fallback; compare mAP + latency on
the same evaluation set before switching the default.

## Honest status

- Tooling: ✅ exercised (refuses to fabricate without GT).
- Dataset: ❌ no labelled images yet — the next real-world step.
- Metrics claimed: none.