"""Formal AI evaluation of the perception modules.

This package computes honest, reproducible metrics for the three
perception tasks plus end-to-end assistive behaviour.  It does NOT
manufacture numbers: if the dataset is too small for a statistically
meaningful claim, the reporting scripts say so explicitly.

Modules:
    detection_metrics — precision, recall, mAP@50, mAP@50:95, FP/FN.
    ocr_metrics       — character/word error rate, detection success.
    distance_metrics  — MAE, RMSE, relative error (re-uses calibration).
    assistive_metrics — correct/incorrect guidance, missed object, false
                        warning, response latency.

The dataset lives under ``evaluation/datasets/`` with COCO-style JSON
annotations; see ``evaluation/README.md`` for the schema.
"""
from src.evaluation.detection_metrics import (
    evaluate_detections,
    mean_average_precision,
)
from src.evaluation.ocr_metrics import (
    aggregate_ocr_metrics,
    character_error_rate,
    exact_match,
    text_detection_success,
    word_error_rate,
)

__all__ = [
    "aggregate_ocr_metrics",
    "character_error_rate",
    "evaluate_detections",
    "exact_match",
    "mean_average_precision",
    "text_detection_success",
    "word_error_rate",
]