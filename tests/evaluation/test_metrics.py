"""Tests for the AI evaluation metrics (pure, hardware-free)."""
import pytest

from src.evaluation.assistive_metrics import (
    AssistiveCase,
    evaluate_assistive,
)
from src.evaluation.detection_metrics import Box, evaluate_detections
from src.evaluation.ocr_metrics import (
    character_error_rate,
    text_detection_success,
    word_error_rate,
)


class TestDetectionMetrics:
    def test_perfect_match(self):
        gt = [Box("person", 1.0, (0, 0, 10, 10))]
        pred = [Box("person", 0.9, (0, 0, 10, 10))]
        m = evaluate_detections(pred, gt)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["mAP@50"] == 1.0
        assert m["false_positives"] == 0
        assert m["false_negatives"] == 0

    def test_wrong_class_is_false_positive(self):
        gt = [Box("person", 1.0, (0, 0, 10, 10))]
        pred = [Box("car", 0.9, (0, 0, 10, 10))]
        m = evaluate_detections(pred, gt)
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["false_negatives"] == 1
        assert m["false_positives"] == 1

    def test_low_iou_counts_as_false_positive(self):
        gt = [Box("person", 1.0, (0, 0, 100, 100))]
        pred = [Box("person", 0.9, (100, 100, 10, 10))]  # no overlap
        m = evaluate_detections(pred, gt)
        assert m["recall"] == 0.0

    def test_empty(self):
        m = evaluate_detections([], [])
        assert m["precision"] == 0.0
        assert m["mAP@50"] == 0.0


class TestOcrMetrics:
    def test_cer_perfect(self):
        assert character_error_rate("EXIT", "EXIT") == 0.0

    def test_cer_known(self):
        # "EXIT" vs "EXIT." => 1 extra char / 4
        assert character_error_rate("EXIT", "EXIT.") == pytest.approx(0.25)

    def test_wer_perfect(self):
        assert word_error_rate("DO NOT WALK", "DO NOT WALK") == 0.0

    def test_detection_success(self):
        assert text_detection_success(["EXIT"], ["EXIT"]) == 1.0
        assert text_detection_success(["EXIT"], ["HELLO"]) == 0.0

    def test_detection_success_partial_word_overlap(self):
        # "DO NOT WALK" vs "DO NOT RUN": 2/3 words => >=0.5 => hit
        assert text_detection_success(["DO NOT WALK"], ["DO NOT RUN"]) == 1.0


class TestAssistiveMetrics:
    def test_correct_and_incorrect(self):
        cases = [
            AssistiveCase("obstacle", "obstacle", "Obstacle ahead"),
            AssistiveCase("person", "person", "Nothing detected"),
        ]
        m = evaluate_assistive(cases)
        assert m["accuracy"] == 0.5
        assert m["correct_guidance"] == 1
        assert m["incorrect_guidance"] == 1

    def test_keyword_fallback(self):
        cases = [AssistiveCase("stop sign", "stop sign",
                               "Stop sign ahead", produced_keywords=["stop"])]
        m = evaluate_assistive(cases)
        assert m["accuracy"] == 1.0

    def test_empty(self):
        m = evaluate_assistive([])
        assert m["cases"] == 0