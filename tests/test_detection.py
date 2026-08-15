"""Unit tests for the object-detection module (hardware-free)."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.detection.detector import (
    COCO_NAMES,
    DetectionResult,
    YoloDetector,
    _looks_like_false_laptop,
    label_detections,
)
from src.utils.exceptions import DetectionError

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "yolov8n.onnx"


@pytest.fixture(scope="module")
def detector():
    if not MODEL_PATH.exists():
        pytest.skip("yolov8n.onnx not present")
    return YoloDetector(str(MODEL_PATH), conf_threshold=0.25)


class TestLetterbox:
    def test_preserves_aspect_ratio(self):
        d = YoloDetector(str(MODEL_PATH)) if MODEL_PATH.exists() else None
        if d is None:
            pytest.skip("model missing")
        blob, ratio, pad_x, pad_y = d._letterbox(
            np.zeros((300, 600, 3), dtype=np.uint8))
        assert blob.shape == (1, 3, 640, 640)
        # 600 wide -> scales to 640 wide, 320 tall, padded vertically
        assert ratio == pytest.approx(640 / 600, rel=0.01)
        assert pad_y > 0 and pad_x == 0

    def test_empty_input_safe(self, detector):
        assert detector.detect(None) == []
        assert detector.detect(np.zeros((0, 0, 3), dtype=np.uint8)) == []


class TestParseOutputs:
    def _make_pred(self, n=8400):
        pred = np.zeros((84, n), dtype=np.float32)
        # centre-x, centre-y, width, height in 640-space
        pred[:4, 0] = [320, 320, 100, 200]
        pred[0, 0] = 320.0
        pred[4:, 0] = np.random.rand(80) * 0.001
        pred[4 + 0, 0] = 0.9  # class 0 (person)
        return pred

    def test_single_high_conf_detection(self):
        d = YoloDetector(str(MODEL_PATH)) if MODEL_PATH.exists() else None
        if d is None:
            pytest.skip("model missing")
        pred = self._make_pred()
        results = d._parse_outputs(pred, ratio=1.0, pad_x=0, pad_y=0,
                                   orig_w=640, orig_h=640)
        assert len(results) >= 1
        assert results[0].label == "person"
        assert results[0].confidence > 0.8

    def test_low_conf_filtered(self):
        d = YoloDetector(str(MODEL_PATH), conf_threshold=0.99) \
            if MODEL_PATH.exists() else None
        if d is None:
            pytest.skip("model missing")
        pred = self._make_pred()
        results = d._parse_outputs(pred, 1.0, 0, 0, 640, 640)
        assert results == []

    def test_scale_back_undoes_letterbox(self):
        d = YoloDetector(str(MODEL_PATH)) if MODEL_PATH.exists() else None
        if d is None:
            pytest.skip("model missing")
        pred = self._make_pred()
        # ratio 2 -> model box of 100x200 at 320,320 maps to original 640x640
        results = d._parse_outputs(pred, ratio=2.0, pad_x=64, pad_y=64,
                                   orig_w=1280, orig_h=1280)
        assert results[0].box[2] == pytest.approx(50, abs=1)


class TestLabelHelper:
    def test_labels_copy(self, sample_scene):
        det = DetectionResult(
            label="person", confidence=0.9, box=(10, 10, 20, 40))
        out = label_detections(sample_scene, [det])
        assert out.shape == sample_scene.shape
        assert np.array_equal(out, sample_scene) is False

    def test_empty_detections_unchanged(self, sample_scene):
        out = label_detections(sample_scene, [])
        assert np.array_equal(out, sample_scene)


class TestModelLoad:
    def test_missing_model_raises(self, tmp_path):
        with pytest.raises(DetectionError):
            YoloDetector(str(tmp_path / "nope.onnx"))

    def test_model_runs_on_blank(self, detector):
        img = np.full((640, 640, 3), 128, dtype=np.uint8)
        results = detector.detect(img)
        assert isinstance(results, list)
        # A blank image should produce no confident detections.
        assert all(r.confidence < 0.99 for r in results)

    def test_class_names(self, detector):
        assert "person" in detector.class_names
        assert len(detector.class_names) == 80
        assert COCO_NAMES[0] == "person"


class TestCategories:
    def test_navigation_categories(self):
        assert DetectionResult(
            label="car", confidence=0.5, box=(0, 0, 1, 1)).category == "vehicle"
        assert DetectionResult(
            label="person", confidence=0.5, box=(0, 0, 1, 1)).category == "person"
        assert DetectionResult(
            label="chair", confidence=0.5, box=(0, 0, 1, 1)).category == "obstacle"
        assert DetectionResult(
            label="laptop", confidence=0.5, box=(0, 0, 1, 1)).category == "object"


class TestFalseLaptopFilter:
    """Door/wall rectangles must not be labelled 'laptop'."""

    def test_tall_narrow_box_rejected(self):
        # A door: much taller than wide.
        assert _looks_like_false_laptop("laptop", (0, 0, 100, 400)) is True

    def test_wide_low_box_kept(self):
        # A real laptop on a desk: wider than tall.
        assert _looks_like_false_laptop("laptop", (0, 0, 400, 180)) is False

    def test_square_laptop_kept(self):
        assert _looks_like_false_laptop("laptop", (0, 0, 300, 280)) is False

    def test_non_laptop_never_filtered(self):
        assert _looks_like_false_laptop("person", (0, 0, 100, 400)) is False
        assert _looks_like_false_laptop("door", (0, 0, 100, 400)) is False
