"""Unit tests for the OCR module (hardware-free)."""
import numpy as np
import pytest

from src.ocr.ocr_engine import (
    OcrEngine,
    _axis_aligned_box,
    draw_text_boxes,
)
from src.utils.exceptions import OcrError


@pytest.fixture(scope="module")
def engine():
    return OcrEngine(min_confidence=0.3)


class TestAxisAlignedBox:
    def test_converts_quad_to_rect(self):
        quad = [[10, 20], [110, 22], [108, 62], [12, 60]]
        assert _axis_aligned_box(quad) == (10, 20, 100, 42)

    def test_invalid_empty(self):
        with pytest.raises(Exception):
            _axis_aligned_box([])


class TestOcrEngine:
    def test_engine_initialises(self, engine):
        assert engine is not None

    def test_empty_image_returns_empty(self, engine):
        assert engine.read_text(np.zeros((0, 0, 3), dtype=np.uint8)) == []
        assert engine.read_text(None) == []

    def test_blank_image_no_text(self, engine):
        blank = np.full((100, 300, 3), 255, dtype=np.uint8)
        assert engine.read_text(blank) == []

    def test_reads_rendered_text(self, engine):
        import cv2

        img = np.full((120, 640, 3), 255, dtype=np.uint8)
        cv2.putText(img, "STOP", (30, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, (0, 0, 0), 4)
        results = engine.read_text(img)
        texts = " ".join(r.text for r in results).upper()
        assert "STOP" in texts

    def test_text_of_joins(self, engine):
        assert engine.text_of(np.full((100, 300, 3), 255, dtype=np.uint8)) == ""

    def test_draw_boxes_preserves_shape(self, engine, sample_scene):
        out = draw_text_boxes(sample_scene, [])
        assert out.shape == sample_scene.shape


class TestErrors:
    def test_bad_input_type_raises(self, engine):
        with pytest.raises(OcrError):
            engine.read_text("not-an-image")  # type: ignore[arg-type]
