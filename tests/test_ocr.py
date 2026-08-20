"""Unit tests for the OCR module (hardware-free)."""
import numpy as np
import pytest

from src.ocr.ocr_engine import (
    OcrEngine,
    OcrResult,
    _axis_aligned_box,
    _order_and_dedupe,
    draw_text_boxes,
)
from src.utils.exceptions import OcrError


@pytest.fixture(scope="module")
def engine():
    return OcrEngine(min_confidence=0.3)


class TestOrderAndDedupe:
    """Spatial reading-order (fixes jumbled "COLA COCA" output)."""

    def test_left_to_right_within_a_line(self):
        right = OcrResult(text="EXIT", confidence=0.9,
                          box=(200, 10, 60, 20))
        left = OcrResult(text="EMERGENCY", confidence=0.9,
                         box=(10, 10, 180, 20))
        out = _order_and_dedupe([right, left])
        assert [r.text for r in out] == ["EMERGENCY", "EXIT"]

    def test_lines_top_to_bottom(self):
        bottom = OcrResult(text="BOTTOM", confidence=0.9,
                           box=(10, 60, 80, 20))
        top = OcrResult(text="TOP", confidence=0.9, box=(10, 5, 50, 20))
        out = _order_and_dedupe([bottom, top])
        assert [r.text for r in out] == ["TOP", "BOTTOM"]

    def test_exact_duplicates_collapse_keep_highest_conf(self):
        a = OcrResult(text="COLA", confidence=0.7, box=(10, 10, 50, 20))
        b = OcrResult(text="COLA", confidence=0.95, box=(30, 12, 50, 20))
        out = _order_and_dedupe([a, b])
        assert len(out) == 1
        assert out[0].confidence == 0.95

    def test_multi_line_left_and_right(self):
        # "DO" on line 1, "NOT ENTER" on line 2 (line 2: NOT left, ENTER
        # right) must come out top-to-bottom, left-to-right.
        do = OcrResult(text="DO", confidence=0.9, box=(10, 5, 30, 20))
        enter = OcrResult(text="ENTER", confidence=0.9, box=(200, 45, 90, 20))
        not_ = OcrResult(text="NOT", confidence=0.9, box=(10, 45, 40, 20))
        out = _order_and_dedupe([enter, do, not_])
        assert [r.text for r in out] == ["DO", "NOT", "ENTER"]

    def test_single_result_unchanged(self):
        r = OcrResult(text="ONE", confidence=0.9, box=(0, 0, 50, 20))
        assert _order_and_dedupe([r]) == [r]

    def test_empty(self):
        assert _order_and_dedupe([]) == []


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
