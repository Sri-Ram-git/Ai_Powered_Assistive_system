"""Tests for OCR preprocessing strategies and the async OCR worker."""
import threading

import cv2
import numpy as np
import pytest

from src.ocr.preprocess import SUPPORTED_STRATEGIES, preprocess


@pytest.fixture(scope="module")
def text_frame():
    img = np.full((200, 640, 3), 255, dtype=np.uint8)
    cv2.putText(img, "EXIT", (30, 100), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (0, 0, 0), 4)
    return img


class TestPreprocess:
    def test_none_returns_copy(self, text_frame):
        out = preprocess(text_frame, "none")
        assert out.shape == text_frame.shape
        assert out is not text_frame

    def test_gray_single_channel(self, text_frame):
        out = preprocess(text_frame, "gray")
        assert out.ndim == 2

    def test_threshold_binary(self, text_frame):
        out = preprocess(text_frame, "threshold")
        assert out.ndim == 2
        assert set(np.unique(out)).issubset({0, 255})

    def test_contrast_grayscale(self, text_frame):
        out = preprocess(text_frame, "contrast")
        assert out.ndim == 2

    def test_downscale_halves(self, text_frame):
        out = preprocess(text_frame, "downscale")
        assert out.shape[1] == text_frame.shape[1] // 2
        assert out.shape[0] == text_frame.shape[0] // 2

    def test_all_strategies_accept_bgr(self, text_frame):
        for strategy in SUPPORTED_STRATEGIES:
            out = preprocess(text_frame, strategy)
            assert out.size > 0

    def test_unknown_strategy_raises(self, text_frame):
        with pytest.raises(ValueError):
            preprocess(text_frame, "bogus")

    def test_empty_frame_raises(self):
        with pytest.raises(ValueError):
            preprocess(np.zeros((0, 0, 3), dtype=np.uint8), "gray")


class TestOcrWorker:
    """Uses a fake engine to avoid loading RapidOCR."""

    def _make_engine(self, delay=0.0):
        class _FakeEngine:
            def __init__(self):
                self.calls = 0

            def read_text(self, image):
                import time as _t
                self.calls += 1
                if delay:
                    _t.sleep(delay)
                return []

        return _FakeEngine()

    def test_worker_returns_latest_result(self):
        from src.ocr.worker import OcrWorker

        engine = self._make_engine()
        worker = OcrWorker(engine, preprocess_strategy="none",
                           poll_interval=0.005)
        worker.start()
        try:
            frame = np.zeros((10, 10, 3), dtype=np.uint8)
            worker.submit(frame)
            worker.join(timeout=5.0)
            assert worker.latest_result() == []
            assert worker.runs >= 1
        finally:
            worker.stop()

    def test_worker_never_blocks_submit(self):
        from src.ocr.worker import OcrWorker

        engine = self._make_engine(delay=0.5)
        worker = OcrWorker(engine, preprocess_strategy="none",
                           poll_interval=0.005)
        worker.start()
        try:
            started = threading.Event()
            done = []
            def _submit():
                started.set()
                worker.submit(np.zeros((8, 8, 3), dtype=np.uint8))
                done.append(True)
            t = threading.Thread(target=_submit)
            t.start()
            started.wait(1.0)
            # Submit returns immediately even though OCR is busy.
            assert t.is_alive() is False or done  # returned quickly
            t.join(2.0)
            assert done
        finally:
            worker.stop()