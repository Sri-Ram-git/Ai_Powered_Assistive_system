"""Integration + failure-mode tests (P17).

Covers behaviours that unit tests skip: pipeline startup with a failing
camera, OCR worker failure tolerance, detector failure handling, and
cross-stage integration with the stub camera.
"""
import time

import numpy as np
import pytest

from src.core.config import PipelineConfig
from src.core.pipeline import AsyncVisionPipeline


class FailingCamera:
    """Camera whose read() always raises (simulates hardware loss)."""

    def __init__(self, resolution=(640, 480)):
        self.resolution = resolution

    def start(self):
        pass

    def stop(self):
        pass

    def read(self):
        raise RuntimeError("device disconnected")


class ThrowingCamera:
    """Camera that fails to start (simulates bad device)."""

    def __init__(self, resolution=(640, 480)):
        self.resolution = resolution

    def start(self):
        raise RuntimeError("cannot open device")

    def stop(self):
        pass

    def read(self):
        raise RuntimeError("never called")


class FlatCamera:
    """Static uniform frame; no detections, no OCR text."""

    def __init__(self, resolution=(640, 480)):
        self.resolution = resolution

    def start(self):
        pass

    def stop(self):
        pass

    def read(self):
        return np.zeros((self.resolution[1], self.resolution[0], 3),
                        dtype=np.uint8)


def test_pipeline_starts_with_stub_camera():
    cfg = PipelineConfig()
    cfg.detect_every = 1
    cfg.ocr_every = 1000
    p = AsyncVisionPipeline(config=cfg, camera=FlatCamera())
    p.start(timeout=5.0)
    try:
        assert p.state_snapshot()["running"] is True
        deadline = time.time() + 10.0
        got = False
        while time.time() < deadline:
            if p.latest_jpeg is not None:
                got = True
                break
            time.sleep(0.05)
        assert got, "pipeline never produced a JPEG"
    finally:
        p.stop()


def test_pipeline_survives_failing_read():
    cfg = PipelineConfig()
    cfg.detect_every = 1
    p = AsyncVisionPipeline(config=cfg, camera=FailingCamera())
    p.start(timeout=5.0)
    try:
        snap = p.state_snapshot()
        # Running is true (start succeeded); a read failure sets error
        # state without crashing the process.
        time.sleep(1.0)
        snap = p.state_snapshot()
        assert snap.get("running") is True or "error" in snap
    finally:
        p.stop()


def test_pipeline_camera_start_failure_sets_error():
    cfg = PipelineConfig()
    # Use the factory path so the pipeline opens (and fails to start)
    # the camera itself.
    p = AsyncVisionPipeline(
        config=cfg, camera_factory=lambda camera_id=0, resolution=(640, 480):
        ThrowingCamera(resolution=resolution))
    p.start(timeout=5.0)
    snap = p.state_snapshot()
    assert snap["running"] is False
    assert "error" in snap and snap["error"]


def test_ocr_worker_tolerates_engine_failure(monkeypatch):
    from src.ocr.worker import OcrWorker

    class BoomEngine:
        def read_text(self, frame):
            raise RuntimeError("ocr engine crashed")

    worker = OcrWorker(BoomEngine())
    worker.start()
    try:
        worker.submit(np.zeros((100, 100, 3), dtype=np.uint8))
        worker.join(timeout=5.0)
        assert worker.latest_result() == []
        assert worker.runs >= 1
    finally:
        worker.stop()


def test_detector_failure_keeps_loop_alive(monkeypatch):
    from src.detection.detector import YoloDetector
    from src.server.pipeline import _camera_factory

    cfg = PipelineConfig()
    cfg.detect_every = 1
    cfg.ocr_every = 1000

    p = AsyncVisionPipeline(config=cfg, camera=FlatCamera())

    def boom(frame):
        raise RuntimeError("inference crashed")

    monkeypatch.setattr(YoloDetector, "detect", boom)
    p.start(timeout=5.0)
    try:
        # The loop must keep running and the process must not crash.
        time.sleep(1.0)
        assert p.state_snapshot()["running"] is True
    finally:
        p.stop()


def test_mode_roundtrip():
    from src.server.pipeline import PipelineServer

    p = PipelineServer()
    p.set_mode("reading")
    assert p.config.mode == "reading"
    with pytest.raises(ValueError):
        p.set_mode("nope")