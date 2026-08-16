"""Optional depth integration in the async pipeline (model-free)."""
import time

import numpy as np

from src.core.config import PipelineConfig
from src.core.pipeline import AsyncVisionPipeline


class _StubDepthCamera:
    """Feeds synthetic frames like the E2E test's stub camera."""

    def __init__(self, resolution=(640, 480)):
        self.resolution = resolution
        self._n = 0
        self.actual_fps = 30.0

    def start(self):
        pass

    def read(self):
        self._n += 1
        frame = np.zeros((self.resolution[1], self.resolution[0], 3),
                         dtype=np.uint8)
        x = (self._n * 5) % (self.resolution[0] - 120)
        frame[100:360, x:x + 80] = (80, 160, 240)
        return frame

    def stop(self):
        pass


def test_pipeline_runs_with_synthetic_depth():
    cfg = PipelineConfig()
    cfg.detect_every = 1
    cfg.ocr_every = 5
    cfg.jpeg_width = 320
    cfg.depth_enabled = True
    cfg.depth_backend = "synthetic"
    cfg.depth_model_path = ""

    pipeline = AsyncVisionPipeline(config=cfg, camera=_StubDepthCamera())
    pipeline.start(timeout=5.0)
    try:
        deadline = time.time() + 5.0
        got_state = False
        while time.time() < deadline:
            snap = pipeline.state_snapshot()
            if snap.get("detections") is not None:
                got_state = True
                break
            time.sleep(0.05)
        assert got_state
        # Depth stage ran without crashing; results published.
        results = pipeline.latest_results.snapshot()
        assert "depth_ms" in results["latencies"]
    finally:
        pipeline.stop()


def test_pipeline_disabled_depth_skips_stage():
    cfg = PipelineConfig()
    cfg.detect_every = 1
    cfg.ocr_every = 5
    cfg.jpeg_width = 320
    cfg.depth_enabled = False

    pipeline = AsyncVisionPipeline(config=cfg, camera=_StubDepthCamera())
    pipeline.start(timeout=5.0)
    try:
        deadline = time.time() + 6.0
        seen_depth_ms = None
        while time.time() < deadline:
            results = pipeline.latest_results.snapshot()
            if "depth_ms" in results["latencies"]:
                seen_depth_ms = results["latencies"]["depth_ms"]
                break
            time.sleep(0.1)
        assert seen_depth_ms is not None
        assert seen_depth_ms == 0.0
    finally:
        pipeline.stop()