"""Realtime pipeline tests (Phase 6-7, 21).

Without hardware, prove the async engine is usable as a real-time
desktop engine:
    - raw ``latest_frame`` is exposed when JPEG encoding is disabled
    - OCR is *not* loaded/started when disabled (Phase 21)
    - the detect loop runs and publishes tracks/state
    - ``reset()`` clears temporal state safely
"""
import time

import numpy as np
from contextlib import contextmanager

from src.core.config import PipelineConfig
from src.core.pipeline import AsyncVisionPipeline


class _StubCamera:
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


@contextmanager
def _patch_camera(factory):
    """Monkeypatch the server-side Camera used by the engine."""
    import src.server.pipeline as pl

    original = pl.Camera
    pl.Camera = lambda camera_id=0, resolution=(640, 480), **kw: factory(
        camera_id=camera_id, resolution=resolution)
    try:
        yield
    finally:
        pl.Camera = original


def _boot(cfg):
    from src.server.pipeline import PipelineServer

    server = PipelineServer(cfg)
    server.start(timeout=5.0)
    return server


class TestRealtimePipeline:
    def _cfg(self):
        cfg = PipelineConfig()
        cfg.camera_resolution = (640, 480)
        cfg.encode_jpeg = False
        cfg.ocr_enabled = False
        cfg.navigation_enabled = False
        return cfg

    def test_latest_frame_exposed_without_jpeg(self):
        cfg = self._cfg()
        with _patch_camera(lambda **kw: _StubCamera(resolution=kw["resolution"])):
            server = _boot(cfg)
            try:
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    if server.latest_frame is not None:
                        break
                    time.sleep(0.05)
                frame = server.latest_frame
                assert frame is not None
                assert frame.shape[0] == 480 and frame.shape[1] == 640
                assert server.latest_jpeg is None  # encoding disabled
            finally:
                server.stop()

    def test_ocr_worker_not_started_when_disabled(self):
        cfg = self._cfg()
        with _patch_camera(lambda **kw: _StubCamera(resolution=kw["resolution"])):
            server = _boot(cfg)
            try:
                assert server._ocr_worker is None
            finally:
                server.stop()

    def test_detect_loop_runs_and_publishes_state(self):
        cfg = self._cfg()
        cfg.detect_every = 1
        with _patch_camera(lambda **kw: _StubCamera(resolution=kw["resolution"])):
            server = _boot(cfg)
            try:
                deadline = time.time() + 6.0
                processed = 0
                while time.time() < deadline:
                    if server.metrics is not None:
                        processed = server.metrics.counter("frames_processed")
                    if processed >= 3:
                        break
                    time.sleep(0.1)
                assert processed >= 3
                state = server.state_snapshot()
                assert state.get("running") is True
                assert "detections" in state
            finally:
                server.stop()

    def test_reset_clears_temporal_state(self):
        cfg = self._cfg()
        with _patch_camera(lambda **kw: _StubCamera(resolution=kw["resolution"])):
            server = _boot(cfg)
            try:
                deadline = time.time() + 4.0
                while time.time() < deadline:
                    if server.metrics and \
                            server.metrics.counter("frames_processed") >= 2:
                        break
                    time.sleep(0.1)
                server.reset()  # must not raise
                assert server._tracker.all_tracks() == []
            finally:
                server.stop()

    def test_pipeline_stop_releases_camera(self):
        cfg = self._cfg()
        with _patch_camera(lambda **kw: _StubCamera(resolution=kw["resolution"])):
            server = _boot(cfg)
            server.stop()
            assert server.state_snapshot().get("running") is False