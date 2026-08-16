"""End-to-end smoke test of the PipelineServer loop with the real ONNX
model and a stub camera (no webcam required).

The pipeline is started, allowed to process a handful of synthetic
frames, then stopped.  We assert that a JPEG frame is produced and that
the state dict is populated (running flag, detections key present).
"""
import time

import numpy as np

from src.server.pipeline import PipelineConfig, PipelineServer


class _StubCamera:
    """Feeds synthetic frames without touching OpenCV VideoCapture."""

    def __init__(self, resolution=(640, 480)):
        self.resolution = resolution
        self._n = 0
        self.actual_fps = 30.0

    def start(self):
        pass

    def read(self):
        self._n += 1
        # A large-ish moving square that the model may or may not detect;
        # the point is the pipeline loop runs and yields frames/state.
        frame = np.zeros((self.resolution[1], self.resolution[0], 3),
                         dtype=np.uint8)
        x = (self._n * 5) % (self.resolution[0] - 120)
        frame[100:360, x:x + 80] = (80, 160, 240)
        return frame

    def stop(self):
        pass


def _run_pipeline_with(factory, seconds=12.0):
    """Boot a pipeline using the given stub camera factory."""
    import src.server.pipeline as pl

    original = pl.Camera

    def _fake_camera(camera_id=0, resolution=(640, 480), **kwargs):
        return factory(camera_id=camera_id, resolution=resolution)

    pl.Camera = _fake_camera
    try:
        cfg = PipelineConfig()
        cfg.detect_every = 1
        cfg.ocr_every = 5
        cfg.jpeg_width = 320
        server = PipelineServer(cfg)
        server.start(timeout=10.0)
        try:
            deadline = time.time() + seconds
            while time.time() < deadline:
                jpeg = server.latest_jpeg
                state = server.state_snapshot()
                if jpeg and state.get("detections") is not None and state.get("fps") is not None:
                    break
                time.sleep(0.1)
            return server, jpeg, state
        finally:
            server.stop()
    finally:
        pl.Camera = original


class TestPipelineEndToEnd:
    def test_pipeline_yields_frames_and_state(self):
        server, jpeg, state = _run_pipeline_with(
            lambda **kw: _StubCamera(resolution=kw["resolution"]),
        )
        assert server is not None
        assert jpeg is not None
        assert len(jpeg) > 1000
        assert b"\xff\xd8" == jpeg[:2]  # JPEG SOI marker
        assert state["running"] is True
        assert "detections" in state
        assert "fps" in state
        assert state["fps"] is not None

    def test_pipeline_state_snapshot_is_copy(self):
        server, _, _ = _run_pipeline_with(
            lambda **kw: _StubCamera(resolution=kw["resolution"]),
        )
        # Wait until the inference thread has published detections.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            snap = server.state_snapshot()
            if snap.get("detections") is not None:
                break
            time.sleep(0.05)

        tampered = server.state_snapshot()
        assert tampered.get("detections") is not None
        tampered["detections"].append("tampered")
        fresh = server.state_snapshot()
        assert "tampered" not in fresh.get("detections", [])
