"""Tests for the metrics registry and the /api/metrics endpoint (P15)."""
import numpy as np
import pytest

from src.metrics import MetricsRegistry


class StubCamera:
    def __init__(self, resolution=(640, 480)):
        self.resolution = resolution
        self._running = False

    def start(self):
        self._running = True

    def stop(self):
        self._running = False

    def read(self):
        frame = np.full((self.resolution[1], self.resolution[0], 3),
                        128, dtype=np.uint8)
        return frame

    def is_open(self):
        return self._running


@pytest.fixture
def pipeline():
    from src.core.config import PipelineConfig
    from src.server.pipeline import PipelineServer

    cfg = PipelineConfig()
    cfg.detect_every = 1
    cfg.ocr_every = 1000
    cfg.metrics = True
    p = PipelineServer(cfg)
    p._camera = StubCamera()
    p._camera.start()
    p.start()
    yield p
    p.stop()


@pytest.fixture
def client(pipeline):
    from src.server.app import create_app

    app = create_app(pipeline)
    app.config["TESTING"] = True
    return app.test_client()


class TestMetricsRegistry:
    def test_counter_increment(self):
        m = MetricsRegistry()
        m.inc("frames")
        m.inc("frames")
        assert "frames 2" in m.render()

    def test_counter_with_labels(self):
        m = MetricsRegistry()
        m.inc("detections_found", labels={"cls": "person"})
        m.inc("detections_found", labels={"cls": "chair"})
        text = m.render()
        assert 'detections_found{cls="chair"} 1' in text
        assert 'detections_found{cls="person"} 1' in text

    def test_gauge(self):
        m = MetricsRegistry()
        m.set("camera_fps", 24.0)
        assert "camera_fps 24" in m.render()

    def test_histogram_min_max(self):
        m = MetricsRegistry()
        m.observe("yolo_latency_ms", 50.0)
        m.observe("yolo_latency_ms", 30.0)
        text = m.render()
        assert "yolo_latency_ms_count 2" in text
        assert "yolo_latency_ms_min 30" in text
        assert "yolo_latency_ms_max 50" in text

    def test_uptime_present(self):
        m = MetricsRegistry()
        assert "process_uptime_seconds" in m.render()


def test_metrics_endpoint(pipeline, client):
    import time

    deadline = time.time() + 10.0
    body = ""
    while time.time() < deadline:
        r = client.get("/api/metrics")
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        if "frames_processed" in body:
            break
        time.sleep(0.1)
    assert "text/plain" in r.content_type
    assert "frames_processed" in body
    assert "yolo_latency_ms" in body
    assert "process_uptime_seconds" in body