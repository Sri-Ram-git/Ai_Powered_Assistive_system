"""Tests for the API layer and the UI/API/server split (P13)."""

import numpy as np
import pytest

from src.core.config import PipelineConfig
from src.server.app import create_app


class StubCamera:
    """Deterministic camera that always returns the same frame."""

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
    cfg = PipelineConfig()
    cfg.detect_every = 1
    cfg.ocr_every = 1000
    cfg.metrics = False
    from src.server.pipeline import PipelineServer

    p = PipelineServer(cfg)
    p._camera = StubCamera()
    p._camera.start()
    p.start()
    yield p
    p.stop()


@pytest.fixture
def client(pipeline):
    app = create_app(pipeline)
    app.config["TESTING"] = True
    return app.test_client()


def test_health_endpoint(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.get_json()
    assert data["status"] == "ok"
    assert "fps" in data
    assert "mode" in data


def test_state_endpoint(client):
    r = client.get("/api/state")
    assert r.status_code == 200
    data = r.get_json()
    assert "detections" in data
    assert "running" in data


def test_config_excludes_secrets(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    data = r.get_json()
    assert "model_path" not in data or "yolov8" not in str(data)
    assert "depth_model_path" not in data


def test_mode_switch_ok(client):
    r = client.post("/api/mode", json={"mode": "reading"})
    assert r.status_code == 200
    assert r.get_json()["ok"] is True


def test_mode_switch_unknown(client):
    r = client.post("/api/mode", json={"mode": "bogus"})
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_command_empty(client):
    r = client.post("/api/command", json={"text": ""})
    assert r.status_code == 400


def test_command_recognised(client):
    r = client.post("/api/command", json={"text": "what do you see"})
    assert r.status_code == 200
    assert r.get_json()["ok"] in (True, False)


def test_dashboard_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Assistive Vision" in r.data


def test_static_css(client):
    r = client.get("/static/dashboard.css")
    assert r.status_code == 200
    assert b"Assistive Vision" in r.data


def test_video_feed_streams(client):
    r = client.get("/video_feed")
    assert r.status_code == 200
    assert "multipart/x-mixed-replace" in r.content_type


def test_set_mode_rejects_invalid():
    from src.server.pipeline import PipelineServer

    p = PipelineServer()
    with pytest.raises(ValueError):
        p.set_mode("bogus")


def test_handle_command_none():
    from src.server.pipeline import PipelineServer

    p = PipelineServer()
    assert p.handle_command(None) is False