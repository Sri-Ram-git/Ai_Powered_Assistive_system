"""Tests for product modes (P21)."""
import numpy as np
import pytest

from src.core.config import PipelineConfig
from src.modes import MODES, ModeManager, validate_mode


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
    from src.server.pipeline import PipelineServer

    cfg = PipelineConfig()
    cfg.detect_every = 1
    cfg.ocr_every = 1000
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


class TestModeManager:
    def test_valid_modes_present(self):
        for name in ("object", "reading", "navigation", "scene", "voice"):
            assert name in MODES

    def test_apply_sets_config_knobs(self):
        cfg = PipelineConfig()
        m = ModeManager(cfg)
        behavior = m.apply("reading")
        assert cfg.mode == "reading"
        assert cfg.ocr_enabled is True
        assert cfg.navigation_enabled is False
        assert behavior.name == "reading"

    def test_apply_navigation_mode(self):
        cfg = PipelineConfig()
        ModeManager(cfg).apply("navigation")
        assert cfg.navigation_enabled is True
        assert cfg.ocr_enabled is False

    def test_unknown_mode_raises(self):
        m = ModeManager(PipelineConfig())
        with pytest.raises(ValueError):
            m.get("bogus")
        with pytest.raises(ValueError):
            m.apply("bogus")

    def test_scene_mode_describes(self):
        behavior = ModeManager(PipelineConfig()).get("scene")
        assert behavior.scene_describe is True
        assert behavior.quiet is True

    def test_voice_mode_quiet(self):
        behavior = ModeManager(PipelineConfig()).get("voice")
        assert behavior.quiet is True
        assert behavior.announce_objects is False


class TestValidateMode:
    def test_valid(self):
        assert validate_mode("OBJECT") == "object"

    def test_invalid(self):
        with pytest.raises(ValueError):
            validate_mode("wat")


def test_pipeline_set_mode_via_api(pipeline, client):
    r = client.post("/api/mode", json={"mode": "scene"})
    assert r.status_code == 200
    assert pipeline.config.mode == "scene"
    assert pipeline.config.navigation_enabled is False
    assert pipeline.mode_behavior.scene_describe is True