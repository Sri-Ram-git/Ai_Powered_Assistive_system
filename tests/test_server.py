"""Tests for the web server layer (Flask endpoints, pipeline state).

These avoid the real camera/model by using stub components, so they run
on any machine.
"""
import threading
import time

import cv2
import numpy as np
import pytest


class _StubPipeline:
    """Minimal stand-in exposing the PipelineServer interface used by app.py."""

    def __init__(self):
        ok, buf = cv2.imencode(".jpg", np.full((120, 160, 3), 0, np.uint8))
        self._jpeg = buf.tobytes()
        self._state = {
            "running": True,
            "fps": 15.0,
            "resolution": [640, 480],
            "latency_ms": 42.0,
            "detections": [
                {"track_id": 0, "label": "person", "confidence": 0.9,
                 "distance": 3.4, "direction": "ahead"},
            ],
            "guidance": "Person ahead, about 3 metres",
        }

    @property
    def latest_jpeg(self):
        return self._jpeg

    def state_snapshot(self):
        return dict(self._state)

    def start(self, speech_callback=None, timeout=None):
        pass

    def stop(self):
        pass


def _make_app():
    from src.server.app import create_app

    return create_app(_StubPipeline())


class TestWebEndpoints:
    def test_index_serves_dashboard(self):
        app = _make_app()
        client = app.test_client()
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"Assistive Vision" in resp.data
        assert b"video_feed" in resp.data

    def test_api_state_returns_json(self):
        app = _make_app()
        client = app.test_client()
        resp = client.get("/api/state")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["running"] is True
        assert data["detections"][0]["label"] == "person"
        assert data["detections"][0]["distance"] == 3.4
        assert "Person ahead" in data["guidance"]

    def test_video_feed_streams_frames(self):
        app = _make_app()
        client = app.test_client()
        with client.get("/video_feed", buffered=False) as resp:
            assert resp.status_code == 200
            assert "multipart/x-mixed-replace" in resp.mimetype
            # Pull a bounded number of chunks from the endless stream.
            got_frame = False
            for i, chunk in enumerate(resp.response):
                if b"Content-Type: image/jpeg" in chunk:
                    got_frame = True
                    break
                if i >= 5:
                    break
            assert got_frame


class TestPipelineServerState:
    """PipelineServer internals that don't need a real camera."""

    def test_config_defaults(self):
        from src.server.pipeline import PipelineConfig

        cfg = PipelineConfig()
        assert cfg.detect_every == 2
        assert cfg.ocr_every == 10
        assert cfg.iou_threshold == 0.3
        assert cfg.camera_resolution == (1280, 720)

    def test_track_distance_and_direction_helpers(self):
        from src.tracking.tracker import TrackedObject
        from src.server.pipeline import _track_distance_m, _track_direction

        track = TrackedObject(
            track_id=0, label="person",
            box=(100, 100, 50, 150), confidence=0.9,
            center=(125.0, 175.0),
        )
        d = _track_distance_m(track, frame_h=480)
        assert 0.5 <= d <= 20.0  # sane pinhole range for this box

        center = TrackedObject(
            track_id=0, label="person", box=(295, 100, 50, 150),
            confidence=0.9, center=(320.0, 175.0),
        )
        assert _track_direction(center, frame_w=640) == "ahead"
        left = TrackedObject(
            track_id=1, label="person", box=(5, 100, 30, 150),
            confidence=0.9, center=(20.0, 175.0),
        )
        assert _track_direction(left, frame_w=640) == "left"
