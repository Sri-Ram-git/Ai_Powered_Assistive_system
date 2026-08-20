"""Integration tests: object-aware OCR inside the pipeline.

Drives the detect-loop scheduling helpers directly (no camera, no YOLO
model, no RapidOCR needed): the object OCR worker, per-track store,
read-aloud, manual read and reset behaviour are exercised through the
public pipeline surface.
"""
import time
from types import SimpleNamespace

import cv2
import numpy as np

from src.core.config import PipelineConfig
from src.core.pipeline import AsyncVisionPipeline
from src.ocr.object_ocr import OcrTrigger
from src.ocr.object_worker import ObjectOcrWorker
from src.ocr.ocr_engine import OcrResult
from src.ocr.policy import OcrPolicy


class _FakeEngine:
    def __init__(self, lines=("HELLO WORLD",)):
        self.lines = list(lines)
        self.calls = 0

    def read_text(self, image):
        self.calls += 1
        return [OcrResult(text=self.lines[0], confidence=0.95,
                          box=(5, 5, 40, 20))]


def _text_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (200, 240), (0, 180, 0), -1)
    return frame


def _track(tid=1, label="bottle", box=(100, 100, 100, 140), conf=0.9):
    return SimpleNamespace(
        track_id=tid, label=label, box=box, confidence=conf,
        area=box[2] * box[3],
    )


class TestPipelineObjectOcr:
    def _pipeline(self, **cfg_kwargs):
        cfg = PipelineConfig()
        cfg.ocr_enabled = True
        cfg.object_ocr_enabled = True
        cfg.encode_jpeg = False
        cfg.navigation_enabled = False
        for key, value in cfg_kwargs.items():
            setattr(cfg, key, value)
        p = AsyncVisionPipeline(config=cfg)
        p._ocr_policy = OcrPolicy.from_yaml(None)
        p._ocr_trigger = OcrTrigger(
            cooldown_s=cfg.ocr_cooldown_s,
            stale_after_s=cfg.ocr_stale_after_s,
            move_px=cfg.ocr_move_px,
        )
        p._ocr_worker = ObjectOcrWorker(
            _FakeEngine(), text_presence=False,
            on_result=p._on_object_ocr,
        )
        p._ocr_worker.start()
        return p

    def test_schedule_ocr_reads_track_and_updates_state(self):
        p = self._pipeline()
        try:
            frame = _text_frame()
            p._schedule_object_ocr(frame, [_track()], 1)
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if p.latest_track_ocr() is not None:
                    break
                time.sleep(0.02)
            info = p.latest_track_ocr()
            assert info is not None
            assert info["text"] == "HELLO WORLD"
            assert info["label"] == "bottle"
            assert p.state_snapshot()["ocr_text"] == "HELLO WORLD"
            assert len(p.track_ocr_history()) == 1
        finally:
            p._ocr_worker.stop()

    def test_read_aloud_speaks_saved_text_without_re_ocr(self):
        phrases = []
        p = self._pipeline()
        p._speech_callback = phrases.append
        try:
            p._schedule_object_ocr(_text_frame(), [_track()], 1)
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if p.latest_track_ocr() is not None:
                    break
                time.sleep(0.02)
            engine_calls_before = p._ocr_worker._engine.calls
            assert p.read_latest_text() is True
            assert phrases == ["Text says: HELLO WORLD"]
            # READ ALOUD never re-runs OCR.
            assert p._ocr_worker._engine.calls == engine_calls_before
        finally:
            p._ocr_worker.stop()

    def test_read_aloud_returns_false_when_nothing_read(self):
        p = self._pipeline()
        try:
            assert p.read_latest_text() is False
        finally:
            p._ocr_worker.stop()

    def test_auto_read_speaks_and_dedupes(self):
        phrases = []
        p = self._pipeline(ocr_auto_read=True)
        p._speech_callback = phrases.append
        try:
            p._schedule_object_ocr(_text_frame(), [_track()], 1)
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if phrases:
                    break
                time.sleep(0.02)
            assert phrases and "HELLO WORLD" in phrases[0]
        finally:
            p._ocr_worker.stop()

    def test_manual_read_uses_eligible_track(self):
        phrases = []
        p = self._pipeline()
        p._speech_callback = phrases.append
        p._tracker = SimpleNamespace(active_tracks=[_track()])
        p._frames.publish(_text_frame())  # a running pipeline has frames
        try:
            assert p.request_manual_ocr() is True
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if p.latest_track_ocr() is not None:
                    break
                time.sleep(0.02)
            assert p.read_latest_text() is True
            assert phrases
        finally:
            p._ocr_worker.stop()

    def test_reset_clears_track_ocr(self):
        p = self._pipeline()
        try:
            p._schedule_object_ocr(_text_frame(), [_track()], 1)
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if p.latest_track_ocr() is not None:
                    break
                time.sleep(0.02)
            assert p.latest_track_ocr() is not None
            p.reset()
            assert p.latest_track_ocr() is None
            assert p.track_ocr_history() == []
        finally:
            p._ocr_worker.stop()

    def test_no_eligible_track_skips_ocr(self):
        p = self._pipeline()
        try:
            p._schedule_object_ocr(_text_frame(), [_track(label="person")], 1)
            time.sleep(0.1)
            assert p._ocr_worker._engine.calls == 0
        finally:
            p._ocr_worker.stop()

    def test_track_ocr_items_projected_for_downstream(self):
        p = self._pipeline()
        try:
            p._schedule_object_ocr(_text_frame(), [_track()], 1)
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if p.latest_track_ocr() is not None:
                    break
                time.sleep(0.02)
            items = p._track_ocr_items()
            assert len(items) == 1
            assert items[0].text == "HELLO WORLD"
            assert items[0].box[2] > 0  # width from ROI
        finally:
            p._ocr_worker.stop()

    def test_ocr_stats_exposed(self):
        p = self._pipeline()
        try:
            stats = p.ocr_stats()
            assert "runs" in stats
        finally:
            p._ocr_worker.stop()

    def test_disabled_worker_is_none_when_ocr_off(self):
        cfg = PipelineConfig()
        cfg.ocr_enabled = False
        p = AsyncVisionPipeline(config=cfg)
        assert p._ocr_worker is None


class TestCameraGeometryWiring:
    """The vision pipeline must request RAW frames from the camera factory
    (mirror=False) so OCR receives true text orientation, and pass the
    configured sensor rotation through exactly once."""

    def test_camera_factory_receives_unmirrored_config(self):
        cfg = PipelineConfig()
        cfg.ocr_enabled = False
        cfg.navigation_enabled = False
        cfg.encode_jpeg = False
        seen = {}

        class _StubCam:
            def __init__(self, **kwargs):
                seen.update(kwargs)
                self.resolution = (640, 480)

            def start(self):
                pass

            def read(self):
                return np.zeros((480, 640, 3), dtype=np.uint8)

            def stop(self):
                pass

        p = AsyncVisionPipeline(
            config=cfg,
            camera_factory=lambda **kw: _StubCam(**kw),
        )
        try:
            p.start(timeout=3.0)
            assert seen["mirror"] is False
            assert seen["rotate"] == 0
        finally:
            p.stop()

    def test_camera_factory_receives_configured_rotation(self):
        cfg = PipelineConfig()
        cfg.camera_rotate = 270
        cfg.ocr_enabled = False
        cfg.navigation_enabled = False
        cfg.encode_jpeg = False
        seen = {}

        class _StubCam:
            def __init__(self, **kwargs):
                seen.update(kwargs)
                self.resolution = (480, 640)

            def start(self):
                pass

            def read(self):
                return np.zeros((640, 480, 3), dtype=np.uint8)

            def stop(self):
                pass

        p = AsyncVisionPipeline(
            config=cfg,
            camera_factory=lambda **kw: _StubCam(**kw),
        )
        try:
            p.start(timeout=3.0)
            assert seen["rotate"] == 270
        finally:
            p.stop()