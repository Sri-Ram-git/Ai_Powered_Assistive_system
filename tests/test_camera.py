"""Unit tests for the camera module (hardware-free where possible).

Camera acquisition itself requires hardware, so acquisition tests use a
fake camera; the pure-logic tests (validation, HUD, scaling, recorder)
run on any machine.
"""
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.camera import (
    Camera,
    CameraManager,
    HUD,
    VideoRecorder,
    auto_select_resolution,
    get_screen_size,
    scale_to_fit,
)
from src.utils.exceptions import (
    CameraAccessError,
    InvalidResolutionError,
    RecordingError,
)



class FakeCamera:
    """Hardware-free stand-in for src.camera.Camera."""

    def __init__(self, resolution=(320, 240)):
        self.resolution = resolution
        self.frame_count = 0
        self.camera_id = 0

    def read(self):
        self.frame_count += 1
        return np.random.randint(0, 255, (*self.resolution[::-1], 3),
                                 dtype=np.uint8)


class TestCameraValidation:
    def test_read_before_start_raises(self):
        cam = Camera(camera_id=999)
        with pytest.raises(CameraAccessError):
            cam.read()

    def test_set_resolution_before_start_raises(self):
        cam = Camera(camera_id=999)
        with pytest.raises(CameraAccessError):
            cam.set_resolution(640, 480)

    def test_invalid_resolution_rejected(self):
        cam = Camera(camera_id=999)
        with pytest.raises(InvalidResolutionError):
            cam.set_resolution(0, 0)
        with pytest.raises(InvalidResolutionError):
            cam.set_resolution(-5, 100)

    def test_mirror_property_and_toggle(self):
        cam = Camera(camera_id=999, mirror=True)
        assert cam.mirror is True
        cam.set_mirror(False)
        assert cam.mirror is False

    def test_mirror_defaults_off(self):
        # Front-camera frames are captured RAW so OCR/YOLO see true text
        # orientation; mirroring is a display concern (preview_mirror).
        cam = Camera(camera_id=999)
        assert cam.mirror is False

    def test_rotate_property(self):
        assert Camera(camera_id=999, rotate=90).rotate == 90
        assert Camera(camera_id=999, rotate=450).rotate == 90
        assert Camera(camera_id=999).rotate == 0


class TestCameraGeometry:
    """Raw-vs-mirrored / rotated frames (hardware-free via stub capture).

    This is the core front-camera regression: OCR must receive text in
    its true reading orientation ("HELLO WORLD", never "DLROW OLLEH").
    """

    @staticmethod
    def _stub_capture(frame):
        class _StubCap:
            def isOpened(self):
                return True

            def set(self, *a):
                return True

            def get(self, prop):
                h, w = frame.shape[:2]
                if prop == cv2.CAP_PROP_FRAME_WIDTH:
                    return w
                if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                    return h
                return 30.0

            def read(self):
                return True, frame.copy()

            def release(self):
                pass

        return _StubCap()

    def _make_camera(self, frame, monkeypatch, mirror=False, rotate=0):
        import cv2 as _cv2
        monkeypatch.setattr(_cv2, "VideoCapture",
                            lambda *a, **k: self._stub_capture(frame))
        return Camera(camera_id=0, mirror=mirror, rotate=rotate)

    def test_raw_frame_is_not_mirrored(self, monkeypatch):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        frame[:, :50] = (0, 0, 255)  # red region on the left
        cam = self._make_camera(frame, mirror=False, monkeypatch=monkeypatch)
        cam.start()
        try:
            out = cam.read()
            assert np.array_equal(out, frame)  # no flip
        finally:
            cam.stop()

    def test_mirrored_frame_is_flipped_once(self, monkeypatch):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        frame[:, :50] = (0, 0, 255)
        cam = self._make_camera(frame, mirror=True, monkeypatch=monkeypatch)
        cam.start()
        try:
            out = cam.read()
            # The red region moves to the right edge after a horizontal flip.
            assert int((out[:, -50:] == (0, 0, 255)).all(axis=-1).sum()) > 0
            assert int((out[:, :50] == (0, 0, 255)).all(axis=-1).sum()) == 0
        finally:
            cam.stop()

    def test_rotate_90_swaps_and_rotates(self, monkeypatch):
        frame = np.zeros((50, 100, 3), dtype=np.uint8)
        frame[10, 10] = (0, 255, 0)  # single green pixel
        cam = self._make_camera(frame, rotate=90, monkeypatch=monkeypatch)
        cam.start()
        try:
            out = cam.read()
            assert out.shape == (100, 50, 3)
            # After 90° clockwise, pixel (10,10) moves to (10, 50-10-1).
            assert int((out == (0, 255, 0)).all(axis=-1).sum()) == 1
            assert (out[10, 39] == (0, 255, 0)).all()
        finally:
            cam.stop()

    def test_rotate_180_is_upside_down(self, monkeypatch):
        frame = np.zeros((50, 100, 3), dtype=np.uint8)
        frame[5, 5] = (0, 255, 0)
        cam = self._make_camera(frame, rotate=180, monkeypatch=monkeypatch)
        cam.start()
        try:
            out = cam.read()
            assert out.shape == frame.shape
            assert np.array_equal(out[50 - 5 - 1, 100 - 5 - 1],
                                  (0, 255, 0))
        finally:
            cam.stop()


class TestCameraManager:
    def test_list_cameras_runs(self):
        manager = CameraManager()
        # On machines without a webcam this returns []; it must not crash.
        manager.list_cameras(max_cameras=3)

    def test_select_camera_validates(self):
        manager = CameraManager()
        manager._cameras = [type("C", (), {"id": 0})(), type("C", (), {"id": 1})()]
        assert manager.select_camera(1) == 1
        assert manager.select_camera() == 0
        with pytest.raises(ValueError):
            manager.select_camera(9)


class TestDisplayHelpers:
    def test_scale_to_fit_letterbox(self):
        frame = np.full((100, 200, 3), 128, dtype=np.uint8)
        out = scale_to_fit(frame, 1920, 1080)
        assert out.shape == (1080, 1920, 3)
        # Content (non-black) must be present in the centre column
        column = out[:, 960]
        assert column.sum() > 0

    def test_get_screen_size_positive(self):
        w, h = get_screen_size()
        assert w > 0 and h > 0

    def test_auto_select_resolution(self):
        # Only supports 320x240; every requested resolution falls back.
        cam = FakeCamera(resolution=(320, 240))
        cam.set_resolution = lambda w, h: None  # camera ignores requests
        result = auto_select_resolution(cam)
        assert result == (320, 240)


class TestHUD:
    def test_render_shape(self):
        hud = HUD()
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        for _ in range(10):
            hud.tick(30.0)
        out = hud.render(frame, camera=FakeCamera(resolution=(640, 480)),
                         mode="LIVE", status="")
        assert out.shape == frame.shape

    def test_hit_test_and_drag(self):
        hud = HUD()
        cw, ch = 1920, 1080
        tx, ty, tw, th = hud.widget_rect("top", cw, ch)
        assert hud.hit_test(tx + 5, ty + 5, cw, ch) == "top"
        hud.set_widget_pos("top", 800, 600, cw, ch)
        assert hud.widget_rect("top", cw, ch)[:2] == (800, 600)

    def test_toast_and_recording_render(self):
        hud = HUD()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hud.set_recording(True)
        hud.show_toast("test toast")
        out = hud.render(frame)
        assert out.shape == frame.shape


class TestVideoRecorder:
    def test_records_video(self, tmp_path):
        rec = VideoRecorder(FakeCamera(), output_dir=str(tmp_path),
                            duration=1, fps=10)
        path = rec.start()
        assert rec.is_recording is True
        # Let the thread capture a few frames
        seen = 0
        for _ in range(300):
            if rec.latest_frame is not None:
                seen += 1
            if not rec.is_recording:
                break
            time.sleep(0.005)
        rec.stop()
        assert rec.saved_path == path
        assert (tmp_path / Path(path).name).stat().st_size > 0
        assert seen > 0

    def test_double_start_raises(self, tmp_path):
        rec = VideoRecorder(FakeCamera(), output_dir=str(tmp_path),
                            duration=5, fps=10)
        rec.start()
        with pytest.raises(RecordingError):
            rec.start()
        rec.stop()
