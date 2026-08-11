"""Unit tests for the camera module (hardware-free where possible).

Camera acquisition itself requires hardware, so acquisition tests use a
fake camera; the pure-logic tests (validation, HUD, scaling, recorder)
run on any machine.
"""
import time
from pathlib import Path

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
