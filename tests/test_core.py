"""Tests for the async core: FrameManager, LatestResults, pipeline."""
import time

import numpy as np

from src.core.config import PipelineConfig
from src.core.frame_manager import FrameManager
from src.core.results import LatestResults


class TestPipelineConfig:
    def test_from_yaml_parses_reject_box_shape(self, tmp_path):
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text(
            "detection:\n"
            "  reject_box_shape:\n"
            "    remote:\n"
            "      min_aspect: 1.8\n"
            "      max_area_ratio: 0.25\n",
            encoding="utf-8",
        )
        cfg = PipelineConfig.from_yaml(str(yaml_path))
        assert cfg.reject_box_shape["remote"]["min_aspect"] == 1.8
        assert cfg.reject_box_shape["remote"]["max_area_ratio"] == 0.25

    def test_from_yaml_missing_file_defaults_empty(self):
        cfg = PipelineConfig.from_yaml("no_such_file_anywhere.yaml")
        assert cfg.reject_box_shape == {}

    def test_default_camera_is_raw_and_unmirrored(self):
        # Vision frames must be geometrically correct: no capture-side
        # mirroring; preview selfie-mirroring is a display-only flag.
        cfg = PipelineConfig()
        assert cfg.camera_mirror is False
        assert cfg.camera_rotate == 0
        assert cfg.preview_mirror is True

    def test_from_yaml_parses_camera_geometry(self, tmp_path):
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text(
            "camera:\n"
            "  mirror: true\n"
            "  rotate: 270\n"
            "  preview_mirror: false\n",
            encoding="utf-8",
        )
        cfg = PipelineConfig.from_yaml(str(yaml_path))
        assert cfg.camera_mirror is True
        assert cfg.camera_rotate == 270
        assert cfg.preview_mirror is False

    def test_from_yaml_normalises_rotate(self, tmp_path):
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text("camera:\n  rotate: 450\n", encoding="utf-8")
        cfg = PipelineConfig.from_yaml(str(yaml_path))
        assert cfg.camera_rotate == 90


class TestFrameManager:
    def test_publish_latest(self):
        fm = FrameManager()
        a = np.zeros((10, 10, 3), dtype=np.uint8)
        b = np.ones((10, 10, 3), dtype=np.uint8)
        assert fm.publish(a) == 1
        assert fm.publish(b) == 2
        fid, frame = fm.latest()
        assert fid == 2
        assert np.array_equal(frame, b)

    def test_latest_none_before_publish(self):
        fm = FrameManager()
        assert fm.latest() == (0, None)

    def test_counters(self):
        fm = FrameManager()
        fm.publish(np.zeros((2, 2, 3), dtype=np.uint8))
        fm.publish(np.zeros((2, 2, 3), dtype=np.uint8))
        fm.publish(np.zeros((2, 2, 3), dtype=np.uint8))
        fm.latest()  # one consume
        c = fm.counters()
        assert c["frames_published"] == 3
        assert c["frames_consumed"] == 1
        assert c["frames_dropped"] == 2

    def test_fps(self):
        fm = FrameManager()
        for _ in range(20):
            fm.publish(np.zeros((2, 2, 3), dtype=np.uint8))
            time.sleep(0.01)
        assert fm.fps() > 0.0


class TestLatestResults:
    def test_update_and_snapshot(self):
        r = LatestResults()
        r.update(detections=["a"], guidance=["hi"])
        snap = r.snapshot()
        assert snap["detections"] == ["a"]
        assert snap["guidance"] == ["hi"]
        # snapshot is a copy
        snap["detections"].append("b")
        assert r.snapshot()["detections"] == ["a"]

    def test_latencies(self):
        r = LatestResults()
        r.update(latencies={"yolo_ms": 50.0})
        assert r.snapshot()["latencies"]["yolo_ms"] == 50.0