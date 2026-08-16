"""Tests for the async core: FrameManager, LatestResults, pipeline."""
import time

import numpy as np
import pytest

from src.core.frame_manager import FrameManager
from src.core.results import LatestResults


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