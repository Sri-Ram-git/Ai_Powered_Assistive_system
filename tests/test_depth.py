"""Tests for the optional depth module (model-free)."""
import numpy as np

from src.depth.depth_estimator import (
    DepthEstimator,
    SyntheticDepthBackend,
    create_depth_estimator,
)
from src.depth.depth_result import DepthResult


class TestDepthResult:
    def test_box_depth_median(self):
        depth = np.zeros((100, 100), dtype=np.float32)
        depth[20:30, 20:30] = 0.8  # box region
        result = DepthResult(map=depth, per_box_depth={})
        d = result.box_depth((20, 20, 10, 10))
        assert d is not None
        assert d > 0.7

    def test_box_depth_out_of_bounds(self):
        result = DepthResult(map=np.zeros((10, 10), dtype=np.float32),
                             per_box_depth={})
        assert result.box_depth((100, 100, 10, 10)) is None
        assert result.box_depth((0, 0, 0, 0)) is None

    def test_empty_map(self):
        result = DepthResult(map=np.zeros((0, 0), dtype=np.float32),
                             per_box_depth={})
        assert result.box_depth((0, 0, 5, 5)) is None


class TestSyntheticBackend:
    def test_estimate_returns_depth_map(self):
        est = SyntheticDepthBackend()
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        result = est.estimate(frame, boxes=[(10, 10, 20, 20)])
        assert isinstance(result, DepthResult)
        assert result.map.shape == (60, 80)
        assert result.backend == "synthetic"
        assert 0 in result.per_box_depth

    def test_depth_grows_with_row(self):
        est = SyntheticDepthBackend()
        result = est.estimate(np.zeros((50, 50, 3), dtype=np.uint8))
        assert result.map[0, 0] < result.map[49, 0]

    def test_gray_frame_accepted(self):
        est = SyntheticDepthBackend()
        result = est.estimate(np.zeros((30, 40), dtype=np.uint8))
        assert result.map.shape == (30, 40)


class TestFactory:
    def test_create_synthetic(self):
        assert isinstance(create_depth_estimator("synthetic"),
                          SyntheticDepthBackend)

    def test_create_unknown_raises(self):
        import pytest

        with pytest.raises(ValueError):
            create_depth_estimator("bogus")

    def test_v2_without_model_raises(self):
        import pytest

        with pytest.raises((ValueError, FileNotFoundError)):
            create_depth_estimator("depth_anything_v2", model_path="nope.onnx")