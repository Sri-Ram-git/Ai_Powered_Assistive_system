"""Distance-estimation accuracy tests (hardware-free).

Validates the calibration machinery against the *synthetic* dataset in
``data/calibration.csv``: the estimator should recover the pinhole model
almost perfectly on samples generated from it, and produce honest MAE /
RMSE / relative-error numbers that the reports can cite.

These are NOT claims of real-world accuracy — they prove the calibration
*machinery* works.  Real accuracy must be measured against physical
measurements (tools/calibrate_camera.py --live).
"""
import csv
from pathlib import Path

import pytest

from src.navigation.calibration import (
    CalibratedDistanceEstimator,
    DistanceSample,
    calibrate_vfov,
    compute_metrics,
    make_default_dataset,
)

CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "calibration.csv"


def _load_csv(path: Path) -> list:
    samples = []
    with open(path, newline="", encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.lstrip().startswith("#") is False]
        for row in csv.DictReader(lines):
            samples.append(DistanceSample(
                label=str(row["label"]).strip(),
                box_height_px=float(row["box_height_px"]),
                frame_height_px=float(row["frame_height_px"]),
                real_distance_m=float(row["real_distance_m"]),
                reference_height_m=(
                    float(row["reference_height_m"])
                    if row.get("reference_height_m") else None
                ),
            ))
    return samples


class TestComputeMetrics:
    def test_perfect_estimates_zero_error(self):
        m = compute_metrics([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])
        assert m["mae_m"] == 0.0
        assert m["rmse_m"] == 0.0
        assert m["relative_error"] == 0.0

    def test_known_error_values(self):
        m = compute_metrics([(1.0, 1.0), (3.0, 2.0), (4.0, 4.0)])
        assert m["mae_m"] == pytest.approx(0.3333, abs=0.001)
        assert m["rmse_m"] == pytest.approx(0.5774, abs=0.001)

    def test_empty(self):
        m = compute_metrics([])
        assert m["mae_m"] == 0.0


class TestCalibrateVfov:
    def test_recovers_known_fov(self):
        # A 1.7m object at 2m in a 720p frame with 55deg VFOV has a box
        # height that our model derives; solve back and expect ~55.
        import numpy as np


        ref_h, dist, frame_h, fov = 1.7, 2.0, 720.0, 55.0
        # box height that yields the target distance
        focal = (frame_h / 2.0) / np.tan(np.radians(fov / 2.0))
        box_h = ref_h * focal / dist
        solved = calibrate_vfov(box_h, frame_h, ref_h, dist)
        assert solved == pytest.approx(fov, abs=0.5)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            calibrate_vfov(0.0, 720.0, 1.7, 2.0)
        with pytest.raises(ValueError):
            calibrate_vfov(100.0, 720.0, 0.0, 2.0)


class TestCalibratedEstimator:
    def test_synthetic_dataset_recovers_model(self):
        estimator = CalibratedDistanceEstimator()
        samples = _load_csv(CSV_PATH) if CSV_PATH.exists() \
            else make_default_dataset()
        result = estimator.calibrate(samples)
        # Generated from the pinhole model with a fixed FOV, so the fit
        # should be very close to the ground truth.
        assert result.samples == len(samples)
        assert result.mae_m < 0.05
        assert result.rmse_m < 0.1
        assert 45.0 <= result.calibrated_vfov_deg <= 65.0

    def test_estimate_returns_distance_and_error_bound(self):
        estimator = CalibratedDistanceEstimator()
        estimator.calibrate(make_default_dataset())
        d, err = estimator.estimate((100, 100, 50, 150), 480, "person")
        assert d > 0.0
        assert err is not None and err >= 0.0

    def test_uncalibrated_estimate_returns_none_error(self):
        estimator = CalibratedDistanceEstimator()
        d, err = estimator.estimate((100, 100, 50, 150), 480, "person")
        assert d > 0.0
        assert err is None

    def test_calibrate_empty_raises(self):
        with pytest.raises(ValueError):
            CalibratedDistanceEstimator().calibrate([])

    def test_reference_height_override(self):
        estimator = CalibratedDistanceEstimator(
            reference_heights={"person": 2.0})
        d, _ = estimator.estimate((100, 100, 50, 150), 480, "person")
        default = CalibratedDistanceEstimator()
        d0, _ = default.estimate((100, 100, 50, 150), 480, "person")
        assert d > d0  # taller reference -> farther estimate at same box