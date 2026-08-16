"""Distance estimation — calibration, validation, and evaluation.

The base pinhole estimator lives in ``src.navigation.guidance``
(``distance_estimate``).  This module adds a *validated* wrapper:

* a calibration dataset (ground-truth distances measured at known
  ranges);
* an error model (MAE / RMSE / relative error);
* optional FOV self-calibration from a known-size object;
* an honest ``CalibratedDistanceEstimator`` that reports confidence so
  callers never present a heuristic as navigation-grade precision.

The calibration data is stored as a small CSV (see ``tools/``) and the
evaluation script writes results to ``evaluation/results/``.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.utils.logger import setup_logger

_logger = setup_logger("Calibration")

DEFAULT_VFOV_DEG = 55.0


@dataclass
class DistanceSample:
    """One measured (box, ground-truth distance) pair."""

    label: str
    box_height_px: float
    frame_height_px: float
    real_distance_m: float
    reference_height_m: Optional[float] = None


@dataclass
class CalibrationResult:
    """Error metrics for the calibration dataset."""

    samples: int = 0
    mae_m: float = 0.0
    rmse_m: float = 0.0
    relative_error: float = 0.0  # mean |error| / real distance
    per_label: Dict[str, Dict[str, float]] = field(default_factory=dict)
    calibrated_vfov_deg: Optional[float] = None


def compute_metrics(
    estimates: List[Tuple[float, float]],
) -> Dict[str, float]:
    """Compute MAE, RMSE, and mean relative error from (estimated, real)."""
    if not estimates:
        return {"mae_m": 0.0, "rmse_m": 0.0, "relative_error": 0.0}
    est = np.asarray([e for e, _ in estimates], dtype=float)
    real = np.asarray([r for _, r in estimates], dtype=float)
    err = np.abs(est - real)
    rel = err / np.maximum(real, 1e-6)
    return {
        "mae_m": float(np.mean(err)),
        "rmse_m": float(np.sqrt(np.mean(err ** 2))),
        "relative_error": float(np.mean(rel)),
    }


def calibrate_vfov(
    box_height_px: float,
    frame_height_px: float,
    reference_height_m: float,
    real_distance_m: float,
) -> float:
    """Solve for the vertical FOV (degrees) that makes the pinhole
    estimate match a single known (size, distance) observation.

    Pinhole: distance = (real_height * focal) / box_height, with
    focal = (frame_height / 2) / tan(vfov/2).  Rearranged for vfov.

    Args:
        box_height_px: Measured box height in pixels.
        frame_height_px: Frame height in pixels.
        reference_height_m: Known real-world height of the object.
        real_distance_m: Known true distance to the object.

    Returns:
        FOV in degrees consistent with the observation.
    """
    if box_height_px <= 0 or reference_height_m <= 0 or real_distance_m <= 0:
        raise ValueError("box height, reference height, and distance must be > 0")
    # distance = ref * (frame_h / 2) / (tan(vfov/2) * box_h)
    # => tan(vfov/2) = ref * (frame_h / 2) / (distance * box_h)
    ratio = (reference_height_m * frame_height_px / 2.0) / (
        real_distance_m * box_height_px
    )
    vfov = 2.0 * float(np.degrees(np.arctan(ratio)))
    return min(120.0, max(1.0, vfov))


class CalibratedDistanceEstimator:
    """Distance estimator validated against a calibration dataset.

    If ``vfov_deg`` is None, the estimator derives a VFOV from the
    calibration samples (single-parameter fit minimising RMSE over the
    samples' reference heights).

    ``estimate`` returns the pinhole distance plus a documented error
    bound from the calibration, so consumers can state confidence
    honestly rather than presenting it as ground truth.
    """

    def __init__(
        self,
        vfov_deg: Optional[float] = None,
        reference_heights: Optional[Dict[str, float]] = None,
    ) -> None:
        from src.navigation.guidance import _REFERENCE_HEIGHTS

        self._heights = dict(_REFERENCE_HEIGHTS)
        if reference_heights:
            self._heights.update({str(k): float(v)
                                  for k, v in reference_heights.items()})
        self._vfov = vfov_deg
        self._calibration: Optional[CalibrationResult] = None

    @property
    def vfov_deg(self) -> Optional[float]:
        return self._vfov

    @property
    def calibration(self) -> Optional[CalibrationResult]:
        return self._calibration

    def calibrate(self, samples: List[DistanceSample]) -> CalibrationResult:
        """Fit the FOV (if not set) and report error metrics.

        Returns a CalibrationResult summarising fit quality.
        """
        if not samples:
            raise ValueError("No calibration samples provided")

        # Fit a single VFOV that minimises RMSE across all samples.
        best_vfov = self._vfov
        best_rmse = float("inf")
        for vfov in np.linspace(1.0, 120.0, 240):
            est = self._estimate_all(samples, vfov)
            metrics = compute_metrics(est)
            if metrics["rmse_m"] < best_rmse:
                best_rmse = metrics["rmse_m"]
                best_vfov = float(vfov)
        self._vfov = best_vfov

        estimates = self._estimate_all(samples, self._vfov)
        metrics = compute_metrics(estimates)

        per_label: Dict[str, Dict[str, float]] = {}
        labels = {s.label for s in samples}
        for label in labels:
            sub = [(est, real) for (est, real), s in zip(estimates, samples)
                   if s.label == label]
            per_label[label] = compute_metrics(sub)

        self._calibration = CalibrationResult(
            samples=len(samples),
            mae_m=metrics["mae_m"],
            rmse_m=metrics["rmse_m"],
            relative_error=metrics["relative_error"],
            per_label=per_label,
            calibrated_vfov_deg=best_vfov,
        )
        _logger.info(
            "Calibration: VFOV=%.1fdeg MAE=%.2fm RMSE=%.2fm rel=%.0f%% "
            "(%d samples)",
            best_vfov, metrics["mae_m"], metrics["rmse_m"],
            metrics["relative_error"] * 100.0, len(samples),
        )
        return self._calibration

    def estimate(
        self,
        box: Tuple[int, int, int, int],
        frame_h: int,
        label: str,
    ) -> Tuple[float, Optional[float]]:
        """Estimate distance (m) and absolute error bound (m) if calibrated.

        Returns:
            (distance_metres, error_metres).  error_metres is None when
            no calibration data exists (unknown confidence).
        """
        from src.navigation.guidance import distance_estimate

        ref = self._heights.get(label, 1.5)
        d = distance_estimate(box, frame_h, ref,
                              vfov_deg=self._vfov or DEFAULT_VFOV_DEG)
        if self._calibration is None:
            return max(0.2, d), None
        # Use calibration MAE as a conservative, documented error bound.
        return max(0.2, d), self._calibration.mae_m

    def _estimate_all(
        self,
        samples: List[DistanceSample],
        vfov: float,
    ) -> List[Tuple[float, float]]:
        from src.navigation.guidance import distance_estimate

        out = []
        for s in samples:
            ref = (s.reference_height_m or
                   self._heights.get(s.label, 1.5))
            d = distance_estimate(
                (0, 0, 1, s.box_height_px), s.frame_height_px, ref,
                vfov_deg=vfov,
            )
            out.append((d, s.real_distance_m))
        return out


def make_default_dataset() -> List[DistanceSample]:
    """A small, honest, synthetic calibration dataset.

    These are *synthetic* samples (box heights generated from the pinhole
    model at known distances) so the calibration machinery can be tested
    without a camera.  For real-world accuracy they MUST be replaced by
    measurements taken against a tape measure / ruler; see
    ``tools/calibrate_camera.py``.
    """
    refs = {"person": 1.7, "chair": 0.9, "stop sign": 2.0}
    samples = [
        DistanceSample("person", 320.0, 720.0, 1.0, refs["person"]),
        DistanceSample("person", 160.0, 720.0, 2.0, refs["person"]),
        DistanceSample("person", 107.0, 720.0, 3.0, refs["person"]),
        DistanceSample("chair", 170.0, 720.0, 1.0, refs["chair"]),
        DistanceSample("chair", 85.0, 720.0, 2.0, refs["chair"]),
        DistanceSample("stop sign", 377.0, 720.0, 1.0, refs["stop sign"]),
        DistanceSample("stop sign", 189.0, 720.0, 2.0, refs["stop sign"]),
    ]
    return samples