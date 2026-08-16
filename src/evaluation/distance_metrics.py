"""Distance evaluation metrics: MAE, RMSE, relative error.

Thin re-export of the calibration metrics so the evaluation package has
one public entry point for distance numbers.
"""
from src.navigation.calibration import (  # noqa: F401
    CalibratedDistanceEstimator,
    DistanceSample,
    compute_metrics,
)