"""Depth estimation module (optional).

Adds monocular depth estimation to the pipeline.  The module is fully
optional: the core pipeline runs without it.  A pluggable backend
interface lets us evaluate Depth Anything V2 (or an ONNX export) against
a synthetic fallback without changing the pipeline.

Backends:
    ``DepthAnythingV2Backend`` — wrapper for a Depth Anything V2 model
        (ONNX export).  Kept behind an optional import: it is only
        instantiated when the model file is present, so the core never
        hard-depends on it.
    ``SyntheticDepthBackend`` — deterministic test/fallback backend that
        produces a plausible depth map and per-box depth without a model
        (used for tests and offline fallback).
"""
from src.depth.depth_estimator import (
    DepthEstimator,
    create_depth_estimator,
)
from src.depth.depth_result import DepthResult

__all__ = [
    "DepthEstimator",
    "DepthResult",
    "create_depth_estimator",
]