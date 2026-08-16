"""OCR preprocessing strategies.

RapidOCR is the current bottleneck (~2.8-4.7 s/frame on CPU).  Feeding
the OCR engine a smaller / cleaner input can cut that latency
significantly.  This module provides the strategies that the OCR worker
and the benchmark script evaluate:

* ``none``        — full frame, untouched.
* ``gray``        — single-channel grayscale (less pixel work).
* ``threshold``   — binary threshold (max contrast, no colour).
* ``contrast``    — CLAHE contrast enhancement (helps low-light text).
* ``downscale``   — halve the frame before OCR.
* ``downscale2``  — quarter the frame before OCR.

All strategies are pure functions returning a NEW array, so they are
safe to benchmark and safe to call on shared camera frames.
"""
from typing import List

import cv2
import numpy as np

SUPPORTED_STRATEGIES: List[str] = [
    "none",
    "gray",
    "threshold",
    "contrast",
    "downscale",
    "downscale2",
]

_DEFAULT_CONTRAST_LIMIT = 2.0
_DEFAULT_THRESHOLD = 127


def _validate(frame: np.ndarray) -> None:
    if frame is None or frame.size == 0:
        raise ValueError("OCR preprocess: empty frame")


def preprocess(
    frame: np.ndarray,
    strategy: str = "none",
    threshold: int = _DEFAULT_THRESHOLD,
    contrast_limit: float = _DEFAULT_CONTRAST_LIMIT,
) -> np.ndarray:
    """Apply the named preprocessing strategy to a frame.

    Args:
        frame: BGR (or grayscale) image.
        strategy: One of SUPPORTED_STRATEGIES.
        threshold: Binary threshold value (for "threshold").
        contrast_limit: CLAHE clip limit (for "contrast").

    Returns:
        A processed copy of the frame.

    Raises:
        ValueError: For an unknown strategy or an empty frame.
    """
    _validate(frame)
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"Unknown OCR preprocess strategy '{strategy}'. "
            f"Supported: {SUPPORTED_STRATEGIES}"
        )

    if strategy == "none":
        return frame.copy()
    if strategy == "gray":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if strategy == "threshold":
        gray = _to_gray(frame)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return binary
    if strategy == "contrast":
        gray = _to_gray(frame)
        clahe = cv2.createCLAHE(clipLimit=contrast_limit,
                                tileGridSize=(8, 8))
        return clahe.apply(gray)
    if strategy == "downscale":
        return _resize(frame, 0.5)
    if strategy == "downscale2":
        return _resize(frame, 0.25)
    raise ValueError(f"Unhandled strategy: {strategy}")  # pragma: no cover


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _resize(frame: np.ndarray, scale: float) -> np.ndarray:
    h, w = frame.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(frame, (new_w, new_h),
                      interpolation=interpolation)