"""Morphological operations and contour utilities.

Morphological ops (erode, dilate, open, close) work on binary images;
contour utilities extract and describe object boundaries.
"""
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.utils.exceptions import ProcessingError
from src.utils.logger import setup_logger

_logger = setup_logger("Morphology")

Array = np.ndarray


def _check_kernel(size: int) -> None:
    if size % 2 == 0 or size < 1:
        raise ProcessingError(f"Kernel size must be positive and odd, "
                              f"got {size}.")


def _kernel(size: int) -> np.ndarray:
    _check_kernel(size)
    return np.ones((size, size), dtype=np.uint8)


# ----------------------------------------------------------------------
# Morphological operations
# ----------------------------------------------------------------------

def erode(image: Array, kernel_size: int = 3, iterations: int = 1) -> Array:
    """Erode (shrink) bright regions of a binary image.

    Args:
        image: Binary (grayscale) input.
        kernel_size: Square structuring-element size.
        iterations: Number of times to apply erosion.
    """
    return cv2.erode(image, _kernel(kernel_size), iterations=iterations)


def dilate(image: Array, kernel_size: int = 3, iterations: int = 1) -> Array:
    """Dilate (grow) bright regions of a binary image."""
    return cv2.dilate(image, _kernel(kernel_size), iterations=iterations)


def opening(image: Array, kernel_size: int = 3, iterations: int = 1) -> Array:
    """Opening = erosion followed by dilation. Removes small specks."""
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, _kernel(kernel_size),
                            iterations=iterations)


def closing(image: Array, kernel_size: int = 3, iterations: int = 1) -> Array:
    """Closing = dilation followed by erosion. Fills small holes."""
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, _kernel(kernel_size),
                            iterations=iterations)


def morph_open(image: Array, kernel_size: int = 3,
               iterations: int = 1) -> Array:
    """Alias for :func:`opening`."""
    return opening(image, kernel_size, iterations)


def morph_close(image: Array, kernel_size: int = 3,
                iterations: int = 1) -> Array:
    """Alias for :func:`closing`."""
    return closing(image, kernel_size, iterations)


# ----------------------------------------------------------------------
# Contours
# ----------------------------------------------------------------------

def find_contours(
    image: Array,
    mode: int = cv2.RETR_EXTERNAL,
    method: int = cv2.CHAIN_APPROX_SIMPLE,
) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    """Find contours in a binary image.

    Args:
        image: Binary (grayscale) input.  Only the outermost contours are
            returned by default.
        mode: OpenCV contour retrieval mode.
        method: OpenCV contour approximation method.

    Returns:
        (contours, hierarchy) tuple.
    """
    contours, hierarchy = cv2.findContours(image, mode, method)
    return list(contours), hierarchy


def contour_area(contour: np.ndarray) -> float:
    """Area enclosed by a contour."""
    return float(cv2.contourArea(contour))


def contour_perimeter(contour: np.ndarray, closed: bool = True) -> float:
    """Arc length (perimeter) of a contour."""
    return float(cv2.arcLength(contour, closed))


def bounding_rect(contour: np.ndarray) -> Tuple[int, int, int, int]:
    """Axis-aligned bounding rectangle as (x, y, width, height)."""
    return tuple(int(v) for v in cv2.boundingRect(contour))


def convex_hull(contour: np.ndarray) -> np.ndarray:
    """Convex hull of a contour (returned as a point array)."""
    return cv2.convexHull(contour)


def convexity_defects(contour: np.ndarray) -> Optional[np.ndarray]:
    """Indices of the deepest convexity defects on a closed contour.

    Returns None if the contour is already fully convex.
    """
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)
    return defects


def center_of_mass(contour: np.ndarray) -> Tuple[float, float]:
    """Centroid (centre of mass) of a contour as (x, y)."""
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return 0.0, 0.0
    return moments["m10"] / moments["m00"], \
        moments["m01"] / moments["m00"]


def draw_contours(image: Array, contours: List[np.ndarray],
                  color: Tuple[int, int, int] = (0, 255, 0),
                  thickness: int = 2) -> Array:
    """Return a copy of the image with contours drawn on it."""
    display = image.copy()
    cv2.drawContours(display, contours, -1, color, thickness)
    return display


def contours_report(contours: List[np.ndarray]) -> str:
    """Return a human-readable summary of contours."""
    lines = []
    for i, contour in enumerate(contours):
        x, y, w, h = bounding_rect(contour)
        lines.append(
            f"[{i}] area={contour_area(contour):7.1f} "
            f"perim={contour_perimeter(contour):7.1f} "
            f"bbox=({x},{y},{w},{h})"
        )
    return "\n".join(lines)
