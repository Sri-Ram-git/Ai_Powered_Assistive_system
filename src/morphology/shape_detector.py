"""Geometric shape detection from binary/grayscale images.

Detects circles, rectangles, and triangles (and generic polygons) by
analysing contour geometry: vertex count from polygon approximation plus
circularity from area/perimeter ratio.
"""
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from src.image_processing.processing import threshold
from src.morphology.contour_utils import (
    contour_area,
    contour_perimeter,
    center_of_mass,
)
from src.utils.logger import setup_logger

_logger = setup_logger("ShapeDetector")


@dataclass
class ShapeResult:
    """Description of one detected shape."""

    shape: str                 # 'circle', 'rectangle', 'triangle', ...
    contour: np.ndarray
    area: float
    perimeter: float
    bounding_box: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[float, float]
    vertices: int


class ShapeDetector:
    """Detects and classifies geometric shapes in an image.

    Usage:
        detector = ShapeDetector()
        results = detector.detect(binary_image)
        for r in results:
            print(r.shape, r.area)
    """

    def __init__(self, min_area: float = 100.0,
                 approx_epsilon_ratio: float = 0.02,
                 circularity_threshold: float = 0.85) -> None:
        """Configure detection parameters.

        Args:
            min_area: Ignore contours smaller than this area.
            approx_epsilon_ratio: Polygon-approximation tolerance as a
                fraction of the contour perimeter.
            circularity_threshold: Area/perimeter² ratio above which a
                high-vertex contour is classed as a circle.
        """
        self._min_area = min_area
        self._epsilon_ratio = approx_epsilon_ratio
        self._circularity = circularity_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> List[ShapeResult]:
        """Detect shapes in a binary (or grayscale/colour) image.

        Args:
            image: Input image. Colour/BGR images are thresholded
                automatically; pass a binary image for best control.

        Returns:
            List of ShapeResult, largest-first by area.
        """
        binary = self._to_binary(image)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        results: List[ShapeResult] = []
        for contour in contours:
            area = contour_area(contour)
            if area < self._min_area:
                continue

            perimeter = contour_perimeter(contour)
            approx = self._approximate(contour)
            vertices = len(approx)
            shape = classify_shape(contour, vertices,
                                   circularity_threshold=self._circularity)

            results.append(ShapeResult(
                shape=shape,
                contour=contour,
                area=area,
                perimeter=perimeter,
                bounding_box=tuple(int(v) for v in cv2.boundingRect(contour)),
                center=center_of_mass(contour),
                vertices=vertices,
            ))

        results.sort(key=lambda r: r.area, reverse=True)
        _logger.info("Detected %d shape(s)", len(results))
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _approximate(self, contour: np.ndarray) -> np.ndarray:
        perimeter = contour_perimeter(contour)
        epsilon = self._epsilon_ratio * perimeter
        return cv2.approxPolyDP(contour, epsilon, True)

    @staticmethod
    def _to_binary(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            # Assume already binary-ish if only two distinct levels.
            levels = np.unique(image)
            if levels.size <= 2:
                return image
            _, binary = cv2.threshold(image, 127, 255,
                                      cv2.THRESH_BINARY)
            return binary
        return threshold(image)


def classify_shape(contour: np.ndarray, vertices: int,
                   circularity_threshold: float = 0.85) -> str:
    """Classify a contour by its polygon approximation.

    Args:
        contour: The contour point array.
        vertices: Number of polygon vertices.
        circularity_threshold: Area/perimeter² threshold for circles.

    Returns:
        One of 'triangle', 'rectangle', 'circle', or a generic
        'polygon-<n>' label.
    """
    if vertices == 3:
        return "triangle"
    if vertices == 4:
        return "rectangle"
    if vertices > 4:
        area = contour_area(contour)
        perimeter = contour_perimeter(contour)
        if perimeter > 0:
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity >= circularity_threshold:
                return "circle"
    return f"polygon-{vertices}"


def label_image(image: np.ndarray, results: List[ShapeResult],
                color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Return a copy of the image annotated with detected shape labels."""
    display = image.copy()
    if display.ndim == 2:
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
    for r in results:
        x, y, w, h = r.bounding_box
        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
        cx, cy = r.center
        cv2.putText(display, r.shape, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.circle(display, (int(cx), int(cy)), 4, (0, 0, 255), -1)
    return display
