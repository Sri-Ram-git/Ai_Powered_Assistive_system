"""Unit tests for morphology & shape detection (hardware-free)."""
import cv2
import numpy as np
import pytest

from src.morphology import contour_utils as C
from src.morphology.shape_detector import (
    ShapeDetector,
    classify_shape,
)
from src.utils.exceptions import ProcessingError


@pytest.fixture
def binary_shapes() -> np.ndarray:
    """Binary scene with one circle, rectangle, and triangle."""
    image = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(image, (80, 80), 50, 255, -1)
    cv2.rectangle(image, (160, 50), (260, 150), 255, -1)
    pts = np.array([[100, 230], [50, 290], [150, 290]], np.int32)
    cv2.fillPoly(image, [pts], 255)
    return image


class TestMorphology:
    def test_erode_shrinks(self, sample_gray):
        mask = (sample_gray > 127).astype(np.uint8) * 255
        before = int(mask.sum())
        after = int(C.erode(mask, 3).sum())
        assert after < before

    def test_dilate_grows(self, sample_gray):
        mask = (sample_gray > 127).astype(np.uint8) * 255
        before = int(mask.sum())
        after = int(C.dilate(mask, 3).sum())
        assert after > before

    def test_open_removes_specks(self):
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(image, (40, 40), (60, 60), 255, -1)
        image[5, 5] = 255  # speck
        opened = C.opening(image, 3)
        assert opened[5, 5] == 0
        assert opened[50, 50] == 255

    def test_close_fills_holes(self):
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(image, (40, 40), (60, 60), 255, -1)
        image[50, 50] = 0  # hole
        closed = C.closing(image, 3)
        assert closed[50, 50] == 255

    def test_even_kernel_rejected(self, binary_shapes):
        with pytest.raises(ProcessingError):
            C.erode(binary_shapes, 4)


class TestContours:
    def test_find_and_metrics(self, binary_shapes):
        contours, _ = C.find_contours(binary_shapes)
        assert len(contours) == 3
        areas = sorted(C.contour_area(c) for c in contours)
        assert areas[0] < areas[-1]

    def test_bounding_rect(self):
        contour = np.array([[[0, 0]], [[10, 0]], [[10, 5]], [[0, 5]]])
        # OpenCV's boundingRect is inclusive of the max coordinate
        assert C.bounding_rect(contour) == (0, 0, 11, 6)

    def test_center_of_mass_rectangle(self):
        contour = np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]])
        cx, cy = C.center_of_mass(contour)
        assert (cx, cy) == pytest.approx((5.0, 5.0), abs=0.5)

    def test_convex_hull_smaller_than_concave(self):
        # U-shape has a concavity -> hull is bigger
        u = np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[5, 5]],
                      [[0, 10]]], np.int32)
        hull = C.convex_hull(u)
        assert len(hull) < len(u)

    def test_contours_report(self, binary_shapes):
        contours, _ = C.find_contours(binary_shapes)
        report = C.contours_report(contours)
        assert "area=" in report and "perim=" in report


class TestShapeDetector:
    def test_detects_known_shapes(self, binary_shapes):
        detector = ShapeDetector(min_area=500)
        results = detector.detect(binary_shapes)
        shapes = {r.shape for r in results}
        assert {"circle", "rectangle", "triangle"} <= shapes

    def test_detects_largest_first(self, binary_shapes):
        detector = ShapeDetector(min_area=500)
        results = detector.detect(binary_shapes)
        areas = [r.area for r in results]
        assert areas == sorted(areas, reverse=True)

    def test_min_area_filters(self, binary_shapes):
        detector = ShapeDetector(min_area=10_000)
        results = detector.detect(binary_shapes)
        assert len(results) <= 1

    def test_classify_vertices(self):
        triangle = np.array([[[0, 0]], [[5, 0]], [[0, 5]]], np.int32)
        assert classify_shape(triangle, 3) == "triangle"
        rect = np.array([[[0, 0]], [[5, 0]], [[5, 5]], [[0, 5]]], np.int32)
        assert classify_shape(rect, 4) == "rectangle"
