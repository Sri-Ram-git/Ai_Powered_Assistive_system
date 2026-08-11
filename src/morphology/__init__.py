"""Morphology & Shape Analysis module.

contour_utils:  erode, dilate, opening, closing, contour metrics,
                bounding boxes, convex hull, centroid.
shape_detector: ShapeDetector (circle / rectangle / triangle detection)
                and label_image annotation helper.
"""
from src.morphology.contour_utils import (
    erode,
    dilate,
    opening,
    closing,
    morph_open,
    morph_close,
    find_contours,
    contour_area,
    contour_perimeter,
    bounding_rect,
    convex_hull,
    convexity_defects,
    center_of_mass,
    draw_contours,
    contours_report,
)
from src.morphology.shape_detector import (
    ShapeDetector,
    ShapeResult,
    classify_shape,
    label_image,
)

__all__ = [
    "erode",
    "dilate",
    "opening",
    "closing",
    "morph_open",
    "morph_close",
    "find_contours",
    "contour_area",
    "contour_perimeter",
    "bounding_rect",
    "convex_hull",
    "convexity_defects",
    "center_of_mass",
    "draw_contours",
    "contours_report",
    "ShapeDetector",
    "ShapeResult",
    "classify_shape",
    "label_image",
]