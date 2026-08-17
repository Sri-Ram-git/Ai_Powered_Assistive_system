"""Object ROI extraction for object-aware OCR.

Turns a tracked object's bounding box into a cropped region suitable for
OCR, with configurable padding so text near the object edge is not
clipped, and smart upscaling so small text (bottles, labels, screens) has
a chance of being recognised.

Rules:
    * pad by a ratio of the box dimensions, clamped to image bounds;
    * reject degenerate ROIs (empty, zero-size, below minimum size);
    * upscale small ROIs so characters are large enough for the OCR net.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class ObjectRoi:
    """A validated, padded object region ready for OCR."""

    image: np.ndarray          # the (possibly upscaled) ROI image
    box: Tuple[int, int, int, int]      # (x1, y1, x2, y2) in frame coords
    scale: float               # upscale factor applied to the image
    original_box: Tuple[int, int, int, int]


def extract_roi(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    padding: float = 0.1,
    min_w: int = 24,
    min_h: int = 12,
) -> Optional[ObjectRoi]:
    """Extract a padded, validated ROI for an object box.

    Args:
        frame: BGR camera frame.
        box: Object box (x, y, w, h) in frame coordinates.
        padding: Fraction of box width/height added on each side.
        min_w, min_h: Minimum accepted ROI width/height (pixels).

    Returns:
        ObjectRoi or None when the ROI is too small or degenerate.
    """
    if frame is None or frame.size == 0:
        return None
    h, w = frame.shape[:2]
    x, y, bw, bh = box
    if bw <= 0 or bh <= 0:
        return None

    pad_x = int(round(bw * padding))
    pad_y = int(round(bh * padding))
    x1 = max(0, int(x) - pad_x)
    y1 = max(0, int(y) - pad_y)
    x2 = min(w, int(x) + int(bw) + pad_x)
    y2 = min(h, int(y) + int(bh) + pad_y)

    if x1 >= x2 or y1 >= y2:
        return None
    if x2 - x1 < min_w or y2 - y1 < min_h:
        return None

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    scaled, scale = smart_upscale(roi)
    return ObjectRoi(
        image=scaled,
        box=(x1, y1, x2, y2),
        scale=scale,
        original_box=(x1, y1, x2, y2),
    )


def smart_upscale(
    roi: np.ndarray,
    max_scale: float = 3.0,
) -> Tuple[np.ndarray, float]:
    """Upscale a small ROI so its characters are large enough for OCR.

    Strategy (benchmarked in docs/ocr/OCR_INTEGRATION_REPORT.md):
        min side < 32 px  -> x3
        min side < 64 px  -> x2
        otherwise         -> x1

    Args:
        roi: ROI image.
        max_scale: Hard cap on the upscale factor.

    Returns:
        (scaled_image, scale_factor).  scale == 1.0 when no upscale.
    """
    h, w = roi.shape[:2]
    min_side = min(h, w)
    if min_side < 32:
        factor = 3.0
    elif min_side < 64:
        factor = 2.0
    else:
        return roi, 1.0

    factor = min(factor, max(1.0, max_scale))
    if factor <= 1.0:
        return roi, 1.0

    new_w = max(1, int(round(w * factor)))
    new_h = max(1, int(round(h * factor)))
    scaled = cv2.resize(roi, (new_w, new_h),
                        interpolation=cv2.INTER_LINEAR)
    return scaled, factor