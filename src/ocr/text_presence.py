"""Cheap text-presence gate for object ROIs.

Runs *before* the slow OCR call so we never spend CPU recognising text
that is not there.  YOLO only tells us "this region contains a bottle";
it does not say the bottle has readable text.

The heuristic combines three signals that hold for real text but not for
plain surfaces:

    * gradient magnitude density  — text strokes create many edges;
    * small connected components  — letters are separate, small blobs;
    * a veto when there is no small-component structure at all — a
      uniform wall or a rectangle border produce one big blob (or none),
      never the many small blobs that characters form.

A blank wall, a plain cup, or a person's clothing yield almost no
gradient; a drawn rectangle yields one large contour and no character
blobs.  Both are skipped in well under a millisecond.
"""
import cv2
import numpy as np


def text_presence_score(roi: np.ndarray) -> float:
    """Score in [0, 1] estimating how likely the ROI contains text.

    Args:
        roi: BGR or grayscale ROI image.

    Returns:
        A score where higher = more text-like. 0.0 means "no text-like
        structure at all".
    """
    if roi is None or roi.size == 0:
        return 0.0
    gray = _to_gray(roi)
    h, w = gray.shape

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_n = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8)
    _, mask = cv2.threshold(mag_n, 70, 255, cv2.THRESH_BINARY)

    edge_ratio = float(np.count_nonzero(mask)) / max(1, mask.size)
    components = _text_like_components(mask)

    if components <= 0:
        # Plain surface or a single large shape (rectangle border).
        return 0.0

    density_score = min(1.0, edge_ratio / 0.05)
    component_score = min(1.0, components / 8.0)
    return 0.5 * density_score + 0.5 * component_score


def has_text(roi: np.ndarray, threshold: float = 0.35) -> bool:
    """Whether the ROI plausibly contains text (cheap gate)."""
    return text_presence_score(roi) >= threshold


def _text_like_components(mask: np.ndarray) -> int:
    """Count gradient blobs shaped like characters.

    Characters are small (2..50% of the ROI) blobs with an aspect ratio
    between ~0.2 and ~5.  Large blobs (whole objects, rectangle borders)
    and noise specks are ignored.
    """
    h, w = mask.shape
    min_dim = min(w, h)
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    count = 0
    for i in range(1, num):
        _, _, cw, ch, area = stats[i]
        if cw < 2 or ch < 2 or area < 4:
            continue
        if cw > 0.5 * w or ch > 0.5 * h or area > 0.25 * min_dim * min_dim:
            continue
        aspect = cw / max(1, ch)
        if 0.2 <= aspect <= 5.0:
            count += 1
    return count


def _to_gray(roi: np.ndarray) -> np.ndarray:
    if roi.ndim == 2:
        return roi
    return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)