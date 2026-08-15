"""Environment guidance helpers for the assistive vision system.

Turns raw detections (bounding boxes) and OCR text into the coarse
guidance cues that a visually-impaired user can act on: relative
direction to the nearest obstacle, rough distance estimate, and
situational warnings (traffic signals, crosswalks, stop signs).

All functions are stateless — input image geometry + detections in,
plain guidance values out.
"""
from typing import Iterable, List, Optional, Tuple

import numpy as np

from src.detection.detector import DetectionResult
from src.ocr.ocr_engine import OcrResult

# Zones (fractions of frame width).  A box centre inside the middle
# third is "ahead"; otherwise it is left or right.
_ZONE_RATIO = 1.0 / 3.0

# Assumed vertical field of view of the camera in degrees.  Together with
# the frame height this yields a focal length in pixels, giving a rough
# pinhole distance estimate.  Override via config (`navigation.vertical_fov`).
_VERTICAL_FOV_DEG = 45.0

# Reference real-world heights (metres) used per detection label so the
# distance estimate is more meaningful than using one height for all.
# Override via config (`navigation.reference_heights`).
_REFERENCE_HEIGHTS = {
    "person": 1.7,
    "bicycle": 1.0,
    "motorcycle": 1.2,
    "car": 1.5,
    "bus": 3.2,
    "truck": 3.2,
    "chair": 0.9,
    "couch": 0.9,
    "bench": 0.9,
    "potted plant": 1.0,
    "stop sign": 2.0,
    "fire hydrant": 1.0,
    "dog": 0.6,
    "cat": 0.3,
    "bird": 0.2,
    "horse": 1.6,
    "sheep": 0.9,
    "cow": 1.4,
    "elephant": 3.2,
    "bear": 2.4,
    "zebra": 2.3,
    "giraffe": 5.0,
    "laptop": 0.25,
    "tv": 0.7,
    "cell phone": 0.15,
    "bottle": 0.3,
    "cup": 0.12,
    "backpack": 0.5,
    "suitcase": 0.7,
    "umbrella": 1.2,
    "traffic light": 0.9,
}
_DEFAULT_REFERENCE_HEIGHT = 1.5


def direction_of(box: Tuple[int, int, int, int], frame_w: int) -> str:
    """Return 'left', 'ahead', or 'right' for a bounding box.

    Args:
        box: (x, y, w, h) axis-aligned box in frame coordinates.
        frame_w: Frame width in pixels.

    Returns:
        One of 'left' | 'ahead' | 'right' based on the box centre.
    """
    if frame_w <= 0:
        return "ahead"
    x, _, w, _ = box
    centre_x = x + w / 2.0
    left_edge = _ZONE_RATIO * frame_w
    right_edge = 2.0 * _ZONE_RATIO * frame_w
    if centre_x < left_edge:
        return "left"
    if centre_x > right_edge:
        return "right"
    return "ahead"


def distance_estimate(box: Tuple[int, int, int, int],
                      frame_h: int = 480,
                      reference_metres: Optional[float] = None,
                      vfov_deg: float = _VERTICAL_FOV_DEG) -> float:
    """Estimate distance in metres from an object's box height.

    Uses a pinhole model: distance = (reference_size * focal_length) /
    box_height, where focal_length derives from the assumed vertical FOV
    and the frame height, and reference_size defaults to the label's
    typical height.

    Args:
        box: (x, y, w, h) bounding box.
        frame_h: Height of the frame in pixels (for focal estimation).
        reference_metres: Real-world height of the object; if None the
            caller is expected to have passed a label-specific default.
            (Retained for backward compatibility with the generic case.)
        vfov_deg: Camera vertical field of view in degrees.  Override
            from config when you know your webcam's real FOV — this is
            the biggest lever on distance accuracy.

    Returns:
        Approximate distance in metres (clamped to >= 0.2 m).
    """
    _, _, _, h = box
    if h <= 0 or frame_h <= 0:
        return float("inf")
    reference = reference_metres or _DEFAULT_REFERENCE_HEIGHT
    focal = (frame_h / 2.0) / np.tan(np.radians(vfov_deg / 2.0))
    distance = reference * focal / h
    return max(0.2, distance)


def reference_height(label: str) -> float:
    """Return the assumed real-world height for a detection label."""
    return _REFERENCE_HEIGHTS.get(label, _DEFAULT_REFERENCE_HEIGHT)


def nearest_obstacle(
    detections: Iterable[DetectionResult],
    frame_w: int,
) -> Optional[DetectionResult]:
    """Return the detection closest to the user (by area).

    The largest box in the frame is treated as the nearest obstacle.
    Only non-trivial detections are considered.

    Args:
        detections: Detections to consider.
        frame_w: Frame width (used to discard full-frame noise).

    Returns:
        The nearest DetectionResult, or None if nothing qualifies.
    """
    candidates = [d for d in detections if d.area > 0]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.area)


def distance_phrase(box: Tuple[int, int, int, int],
                    frame_h: int,
                    label: str) -> str:
    """Human-friendly distance phrase for a detection.

    Returns e.g. "about 2 metres" or "very close" for objects filling
    the frame (where the scale model is least reliable).
    """
    dist = distance_estimate(box, frame_h, reference_height(label))
    if dist >= 15.0:
        return "far away"
    if dist <= 0.5:
        return "very close"
    return f"about {dist:.0f} metres"


def scene_cues(
    detections: List[DetectionResult],
    ocr_items: List[OcrResult],
    frame_w: int,
    frame_h: int = 480,
) -> List[str]:
    """Produce human-readable guidance cues for a frame.

    Args:
        detections: Object detections for the frame.
        ocr_items: Recognised text lines for the frame.
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels (for distance estimation).

    Returns:
        A list of short cue strings, e.g. "Person ahead, about 2 metres".
    """
    cues: List[str] = []

    traffic_signals = [d for d in detections if d.label == "traffic light"]
    if traffic_signals:
        d = max(traffic_signals, key=lambda x: x.area)
        cues.append(f"Traffic light {direction_of(d.box, frame_w)}")

    stop_signs = [d for d in detections if d.label == "stop sign"]
    if stop_signs:
        d = max(stop_signs, key=lambda x: x.area)
        cues.append(f"Stop sign {direction_of(d.box, frame_w)}")

    for label, words in _CROSSWALK_KEYWORDS.items():
        if any(w in text.lower() for item in ocr_items
               for w in words for text in [item.text.lower()]):
            cues.append(f"{label.capitalize()} sign ahead")

    people = [d for d in detections if d.label == "person"]
    for d in people:
        cues.append(
            f"Person {direction_of(d.box, frame_w)}, "
            f"{distance_phrase(d.box, frame_h, d.label)}"
        )

    vehicles = [d for d in detections if d.category == "vehicle"]
    for d in vehicles:
        cues.append(
            f"{d.label.capitalize()} {direction_of(d.box, frame_w)}, "
            f"{distance_phrase(d.box, frame_h, d.label)}"
        )

    return cues


_CROSSWALK_KEYWORDS = {
    "crosswalk": ("crosswalk", "walk", "pedestrian"),
    "do not walk": ("don't walk", "do not walk", "dont walk"),
}
