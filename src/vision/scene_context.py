"""Scene Context — the deterministic internal world model.

Combines the outputs of the perception stages (YOLO objects, OCR text,
depth, tracking) into one structured snapshot that the safety engine,
response planner, and (optionally) a VLM consume.  This is the internal
representation of "what the system believes is around the user".

Deliberately deterministic and dependency-free: no LLM is involved in
building it.
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from src.tracking.tracker import TrackedObject


@dataclass
class SceneObject:
    """One object in the scene context."""

    label: str
    confidence: float
    box: tuple                       # (x, y, w, h)
    direction: str                   # left | ahead | right
    distance_m: Optional[float]      # None = not estimated
    track_id: Optional[int] = None
    velocity: Optional[float] = None  # m/s (positive = approaching)
    category: str = "object"

    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 2),
            "direction": self.direction,
            "distance_m": (round(self.distance_m, 2)
                           if self.distance_m is not None else None),
            "track_id": self.track_id,
            "velocity": (round(self.velocity, 2)
                         if self.velocity is not None else None),
            "category": self.category,
        }


@dataclass
class SceneContext:
    """A full snapshot of the scene at a point in time."""

    objects: List[SceneObject] = field(default_factory=list)
    text: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    frame_w: int = 0
    frame_h: int = 0
    depth_available: bool = False

    @property
    def nearest(self) -> Optional[SceneObject]:
        """The closest object (by distance, then by box area)."""
        with_distance = [o for o in self.objects
                         if o.distance_m is not None]
        if with_distance:
            return min(with_distance, key=lambda o: o.distance_m)
        if self.objects:
            return max(self.objects, key=lambda o: _area(o.box))
        return None

    @property
    def hazards(self) -> List[SceneObject]:
        """Objects considered hazardous by the safety engine."""
        return [o for o in self.objects if o.category in _HAZARD_CATEGORIES]

    def to_dict(self) -> Dict:
        return {
            "objects": [o.to_dict() for o in self.objects],
            "text": list(self.text),
            "timestamp": self.timestamp,
            "frame": [self.frame_w, self.frame_h],
            "depth_available": self.depth_available,
        }


_HAZARD_CATEGORIES = {
    "vehicle", "traffic signal", "stop sign", "obstacle",
}


def build_scene_context(
    tracks: Sequence[TrackedObject],
    ocr_text: Sequence[str],
    frame_w: int,
    frame_h: int,
    distance_of=None,
    direction_of=None,
    depth_available: bool = False,
) -> SceneContext:
    """Build a SceneContext from tracked objects and OCR text.

    Args:
        tracks: Tracked objects from the IoUTracker.
        ocr_text: Recognised text lines (strings).
        frame_w, frame_h: Frame geometry.
        distance_of: Optional callable(track, frame_h) -> metres.
            Defaults to the heuristic pinhole estimate.
        direction_of: Optional callable(track, frame_w) -> str.
            Defaults to the guidance zone logic.
        depth_available: Whether a depth map fed this snapshot.

    Returns:
        A SceneContext with all deterministic fields populated.
    """
    from src.navigation.guidance import (
        direction_of as _default_direction,
        distance_estimate,
        reference_height,
    )

    if distance_of is None:
        def distance_of(t, fh):
            return distance_estimate(t.box, fh,
                                     reference_height(t.label))

    if direction_of is None:
        direction_of = _default_direction

    objects = []
    for t in tracks:
        dist = distance_of(t, frame_h)
        objects.append(SceneObject(
            label=t.label,
            confidence=t.confidence,
            box=t.box,
            direction=direction_of(t.box, frame_w),
            distance_m=dist,
            track_id=t.track_id,
            category=t.category,
        ))

    return SceneContext(
        objects=objects,
        text=list(ocr_text),
        frame_w=frame_w,
        frame_h=frame_h,
        depth_available=depth_available,
    )


def _area(box) -> float:
    _, _, w, h = box
    return float(w * h)