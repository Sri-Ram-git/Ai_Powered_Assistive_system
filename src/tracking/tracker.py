"""Lightweight multi-object tracker (IoU association).

Associates detections frame-to-frame by box intersection-over-union so
each object keeps a stable ID.  No ML tracking models or extra
dependencies — just NumPy.  Suitable for the assistive pipeline where
detections arrive every few frames.

Usage:
    tracker = IoUTracker()
    for detections in frames:
        tracks = tracker.update(detections)
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.detection.detector import DetectionResult
from src.utils.logger import setup_logger

_logger = setup_logger("IoUTracker")


@dataclass
class TrackedObject:
    """A stable, persistent object identity across frames."""

    track_id: int
    label: str
    box: Tuple[int, int, int, int]       # (x, y, w, h) frame coords
    confidence: float
    age: int = 0                          # frames this track has lived
    missed: int = 0                       # consecutive frames without a match
    center: Tuple[float, float] = field(default=(0.0, 0.0))
    first_seen: int = 0

    @property
    def area(self) -> float:
        _, _, w, h = self.box
        return float(w * h)

    @property
    def alive(self) -> bool:
        return self.missed == 0

    @property
    def category(self) -> str:
        """Coarse group (vehicle/person/...) used by the decision engine."""
        from src.detection.detector import NAVIGATION_CLASSES

        return NAVIGATION_CLASSES.get(self.label, "object")


class IoUTracker:
    """Associates detections into persistent tracks by IoU matching."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missed: int = 8,
    ) -> None:
        """Configure the tracker.

        Args:
            iou_threshold: Minimum IoU to associate a detection with a
                track (detections are throttled, so keep this modest).
            max_missed: Drop a track after this many unmatched frames.
        """
        self._iou_threshold = float(iou_threshold)
        self._max_missed = int(max_missed)
        self._tracks: Dict[int, TrackedObject] = {}
        self._next_id = 0
        self._frame = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def active_tracks(self) -> List[TrackedObject]:
        """Tracks that had a detection in the current frame."""
        return [t for t in self._tracks.values() if t.alive]

    def all_tracks(self) -> List[TrackedObject]:
        """All tracks (including ones being dropped)."""
        return list(self._tracks.values())

    def update(self, detections: List[DetectionResult]) -> List[TrackedObject]:
        """Associate a new batch of detections into tracks.

        Args:
            detections: Detections for the current frame (frame coords).

        Returns:
            List of active TrackedObject (those matched this frame).
        """
        self._frame += 1

        # Build detection rects (x1, y1, x2, y2).
        det_boxes = [_x1y1x2y2(d.box) for d in detections]

        matched_track_ids = self._greedy_match(det_boxes, detections)

        for track_id, track in self._tracks.items():
            if track_id in matched_track_ids:
                continue
            track.missed += 1

        self._drop_stale()

        active = self.active_tracks
        _logger.debug(
            "Frame %d: %d detections, %d active tracks",
            self._frame, len(detections), len(active),
        )
        return active

    def reset(self) -> None:
        """Clear all tracks (e.g. scene change)."""
        self._tracks.clear()
        self._next_id = 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _greedy_match(
        self,
        det_boxes: List[Tuple[int, int, int, int]],
        detections: List[DetectionResult],
    ) -> set:
        """Match detections to tracks greedily by IoU (high conf first)."""
        matched_track: set = set()
        order = sorted(range(len(detections)),
                       key=lambda i: detections[i].confidence, reverse=True)

        for det_idx in order:
            det = detections[det_idx]
            det_rect = det_boxes[det_idx]

            best_track_id: Optional[int] = None
            best_iou = self._iou_threshold

            for track_id, track in self._tracks.items():
                if track_id in matched_track:
                    continue
                iou = _iou(det_rect, _x1y1x2y2(track.box))
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None:
                track = self._tracks[best_track_id]
                track.box = det.box
                track.confidence = det.confidence
                track.age += 1
                track.missed = 0
                track.center = _center(det.box)
                matched_track.add(best_track_id)
            else:
                track = self._new_track(det)
                self._tracks[track.track_id] = track
                matched_track.add(track.track_id)

        return matched_track

    def _new_track(self, det: DetectionResult) -> TrackedObject:
        track = TrackedObject(
            track_id=self._next_id,
            label=det.label,
            box=det.box,
            confidence=det.confidence,
            age=1,
            missed=0,
            center=_center(det.box),
            first_seen=self._frame,
        )
        self._next_id += 1
        _logger.debug("New track id=%d label=%s", track.track_id, det.label)
        return track

    def _drop_stale(self) -> None:
        stale = [tid for tid, t in self._tracks.items()
                 if t.missed > self._max_missed]
        for tid in stale:
            t = self._tracks.pop(tid)
            _logger.debug("Dropped track id=%d label=%s", tid, t.label)


def _x1y1x2y2(box: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
    x, y, w, h = box
    return float(x), float(y), float(x + w), float(y + h)


def _iou(a: Tuple[float, float, float, float],
         b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return inter / union


def _center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)
