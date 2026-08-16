"""Robust multi-object tracker (Phase 8-13).

Associates detections frame-to-frame into persistent identities.  The
previous IoU-only matcher stole IDs when boxes of different classes
overlapped (a "chair" box riding on a "person" track) and boxes jittered
because nothing was smoothed.  This version:

    association   affinity = w_iou*IoU + w_center*centre-proximity
                  + w_size*size-similarity, with a strong penalty when
                  the class differs (never steal a person's ID with a
                  chair box).  Greedy, highest-confidence first.
    smoothing     EMA on box + confidence so the display does not jitter.
    labels        temporal majority voting over a rolling window so a
                  single misclassification does not flip the spoken label.

No ML tracking models or extra dependencies — just NumPy.
"""
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from src.detection.detector import DetectionResult
from src.utils.logger import setup_logger

_logger = setup_logger("IoUTracker")


@dataclass
class TrackedObject:
    """A stable, persistent object identity across frames.

    Fields (Phase 9):
        track_id, label, confidence, box (smoothed), center, width,
        height, age, missed (consecutive unmatched frames), first_seen,
        last_seen, timestamp, direction/distance (derived downstream).
    """

    track_id: int
    label: str
    box: Tuple[int, int, int, int]          # smoothed (x, y, w, h)
    confidence: float                        # smoothed confidence
    age: int = 0
    missed: int = 0
    center: Tuple[float, float] = field(default=(0.0, 0.0))
    first_seen: int = 0
    last_seen: int = 0
    timestamp: float = 0.0
    raw_box: Tuple[int, int, int, int] = field(default=(0, 0, 0, 0))
    raw_confidence: float = 0.0
    label_history: Deque[str] = field(default_factory=deque)
    _last_direction: str = "ahead"
    _last_distance_m: float = 0.0

    @property
    def width(self) -> int:
        return int(self.box[2])

    @property
    def height(self) -> int:
        return int(self.box[3])

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

    @property
    def direction(self) -> str:
        """Last computed left/centre/right direction (cached)."""
        return self._last_direction

    @property
    def distance_m(self) -> float:
        """Last computed distance estimate in metres (cached)."""
        return self._last_distance_m


class IoUTracker:
    """Associates detections into persistent tracks.

    Args:
        iou_threshold: Minimum IoU for a candidate match (used inside the
            affinity score, not as a hard gate by itself).
        max_missed: Drop a track after this many unmatched frames.
        smoothing: EMA alpha for the box (0 = no smoothing, 1 = raw).
        conf_smoothing: EMA alpha for confidence (0..1).
        class_consistent: Penalise matches across different classes so
            boxes never steal another object's identity.
        label_vote_window: Rolling window (frames) for label voting.
        label_vote_ratio: Fraction of the window a label needs before
            the track's displayed label switches (e.g. 0.6).
        w_iou, w_center, w_size: Affinity weights (must sum to 1).
        min_affinity: Minimum combined affinity to associate.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missed: int = 8,
        smoothing: float = 0.4,
        conf_smoothing: float = 0.5,
        class_consistent: bool = True,
        label_vote_window: int = 5,
        label_vote_ratio: float = 0.6,
        w_iou: float = 0.6,
        w_center: float = 0.25,
        w_size: float = 0.15,
        min_affinity: float = 0.3,
    ) -> None:
        self._iou_threshold = float(iou_threshold)
        self._max_missed = int(max_missed)
        self._smoothing = float(smoothing)
        self._conf_smoothing = float(conf_smoothing)
        self._class_consistent = bool(class_consistent)
        self._vote_window = int(label_vote_window)
        self._vote_ratio = float(label_vote_ratio)
        self._w_iou = float(w_iou)
        self._w_center = float(w_center)
        self._w_size = float(w_size)
        self._min_affinity = float(min_affinity)

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
        """Associate a new batch of detections into tracks."""
        self._frame += 1
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
        """Match detections to tracks greedily by combined affinity."""
        matched_track: set = set()
        order = sorted(range(len(detections)),
                       key=lambda i: detections[i].confidence, reverse=True)

        for det_idx in order:
            det = detections[det_idx]
            det_rect = det_boxes[det_idx]

            best_track_id: Optional[int] = None
            best_score = self._min_affinity

            for track_id, track in self._tracks.items():
                if track_id in matched_track:
                    continue
                score = self._affinity(
                    det, det_rect, track, _x1y1x2y2(track.box))
                if score > best_score:
                    best_score = score
                    best_track_id = track_id

            if best_track_id is not None:
                track = self._tracks[best_track_id]
                self._associate(track, det)
                matched_track.add(best_track_id)
            else:
                track = self._new_track(det)
                self._tracks[track.track_id] = track
                matched_track.add(track.track_id)

        return matched_track

    def _affinity(
        self,
        det: DetectionResult,
        det_rect: Tuple[float, float, float, float],
        track: TrackedObject,
        track_rect: Tuple[float, float, float, float],
    ) -> float:
        """Combined IoU + centre-proximity + size-similarity score."""
        iou = _iou(det_rect, track_rect)

        cx_d, cy_d = _center(det.box)
        cx_t, cy_t = track.center
        tw, th = track.width, max(1, track.height)
        diag = float(np.hypot(tw, th)) or 1.0
        centre_dist = float(np.hypot(cx_d - cx_t, cy_d - cy_t))
        centre_score = max(0.0, 1.0 - centre_dist / diag)

        dw, dh = det.box[2], det.box[3]
        size_score = (
            (min(dw, tw) / max(1.0, max(dw, tw))) *
            (min(dh, th) / max(1.0, max(dh, th)))
        )

        score = (
            self._w_iou * iou +
            self._w_center * centre_score +
            self._w_size * size_score
        )
        # Phase 10: never let a different-class box take an identity.
        if self._class_consistent and det.label != track.label:
            score *= 0.05
        return float(score)

    def _associate(self, track: TrackedObject, det: DetectionResult) -> None:
        """Update a matched track: smooth, vote, extend."""
        track.age += 1
        track.missed = 0
        track.raw_box = det.box
        track.raw_confidence = det.confidence

        # Phase 11: EMA smoothing of the box.
        if self._smoothing <= 0.0 or track.age == 1:
            track.box = det.box
        else:
            a = self._smoothing
            x, y, w, h = det.box
            tx, ty, tw, th = track.box
            track.box = (
                int(round(a * x + (1 - a) * tx)),
                int(round(a * y + (1 - a) * ty)),
                int(round(a * w + (1 - a) * tw)),
                int(round(a * h + (1 - a) * th)),
            )

        # Phase 13: EMA smoothing of confidence.
        if self._conf_smoothing <= 0.0 or track.age == 1:
            track.confidence = det.confidence
        else:
            a = self._conf_smoothing
            track.confidence = a * det.confidence + (1 - a) * track.confidence

        track.center = _center(track.box)
        track.last_seen = self._frame
        track.timestamp = time.time()

        # Phase 12: temporal label voting.
        track.label_history.append(det.label)
        while len(track.label_history) > self._vote_window:
            track.label_history.popleft()
        track.label = self._voted_label(track)

    def _voted_label(self, track: TrackedObject) -> str:
        """Majority label over the window; switch only on a clear win.

        A single spurious detection must not flip the spoken label.
        """
        if not track.label_history:
            return track.label
        counts: Dict[str, int] = {}
        for label in track.label_history:
            counts[label] = counts.get(label, 0) + 1
        best, best_n = max(counts.items(), key=lambda kv: kv[1])
        if best_n >= self._vote_ratio * len(track.label_history):
            return best
        return track.label

    def _new_track(self, det: DetectionResult) -> TrackedObject:
        now = time.time()
        track = TrackedObject(
            track_id=self._next_id,
            label=det.label,
            box=det.box,
            confidence=det.confidence,
            age=1,
            missed=0,
            center=_center(det.box),
            first_seen=self._frame,
            last_seen=self._frame,
            timestamp=now,
            raw_box=det.box,
            raw_confidence=det.confidence,
        )
        track.label_history.append(det.label)
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