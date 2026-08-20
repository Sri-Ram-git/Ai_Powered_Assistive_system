"""Tracking monitor — turns tracked objects into spoken guidance.

Watches `TrackedObject`s produced by the IoUTracker and emits natural
phrases whenever something *changes*: a new object appears, an object's
distance changes meaningfully, or an object leaves the view.  This gives
continuous feedback (not a single one-off announcement).

Stateless core: `events(tracks, ...)` is a pure function of the current
tracks + remembered state; the `TrackingMonitor` wrapper stores the
per-track memory (last distance, last direction, announced set).
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.navigation.guidance import (
    direction_of,
    distance_estimate,
    reference_height,
)
from src.tracking.tracker import TrackedObject
from src.utils.logger import setup_logger

_logger = setup_logger("TrackingMonitor")


@dataclass
class _TrackMemory:
    """What the monitor remembers about one tracked object."""

    track_id: int
    label: str
    last_distance: float
    last_direction: str
    announced: bool = False
    last_announced: float = 0.0


class TrackingMonitor:
    """Emits guidance phrases when tracked objects change state."""

    def __init__(
        self,
        distance_change_metres: float = 1.0,
        min_announce_interval: float = 3.0,
        vfov_deg: float = 55.0,
    ) -> None:
        """Configure the monitor.

        Args:
            distance_change_metres: Minimum distance change (metres)
                before the distance is re-announced.
            min_announce_interval: Minimum seconds between announcements
                for the *same* track (avoids chatter on jitter).
            vfov_deg: Camera vertical FOV used for distance estimation.
        """
        self._distance_delta = float(distance_change_metres)
        self._min_interval = float(min_announce_interval)
        self._vfov_deg = float(vfov_deg)
        self._memory: Dict[int, _TrackMemory] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def events(
        self,
        tracks: List[TrackedObject],
        frame_w: int,
        frame_h: int,
        now: Optional[float] = None,
    ) -> List[str]:
        """Return guidance phrases for the current set of tracks.

        Args:
            tracks: Active tracked objects for this frame.
            frame_w, frame_h: Frame geometry for direction/distance.
            now: Current time in seconds (defaults to wall clock).

        Returns:
            List of phrases, e.g. ["Person ahead, about 3 metres"].
        """
        import time

        if now is None:
            now = time.monotonic()

        phrases: List[str] = []
        seen: set = set()

        for track in tracks:
            seen.add(track.track_id)
            mem = self._memory.get(track.track_id)

            if mem is None or mem.label != track.label:
                # Brand-new object -> announce it.
                phrase = self._new_object_phrase(track, frame_w, frame_h)
                self._memory[track.track_id] = _TrackMemory(
                    track_id=track.track_id,
                    label=track.label,
                    last_distance=self._distance_of(track, frame_h),
                    last_direction=direction_of(track.box, frame_w),
                    announced=True,
                )
                phrases.append(phrase)
                continue
                continue

            # Existing object -> announce meaningful changes.
            phrase = self._change_phrase(
                mem, track, frame_w, frame_h, now,
            )
            if phrase:
                phrases.append(phrase)

        self._remove_lost(seen)
        return phrases

    def reset(self) -> None:
        """Forget all tracked objects (e.g. scene change)."""
        self._memory.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _new_object_phrase(
        self, track: TrackedObject, frame_w: int, frame_h: int,
    ) -> str:
        distance = self._distance_of(track, frame_h)
        direction = direction_of(track.box, frame_w)
        return (
            f"{track.label.capitalize()} {direction}, "
            f"{_distance_phrase(distance)}"
        )

    def _change_phrase(
        self,
        mem: _TrackMemory,
        track: TrackedObject,
        frame_w: int,
        frame_h: int,
        now: float,
    ) -> Optional[str]:

        distance = self._distance_of(track, frame_h)
        direction = direction_of(track.box, frame_w)

        # Threshold crossings: only announce if the change is significant
        # and enough time has passed since the previous announcement.
        delta = abs(distance - mem.last_distance)
        direction_changed = direction != mem.last_direction

        if delta < self._distance_delta and not direction_changed:
            mem.last_distance = distance
            return None

        if now - mem.last_announced < self._min_interval:
            mem.last_distance = distance
            return None

        mem.last_distance = distance
        mem.last_direction = direction
        mem.last_announced = now
        phrase = (
            f"{track.label.capitalize()} now {direction}, "
            f"{_distance_phrase(distance)}"
        )
        _logger.debug("Track %d change -> %s", track.track_id, phrase)
        return phrase

    def _remove_lost(self, seen: set) -> None:
        for track_id in list(self._memory):
            if track_id not in seen:
                del self._memory[track_id]


    def _distance_of(self, track: TrackedObject, frame_h: int) -> float:
        return distance_estimate(
            track.box, frame_h, reference_height(track.label),
            vfov_deg=self._vfov_deg,
        )


def _distance_phrase(distance: float) -> str:
    if distance >= 15.0:
        return "far away"
    if distance <= 0.5:
        return "very close"
    return f"about {distance:.0f} metres"
