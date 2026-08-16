"""Frame manager — the single, thread-safe hub between the camera grab
thread and the AI worker threads.

The grab thread calls :meth:`FrameManager.publish` with each new frame;
AI workers call :meth:`FrameManager.latest` to grab the newest frame
without ever blocking the camera.  If a worker is slower than the camera,
the intermediate frames are naturally *dropped* (overwritten) — exactly
what we want for a real-time assistive pipeline.

Counters:
    published: total frames handed to the manager (from the camera).
    consumed:  total frames actually picked up by workers (<= published).
    dropped:   frames overwritten before a worker read them.
"""
import threading
import time
from typing import Optional, Tuple

import numpy as np


class FrameManager:
    """Latest-frame store with monotonic counters and FPS accounting."""

    def __init__(self, window: float = 1.0) -> None:
        """Create the manager.

        Args:
            window: Rolling window (seconds) used for the FPS estimate.
        """
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_id = 0
        self._published = 0
        self._consumed = 0
        self._window = float(window)
        self._timestamps: list = []

    # ------------------------------------------------------------------
    # Producers (grab thread)
    # ------------------------------------------------------------------

    def publish(self, frame: np.ndarray) -> int:
        """Store the newest frame and return its frame id.

        Dropping is implicit: publishing simply overwrites the previous
        frame, so a slow worker naturally sees the latest frame only.
        """
        with self._lock:
            self._frame_id += 1
            self._frame = frame
            self._published += 1
            now = time.monotonic()
            self._timestamps.append(now)
            self._trim(now)
            return self._frame_id

    # ------------------------------------------------------------------
    # Consumers (AI workers)
    # ------------------------------------------------------------------

    def latest(self) -> Tuple[int, Optional[np.ndarray]]:
        """Return (frame_id, frame) of the newest frame.

        ``frame`` is None if nothing has been published yet.  The frame
        is the *same* array object the grab thread wrote, so consumers
        must not mutate it.
        """
        with self._lock:
            self._consumed += 1
            return self._frame_id, self._frame

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def fps(self) -> float:
        """Frames published per second over the rolling window."""
        with self._lock:
            if len(self._timestamps) < 2:
                return 0.0
            span = self._timestamps[-1] - self._timestamps[0]
            if span <= 0:
                return 0.0
            return (len(self._timestamps) - 1) / span

    def counters(self) -> dict:
        """Snapshot of publish/consume/drop counters."""
        with self._lock:
            return {
                "frames_published": self._published,
                "frames_consumed": self._consumed,
                "frames_dropped": self._published - self._consumed,
            }

    def _trim(self, now: float) -> None:
        cutoff = now - self._window
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.pop(0)