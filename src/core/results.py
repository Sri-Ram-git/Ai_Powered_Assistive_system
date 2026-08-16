"""Latest-results container shared between the async pipeline workers and
the UI/API consumers.

Each AI stage (YOLO, OCR, depth, STT) publishes its newest result here;
the grab/annotate thread and the dashboard read the *latest* values.  A
slow stage therefore never blocks the fast ones — consumers always see
the most recent available result.
"""
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional


@dataclass
class LatestResults:
    """Thread-safe holder for the most recent per-stage results."""

    detections: List[Any] = field(default_factory=list)
    tracks: List[Any] = field(default_factory=list)
    ocr_items: List[Any] = field(default_factory=list)
    depth_map: Optional[Any] = None
    guidance: List[str] = field(default_factory=list)
    stt_command: Optional[str] = None

    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    # Per-stage latency in milliseconds (0.0 = not measured this tick).
    latencies: Dict[str, float] = field(default_factory=dict, repr=False)

    def update(self, **kwargs: Any) -> None:
        """Atomically replace the given fields."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def snapshot(self) -> Dict[str, Any]:
        """Return a copy safe to pass across threads."""
        with self._lock:
            return {
                "detections": list(self.detections),
                "tracks": list(self.tracks),
                "ocr_items": list(self.ocr_items),
                "depth_map": self.depth_map,
                "guidance": list(self.guidance),
                "stt_command": self.stt_command,
                "latencies": dict(self.latencies),
            }