"""Object Tracking module.

tracker: IoUTracker, TrackedObject — frame-to-frame identity by IoU
         association (no ML trackers needed).
monitor: TrackingMonitor — emits guidance phrases when objects appear,
         move closer/farther, or leave the view.
"""
from src.tracking.monitor import TrackingMonitor
from src.tracking.tracker import (
    IoUTracker,
    TrackedObject,
)

__all__ = [
    "IoUTracker",
    "TrackedObject",
    "TrackingMonitor",
]
