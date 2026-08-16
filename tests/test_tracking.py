"""Tests for the IoU tracker and tracking monitor (hardware-free)."""


from src.detection.detector import DetectionResult
from src.tracking.monitor import TrackingMonitor
from src.tracking.tracker import IoUTracker, TrackedObject


def _det(label, box, conf=0.9):
    return DetectionResult(label=label, confidence=conf, box=box)


def _track(track_id, label, box, conf=0.9):
    x, y, w, h = box
    return TrackedObject(
        track_id=track_id,
        label=label,
        box=box,
        confidence=conf,
        age=1,
        missed=0,
        center=(x + w / 2.0, y + h / 2.0),
        first_seen=0,
    )


class TestIoUTracker:
    """IoU association, persistence, and staleness."""

    def test_new_track_created_on_first_frame(self):
        tracker = IoUTracker()
        det = _det("person", (100, 100, 50, 150))
        tracks = tracker.update([det])

        assert len(tracks) == 1
        assert tracks[0].track_id == 0
        assert tracks[0].label == "person"

    def test_same_object_keeps_id_across_frames(self):
        tracker = IoUTracker()
        tracker.update([_det("person", (100, 100, 50, 150))])

        # Small motion -> high IoU -> same track id.
        tracks = tracker.update([_det("person", (102, 100, 50, 150))])
        assert len(tracks) == 1
        assert tracks[0].track_id == 0
        assert tracks[0].age == 2

    def test_multiple_objects_get_distinct_ids(self):
        tracker = IoUTracker()
        tracks = tracker.update([
            _det("person", (50, 100, 50, 150)),
            _det("car", (300, 200, 120, 80)),
        ])
        ids = {t.track_id for t in tracks}
        assert len(ids) == 2

    def test_two_objects_cross_without_swapping_ids(self):
        tracker = IoUTracker()
        tracker.update([
            _det("person", (50, 100, 50, 150)),
            _det("person", (300, 100, 50, 150)),
        ])
        # Frame 2: both shifted right by 10px; should keep identities.
        tracks = tracker.update([
            _det("person", (60, 100, 50, 150)),
            _det("person", (310, 100, 50, 150)),
        ])
        assert {t.track_id for t in tracks} == {0, 1}

    def test_track_marks_missing_and_survives_brief_gap(self):
        tracker = IoUTracker(max_missed=5)
        tracker.update([_det("person", (100, 100, 50, 150))])

        # One frame without detection: track is missing but still alive.
        tracks = tracker.update([])
        assert tracks == []

        t = tracker.all_tracks()[0]
        assert t.missed == 1
        assert t.alive is False

        # Detection returns -> same id restored.
        tracks = tracker.update([_det("person", (100, 100, 50, 150))])
        assert tracks[0].track_id == 0

    def test_track_dropped_after_max_missed(self):
        tracker = IoUTracker(max_missed=2)
        tracker.update([_det("person", (100, 100, 50, 150))])

        for _ in range(3):
            tracker.update([])
        assert tracker.all_tracks() == []

    def test_reset_clears_state_and_restarts_ids(self):
        tracker = IoUTracker()
        tracker.update([_det("person", (100, 100, 50, 150))])
        tracker.reset()
        assert tracker.all_tracks() == []

        tracks = tracker.update([_det("person", (100, 100, 50, 150))])
        assert tracks[0].track_id == 0

    def test_high_confidence_detection_matched_first(self):
        tracker = IoUTracker()
        # Two overlapping boxes with different confidences.
        tracker.update([_det("person", (100, 100, 50, 150), conf=0.95)])

        tracks = tracker.update([
            _det("person", (100, 100, 50, 150), conf=0.5),
            _det("person", (105, 105, 50, 150), conf=0.9),
        ])
        # Both should associate (only one existing track though).
        assert len(tracks) == 2

    def test_area_property(self):
        track = _track(0, "person", (10, 10, 40, 60))
        assert track.area == 40 * 60
        assert track.alive is True


class TestTrackingMonitor:
    """Guidance phrases from tracked objects."""

    def test_new_object_is_announced(self):
        monitor = TrackingMonitor()
        track = _track(0, "person", (100, 100, 50, 150))
        phrases = monitor.events([track], frame_w=640, frame_h=480)

        assert len(phrases) == 1
        assert "person" in phrases[0].lower()
        assert "metres" in phrases[0]

    def test_static_object_is_not_reannounced(self):
        monitor = TrackingMonitor()
        track = _track(0, "person", (100, 100, 50, 150))
        monitor.events([track], 640, 480, now=0.0)

        # No movement, same track -> no new phrases.
        phrases = monitor.events([track], 640, 480, now=1.0)
        assert phrases == []

    def test_distance_change_triggers_reannounce(self):
        monitor = TrackingMonitor(distance_change_metres=1.0,
                                  min_announce_interval=0.0)
        track = _track(0, "person", (100, 100, 50, 150))
        monitor.events([track], 640, 480, now=0.0)

        # Box much taller -> distance shrank by > 1 m.
        closer = _track(0, "person", (100, 200, 50, 260))
        phrases = monitor.events([closer], 640, 480, now=1.0)
        assert len(phrases) == 1
        assert "now" in phrases[0]

    def test_min_announce_interval_throttles(self):
        monitor = TrackingMonitor(distance_change_metres=1.0,
                                  min_announce_interval=5.0)
        track = _track(0, "person", (100, 100, 50, 150))
        monitor.events([track], 640, 480, now=0.0)

        # Distance changed but too soon after the last announcement.
        closer = _track(0, "person", (100, 200, 50, 260))
        phrases = monitor.events([closer], 640, 480, now=1.0)
        assert phrases == []

        # A further change after the interval elapsed is announced.
        much_closer = _track(0, "person", (100, 300, 50, 420))
        phrases = monitor.events([much_closer], 640, 480, now=6.0)
        assert len(phrases) == 1
        assert "now" in phrases[0]

    def test_lost_track_is_forgotten(self):
        monitor = TrackingMonitor()
        track = _track(0, "person", (100, 100, 50, 150))
        monitor.events([track], 640, 480, now=0.0)

        # Track gone -> its memory is removed; a new identical object
        # later is treated as new.
        monitor.events([], 640, 480, now=1.0)
        phrases = monitor.events([_track(5, "person", (100, 100, 50, 150))],
                                 640, 480, now=2.0)
        assert len(phrases) == 1
        assert "person" in phrases[0].lower()

    def test_direction_change_reannounces(self):
        monitor = TrackingMonitor(distance_change_metres=1.0,
                                  min_announce_interval=0.0)
        track = _track(0, "person", (500, 100, 50, 150))  # right side
        monitor.events([track], 640, 480, now=0.0)

        # Move to the left side of the frame -> direction changed.
        left = _track(0, "person", (30, 100, 50, 150))
        phrases = monitor.events([left], 640, 480, now=1.0)
        assert len(phrases) == 1
        assert "left" in phrases[0]

    def test_reset_forgets_everything(self):
        monitor = TrackingMonitor()
        track = _track(0, "person", (100, 100, 50, 150))
        monitor.events([track], 640, 480, now=0.0)
        monitor.reset()

        phrases = monitor.events([track], 640, 480, now=1.0)
        assert len(phrases) == 1  # treated as new again
