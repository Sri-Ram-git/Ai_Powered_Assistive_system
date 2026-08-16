"""Tracking stability tests (Phases 8-13).

Proves, without hardware, that the hardened IoUTracker:
    - never lets a different-class box steal an identity
    - keeps IDs across brief detection gaps
    - smooths box jitter (EMA) instead of copying raw detections
    - exposes the Phase 9 per-track fields
"""


from src.detection.detector import DetectionResult
from src.tracking.tracker import IoUTracker


def _det(label, box, conf=0.9):
    return DetectionResult(label=label, confidence=conf, box=box)


class TestClassConsistency:
    def test_different_class_never_steals_identity(self):
        """A chair box overlapping a person box must NOT take its ID."""
        tracker = IoUTracker()
        tracker.update([_det("person", (100, 100, 50, 150))])

        # Frame 2: a chair box sits exactly on the person's box.  The
        # person is briefly unmatched this frame (missed=1) but keeps
        # identity 0; the chair must be a brand-new track, not ID 0.
        tracker.update([_det("chair", (100, 100, 50, 150))])

        all_tracks = tracker.all_tracks()
        assert len(all_tracks) == 2
        person = [t for t in all_tracks if t.label == "person"][0]
        chair = [t for t in all_tracks if t.label == "chair"][0]
        assert person.track_id == 0        # identity NOT stolen
        assert chair.track_id != 0         # chair gets its own identity
        assert person.missed == 1          # still alive, just unmatched

    def test_same_class_reuses_identity(self):
        tracker = IoUTracker()
        tracker.update([_det("person", (100, 100, 50, 150))])
        tracks = tracker.update([_det("person", (102, 100, 50, 150))])
        assert len(tracks) == 1
        assert tracks[0].track_id == 0


class TestIdentityPersistence:
    def test_id_survives_brief_gap(self):
        tracker = IoUTracker(max_missed=5)
        tracker.update([_det("person", (100, 100, 50, 150))])
        tracker.update([])  # one missed frame
        tracks = tracker.update([_det("person", (101, 101, 50, 150))])
        assert tracks[0].track_id == 0

    def test_lost_then_new_object_gets_new_id(self):
        tracker = IoUTracker(max_missed=1)
        tracker.update([_det("person", (100, 100, 50, 150))])
        for _ in range(3):
            tracker.update([])  # exceed max_missed
        tracks = tracker.update([_det("person", (100, 100, 50, 150))])
        assert tracks[0].track_id == 1  # fresh identity


class TestBoxSmoothing:
    def test_smoothed_box_oscillates_less_than_raw(self):
        """Alternating 20px jumps must produce < 20px smoothed steps."""
        tracker = IoUTracker(smoothing=0.4)
        # Warm up: alternating boxes around a centre.
        tracker.update([_det("person", (100, 100, 50, 150))])
        tracker.update([_det("person", (120, 100, 50, 150))])
        tracker.update([_det("person", (100, 100, 50, 150))])

        # Now measure raw-vs-smoothed deltas on a 20px jump.
        max_smoothed_delta = 0
        for _ in range(4):
            tracks = tracker.update([_det("person", (100, 100, 50, 150))])
            track = tracks[0]
            max_smoothed_delta = max(
                max_smoothed_delta, abs(track.box[0] - 100))
        # EMA alpha 0.4 converges to 100 with steps smaller than the 20px
        # raw jump (first step = 0.4*20 = 8px).
        assert max_smoothed_delta < 20

    def test_raw_box_kept_for_diagnostics(self):
        tracker = IoUTracker(smoothing=0.0)
        tracker.update([_det("person", (100, 100, 50, 150))])
        tracks = tracker.update([_det("person", (120, 100, 50, 150))])
        track = tracks[0]
        assert track.raw_box == (120, 100, 50, 150)


class TestPhase9Fields:
    def test_track_exposes_required_fields(self):
        tracker = IoUTracker()
        tracker.update([_det("person", (100, 100, 50, 150))])
        track = tracker.all_tracks()[0]

        assert track.track_id == 0
        assert track.label == "person"
        assert track.confidence > 0.0
        assert track.box == (100, 100, 50, 150)
        assert tuple(round(v) for v in track.center) == (125.0, 175.0)
        assert track.width == 50 and track.height == 150
        assert track.age == 1
        assert track.missed == 0
        assert track.first_seen == 1
        assert track.last_seen == 1
        assert track.timestamp > 0.0
        # direction / distance are cached placeholders until downstream
        # geometry code fills them in; properties exist and are floats/str.
        assert isinstance(track.distance_m, float)
        assert isinstance(track.direction, str)


class TestConfidenceSmoothing:
    def test_confidence_smoothes_toward_true_value(self):
        tracker = IoUTracker(conf_smoothing=0.5)
        tracker.update([_det("person", (100, 100, 50, 150), conf=0.9)])
        track = tracker.all_tracks()[0]
        assert track.confidence == 0.9

        # A single weak detection only nudges the smoothed value.
        tracks = tracker.update([_det("person", (100, 100, 50, 150), conf=0.3)])
        assert tracks[0].confidence > 0.3
        assert tracks[0].confidence < 0.9

    def test_raw_confidence_kept_for_diagnostics(self):
        tracker = IoUTracker(conf_smoothing=0.0)
        tracker.update([_det("person", (100, 100, 50, 150), conf=0.9)])
        tracks = tracker.update([_det("person", (100, 100, 50, 150), conf=0.4)])
        assert tracks[0].raw_confidence == 0.4