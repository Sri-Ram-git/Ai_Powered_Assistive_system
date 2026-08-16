"""Label stability tests (Phase 12).

A single spurious detection must not flip the spoken label; a sustained,
decisive change must.  These tests isolate the voting mechanism by using
``class_consistent=False`` so a *single* track receives both labels and
the rolling-window vote decides what is displayed.
"""
from src.detection.detector import DetectionResult
from src.tracking.tracker import IoUTracker


def _det(label, box, conf=0.9):
    return DetectionResult(label=label, confidence=conf, box=box)


def _tracker():
    # class_consistent=False: same-box detections of any class associate
    # with the same track, so only the label vote decides the display.
    return IoUTracker(class_consistent=False,
                      label_vote_window=5, label_vote_ratio=0.6)


class TestLabelVoting:
    def test_single_spurious_detection_does_not_flip_label(self):
        tracker = _tracker()
        tracker.update([_det("person", (100, 100, 50, 150))])
        tracker.update([_det("person", (100, 100, 50, 150))])
        tracker.update([_det("person", (100, 100, 50, 150))])

        # One wrong "chair" detection at the same box.
        tracks = tracker.update([_det("chair", (100, 100, 50, 150))])
        assert tracks[0].label == "person"
        assert len(tracker.all_tracks()) == 1  # same track, not a new one

        # Back to correct detections; still person.
        tracks = tracker.update([_det("person", (100, 100, 50, 150))])
        assert tracks[0].label == "person"

    def test_sustained_change_flips_label(self):
        tracker = _tracker()
        tracker.update([_det("person", (100, 100, 50, 150))])
        tracker.update([_det("person", (100, 100, 50, 150))])

        label = "person"
        for _ in range(4):
            tracks = tracker.update([_det("chair", (100, 100, 50, 150))])
            label = tracks[0].label
        assert label == "chair"

    def test_new_track_uses_initial_label(self):
        tracker = IoUTracker()
        tracks = tracker.update([_det("person", (100, 100, 50, 150))])
        assert tracks[0].label == "person"

    def test_label_never_oscillates_frame_to_frame(self):
        """Alternating person/chair within one track must stay stable."""
        tracker = _tracker()
        tracker.update([_det("person", (100, 100, 50, 150))])
        seen = []
        for label in ("chair", "person", "chair", "person"):
            tracks = tracker.update([_det(label, (100, 100, 50, 150))])
            seen.append(tracks[0].label)
        # No single-frame oscillation: the vote holds the majority.
        flips = sum(1 for a, b in zip(seen, seen[1:]) if a != b)
        assert flips <= 1