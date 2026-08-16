"""Unit tests for the decision engine (hardware-free)."""
from src.decision.engine import (
    Decision,
    DecisionEngine,
    FrameSummary,
    evaluate,
    _priority_for_cue,
)
from src.detection.detector import DetectionResult
from src.ocr.ocr_engine import OcrResult


def _det(label, box, conf=0.9):
    return DetectionResult(label=label, confidence=conf, box=box)


def _ocr(text):
    return OcrResult(text=text, confidence=0.9, box=(0, 0, 50, 10))


class TestEvaluate:
    def test_sorted_by_priority(self):
        summary = FrameSummary(
            detections=[_det("person", (100, 0, 60, 200)),
                        _det("traffic light", (160, 0, 30, 60))],
            ocr_items=[],
            frame_w=640, frame_h=480,
        )
        decisions = evaluate(summary)
        assert decisions[0].source == "detection"
        assert "Traffic light" in decisions[0].text
        priorities = [d.priority for d in decisions]
        assert priorities == sorted(priorities)

    def test_empty_summary(self):
        assert evaluate(FrameSummary()) == []

    def test_crosswalk_from_ocr_priority(self):
        summary = FrameSummary(
            detections=[], ocr_items=[_ocr("CROSSWALK")],
            frame_w=640, frame_h=480,
        )
        decisions = evaluate(summary)
        assert decisions[0].text == "Crosswalk sign ahead"


class TestDecisionEngine:
    def test_returns_phrase_when_due(self):
        engine = DecisionEngine(cooldown_seconds=1.0)
        summary = FrameSummary(detections=[_det("person", (290, 0, 60, 200))],
                               ocr_items=[], frame_w=640, frame_h=480)
        assert engine.decide(summary, now=0.0) == "Person ahead, about 5 metres"

    def test_cooldown_blocks_repeats(self):
        engine = DecisionEngine(cooldown_seconds=5.0)
        summary = FrameSummary(detections=[_det("person", (100, 0, 60, 200))],
                               ocr_items=[], frame_w=640, frame_h=480)
        assert engine.decide(summary, now=0.0) is not None
        assert engine.decide(summary, now=2.0) is None
        assert engine.decide(summary, now=6.0) is None  # same text

    def test_min_priority_filter(self):
        engine = DecisionEngine(min_priority=1)
        summary = FrameSummary(detections=[_det("person", (100, 0, 60, 200))],
                               ocr_items=[], frame_w=640, frame_h=480)
        assert engine.decide(summary, now=0.0) is None  # person=3 > 1

    def test_different_text_speaks_after_cooldown(self):
        engine = DecisionEngine(cooldown_seconds=5.0)
        s1 = FrameSummary(detections=[_det("person", (290, 0, 60, 200))],
                          ocr_items=[], frame_w=640, frame_h=480)
        s2 = FrameSummary(detections=[_det("car", (0, 0, 40, 80))],
                          ocr_items=[], frame_w=640, frame_h=480)
        assert engine.decide(s1, now=0.0) is not None
        assert engine.decide(s2, now=0.5) is None       # inside cooldown
        assert engine.decide(s2, now=6.0) is not None   # new text, cooldown passed

    def test_reset_clears_state(self):
        engine = DecisionEngine(cooldown_seconds=5.0)
        summary = FrameSummary(detections=[_det("person", (100, 0, 60, 200))],
                               ocr_items=[], frame_w=640, frame_h=480)
        assert engine.decide(summary, now=0.0) is not None
        engine.reset()
        assert engine.decide(summary, now=1.0) is not None

    def test_distance_jitter_does_not_respeak(self):
        # Box size jitters frame-to-frame, so the distance phrase changes
        # ("about 5 metres" -> "about 6 metres").  That is the SAME
        # message and must not be re-spoken.
        engine = DecisionEngine(cooldown_seconds=1.0)
        s1 = FrameSummary(detections=[_det("person", (100, 0, 60, 200))],
                          ocr_items=[], frame_w=640, frame_h=480)
        s2 = FrameSummary(detections=[_det("person", (100, 0, 64, 204))],
                          ocr_items=[], frame_w=640, frame_h=480)
        assert engine.decide(s1, now=0.0) is not None
        assert engine.decide(s2, now=5.0) is None  # cooldown passed but same identity

    def test_identity_change_after_cooldown_respeaks(self):
        engine = DecisionEngine(cooldown_seconds=1.0)
        s_right = FrameSummary(detections=[_det("person", (600, 0, 60, 200))],
                               ocr_items=[], frame_w=640, frame_h=480)
        s_left = FrameSummary(detections=[_det("person", (0, 0, 60, 200))],
                              ocr_items=[], frame_w=640, frame_h=480)
        assert engine.decide(s_right, now=0.0) is not None
        assert engine.decide(s_right, now=5.0) is None      # same message
        assert engine.decide(s_left, now=10.0) is not None  # new direction

    def test_already_spoken_identity_is_skipped(self):
        # The tracking monitor announces "Person ahead, about 5 metres";
        # the decision engine must not narrate the same person again.
        engine = DecisionEngine(cooldown_seconds=0.0)
        summary = FrameSummary(detections=[_det("person", (290, 0, 60, 200))],
                               ocr_items=[], frame_w=640, frame_h=480)
        phrase = engine.decide(summary, now=0.0,
                               already_spoken=["Person ahead, about 5 metres"])
        assert phrase is None

    def test_phone_detection_gets_a_decision(self):
        # cell phone is category "object" (not person/vehicle/signal),
        # but it must still be narrated — with its distance.
        engine = DecisionEngine(cooldown_seconds=1.0)
        summary = FrameSummary(detections=[_det("cell phone", (200, 0, 60, 120))],
                               ocr_items=[], frame_w=640, frame_h=480)
        phrase = engine.decide(summary, now=0.0)
        assert phrase is not None
        assert "Cell phone" in phrase
        assert "metres" in phrase

    def test_no_detections_returns_none(self):
        engine = DecisionEngine()
        assert engine.decide(FrameSummary(), now=0.0) is None

    def test_reads_ocr_text_aloud(self):
        engine = DecisionEngine(cooldown_seconds=0, read_ocr_text=True)
        summary = FrameSummary(detections=[], ocr_items=[_ocr("EXIT SIGN")],
                               frame_w=640, frame_h=480)
        assert engine.decide(summary, now=0.0) == "Text says, EXIT SIGN"

    def test_ocr_reading_off_by_default(self):
        engine = DecisionEngine(cooldown_seconds=0)
        summary = FrameSummary(detections=[], ocr_items=[_ocr("EXIT SIGN")],
                               frame_w=640, frame_h=480)
        assert engine.decide(summary, now=0.0) is None

    def test_toggle_ocr_reading(self):
        engine = DecisionEngine(cooldown_seconds=0, read_ocr_text=True)
        summary = FrameSummary(detections=[], ocr_items=[_ocr("HELLO")],
                               frame_w=640, frame_h=480)
        assert engine.decide(summary, now=0.0) == "Text says, HELLO"
        engine.set_read_ocr_text(False)
        assert engine.decide(summary, now=1.0) is None
        engine.set_read_ocr_text(True)
        assert engine.decide(summary, now=2.0) == "Text says, HELLO"

    def test_ocr_text_truncated(self):
        engine = DecisionEngine(cooldown_seconds=0, read_ocr_text=True,
                                max_ocr_chars=5)
        summary = FrameSummary(detections=[], ocr_items=[_ocr("LONG TEXT")],
                               frame_w=640, frame_h=480)
        assert engine.decide(summary, now=0.0) == "Text says, LONG "


class TestTrackedObjectIntegration:
    """TrackedObject from src.tracking must satisfy the decision engine
    (the assist app passes tracks directly into FrameSummary)."""

    def test_tracked_object_works_with_evaluate(self):
        from src.tracking import TrackedObject

        track = TrackedObject(
            track_id=0, label="car",
            box=(300, 100, 120, 60), confidence=0.9,
        )
        summary = FrameSummary(
            detections=[track], ocr_items=[],
            frame_w=640, frame_h=480,
        )
        decisions = evaluate(summary)
        assert decisions  # car is a vehicle -> has a cue
        assert any("Car" in d.text for d in decisions)

    def test_tracked_object_category_matches_detection(self):
        from src.detection.detector import NAVIGATION_CLASSES
        from src.tracking import TrackedObject

        for label, cat in NAVIGATION_CLASSES.items():
            track = TrackedObject(
                track_id=0, label=label,
                box=(0, 0, 10, 10), confidence=0.9,
            )
            assert track.category == cat, label

    def test_tracked_object_obstacle_detection(self):
        from src.tracking import TrackedObject

        chair = TrackedObject(
            track_id=0, label="chair",
            box=(200, 100, 120, 200), confidence=0.9,
        )
        summary = FrameSummary(
            detections=[chair], ocr_items=[],
            frame_w=640, frame_h=480,
        )
        decisions = evaluate(summary)
        assert any("Obstacle" in d.text for d in decisions)


class TestPriorities:
    def test_priority_mapping(self):
        assert _priority_for_cue("Traffic light ahead") == 0
        assert _priority_for_cue("Stop sign right") == 1
        assert _priority_for_cue("Crosswalk sign ahead") == 2
        assert _priority_for_cue("Person ahead") == 3
        assert _priority_for_cue("Car left") == 4
        assert _priority_for_cue("something else") == 5

    def test_decision_dataclass(self):
        d = Decision(text="hi", priority=1, source="detection")
        assert d.text == "hi" and d.priority == 1
