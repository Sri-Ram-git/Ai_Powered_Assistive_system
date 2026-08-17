"""Unit tests for the object-aware OCR subsystem.

Covers the pure building blocks (policy, ROI, text presence, validation,
variant selection, track store, trigger, target ranking) and the async
ObjectOcrWorker.  Everything here runs without a camera, a YOLO model, or
RapidOCR.
"""
import time
from types import SimpleNamespace

import cv2
import numpy as np

from src.ocr.object_ocr import (
    ObjectOcrResult,
    OcrTrigger,
    TrackOcrStore,
    best_variant,
    combine_results,
    is_garbage,
    normalize_text,
    rank_targets,
    run_variants,
    validate_text,
)
from src.ocr.object_worker import ObjectOcrWorker
from src.ocr.ocr_engine import OcrResult
from src.ocr.policy import OcrPolicy
from src.ocr.roi import extract_roi, smart_upscale
from src.ocr.text_presence import has_text, text_presence_score


# ----------------------------------------------------------------------
# Policy
# ----------------------------------------------------------------------

class TestOcrPolicy:
    def test_defaults_are_coco_only(self):
        p = OcrPolicy.defaults()
        supported = {"book", "bottle", "laptop", "cell phone", "tv",
                     "stop sign", "cup", "backpack", "handbag", "suitcase",
                     "keyboard", "remote", "vase", "clock", "person"}
        assert set(p.high) <= supported
        assert p.is_eligible("book")
        assert p.is_eligible("cup")
        assert not p.is_eligible("person")

    def test_from_yaml_loads_tiers(self):
        p = OcrPolicy.from_yaml("configs/ocr_policy.yaml")
        assert "book" in p.high
        assert "person" in p.disabled
        assert not p.is_eligible("person")
        assert p.is_eligible("bottle")
        assert p.tier_for("book") == "high"
        assert p.tier_for("cup") == "medium"
        assert p.tier_for("tv") == "high"

    def test_unknown_and_invalid_labels_ignored(self):
        p = OcrPolicy.from_yaml(None)
        assert p.tier_for("dragon") == p.default_tier
        assert p.rank("dragon") == p.rank(p.default_tier)

    def test_missing_file_falls_back_to_defaults(self):
        p = OcrPolicy.from_yaml("does_not_exist.yaml")
        assert p.high == OcrPolicy.defaults().high

    def test_rank_orders_tiers(self):
        p = OcrPolicy.defaults()
        assert p.rank("book") > p.rank("cup") > p.rank("person")


# ----------------------------------------------------------------------
# ROI extraction + smart upscaling
# ----------------------------------------------------------------------

class TestRoi:
    def _frame(self, w=640, h=480):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_extract_pads_and_clamps(self):
        frame = self._frame()
        roi = extract_roi(frame, (100, 100, 80, 120), padding=0.1)
        assert roi is not None
        # pad_x = 8, pad_y = 12
        assert roi.box == (92, 88, 188, 232)
        assert roi.image.shape[0] > 0 and roi.image.shape[1] > 0

    def test_clamps_to_frame_bounds(self):
        frame = self._frame()
        roi = extract_roi(frame, (620, 460, 40, 40), padding=0.1)
        assert roi.box[2] <= 640 and roi.box[3] <= 480

    def test_too_small_roi_rejected(self):
        frame = self._frame()
        assert extract_roi(frame, (10, 10, 5, 5), min_w=24, min_h=12) is None

    def test_degenerate_box_rejected(self):
        frame = self._frame()
        assert extract_roi(frame, (10, 10, 0, 10)) is None
        assert extract_roi(frame, (10, 10, 10, -5)) is None

    def test_smart_upscale_factors(self):
        small = np.zeros((20, 40, 3), dtype=np.uint8)
        img, scale = smart_upscale(small)
        assert scale == 3.0
        assert img.shape[0] == 60

        mid = np.zeros((40, 80, 3), dtype=np.uint8)
        img, scale = smart_upscale(mid)
        assert scale == 2.0
        assert img.shape[1] == 160

        big = np.zeros((100, 200, 3), dtype=np.uint8)
        img, scale = smart_upscale(big)
        assert scale == 1.0
        assert img is big

    def test_max_upscale_cap(self):
        small = np.zeros((20, 20, 3), dtype=np.uint8)
        _, scale = smart_upscale(small, max_scale=2.0)
        assert scale == 2.0

    def test_extract_uses_upscaled_image(self):
        frame = self._frame()
        roi = extract_roi(frame, (100, 100, 20, 20))
        assert roi is not None
        assert roi.scale > 1.0
        assert roi.image.shape[0] > roi.box[3] - roi.box[1]


# ----------------------------------------------------------------------
# Text presence
# ----------------------------------------------------------------------

class TestTextPresence:
    def test_blank_is_not_text(self):
        blank = np.full((100, 200), 128, dtype=np.uint8)
        assert text_presence_score(blank) < 0.35
        assert not has_text(blank)

    def test_synthetic_text_is_text(self):
        img = np.full((100, 300), 0, dtype=np.uint8)
        cv2.putText(img, "HELLO WORLD 123", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        score = text_presence_score(img)
        assert score >= 0.35
        assert has_text(img)

    def test_plain_rectangle_is_not_text(self):
        img = np.full((100, 300), 0, dtype=np.uint8)
        cv2.rectangle(img, (30, 30), (270, 70), (255, 255, 255), 2)
        assert text_presence_score(img) < 0.35


# ----------------------------------------------------------------------
# Text validation
# ----------------------------------------------------------------------

class TestValidation:
    def test_normalize(self):
        assert normalize_text("  HELLO   world  ") == "HELLO world"

    def test_garbage_detection(self):
        assert is_garbage("")
        assert is_garbage("!!!")
        assert is_garbage("AAAAAAA")
        assert not is_garbage("HELLO")
        assert not is_garbage("COCA COLA")

    def test_validate(self):
        assert validate_text("  hello  ") == "hello"
        assert validate_text("@@@") is None
        assert validate_text("a", min_chars=2) is None


# ----------------------------------------------------------------------
# Variant selection / OCR result combining
# ----------------------------------------------------------------------

def _res(text, conf, box=(0, 0, 10, 10)):
    return OcrResult(text=text, confidence=conf, box=box)


class TestVariants:
    def test_best_variant_picks_highest_quality(self):
        items = [
            ("none", [_res("AB", 0.5)], 5.0),
            ("contrast", [_res("ABCD", 0.9)], 5.0),
        ]
        name, results = best_variant(items)
        assert name == "contrast"
        assert results[0].text == "ABCD"

    def test_best_variant_skips_empty(self):
        items = [("none", [], 1.0), ("contrast", [_res("X", 0.8)], 1.0)]
        name, results = best_variant(items)
        assert name == "contrast"

    def test_combine_results(self):
        text, conf = combine_results([_res("HELLO", 0.9), _res("WORLD", 0.7)])
        assert text == "HELLO WORLD"
        assert 0.7 < conf <= 0.9
        assert combine_results([]) == ("", 0.0)

    def test_run_variants_uses_engine(self):
        class _Engine:
            def __init__(self):
                self.calls = 0

            def read_text(self, image):
                self.calls += 1
                return [_res("HELLO WORLD", 0.95)]

        engine = _Engine()
        img = np.full((100, 300, 3), 200, dtype=np.uint8)
        variant, items, latency = run_variants(engine, img, ["none"])
        assert variant == "none"
        assert items[0].text == "HELLO WORLD"
        assert engine.calls == 1

    def test_run_variants_stops_early_on_high_conf(self):
        class _Engine:
            def __init__(self):
                self.calls = 0

            def read_text(self, image):
                self.calls += 1
                return [_res("GOOD", 0.99)]

        engine = _Engine()
        img = np.full((100, 300, 3), 200, dtype=np.uint8)
        run_variants(engine, img, ["none", "contrast", "adaptive"])
        assert engine.calls == 1  # short-circuited after first


# ----------------------------------------------------------------------
# Track OCR store (voting + expiry + history)
# ----------------------------------------------------------------------

def _obj_result(track_id, text, conf=0.9, ts=None):
    return ObjectOcrResult(
        track_id=track_id, label="bottle", text=text, confidence=conf,
        roi_box=(10, 10, 110, 130), timestamp=ts or time.time(),
        latency_ms=12.0, status="ok",
    )


class TestTrackOcrStore:
    def test_first_read_is_adopted_immediately(self):
        store = TrackOcrStore()
        entry = store.update(_obj_result(1, "COCA COLA"))
        assert entry is not None
        assert store.for_track(1).text == "COCA COLA"
        assert store.for_track(1).stable

    def test_noise_requires_two_consecutive_votes(self):
        store = TrackOcrStore()
        store.update(_obj_result(1, "COCA COLA"))
        # A single different read must NOT replace the stable text.
        store.update(_obj_result(1, "ABC123"))
        assert store.for_track(1).text == "COCA COLA"
        # A second identical read adopts it.
        store.update(_obj_result(1, "ABC123"))
        assert store.for_track(1).text == "ABC123"

    def test_garbage_reads_never_adopt(self):
        store = TrackOcrStore()
        store.update(_obj_result(1, "COCA COLA"))
        store.update(_obj_result(1, "!!!"))
        assert store.for_track(1).text == "COCA COLA"

    def test_latest_and_history(self):
        store = TrackOcrStore()
        store.update(_obj_result(1, "ONE", ts=100.0))
        store.update(_obj_result(2, "TWO", ts=200.0))
        assert store.latest().track_id == 2
        hist = store.history()
        assert [h.track_id for h in hist] == [2, 1]

    def test_expire_drops_dead_tracks(self):
        store = TrackOcrStore()
        store.update(_obj_result(1, "ONE", ts=100.0))
        store.update(_obj_result(2, "TWO", ts=200.0))
        store.expire(max_age=50.0, alive_track_ids=[1], now=400.0)
        assert store.for_track(1) is not None
        assert store.for_track(2) is None

    def test_texts_and_clear(self):
        store = TrackOcrStore()
        store.update(_obj_result(1, "HELLO"))
        assert store.texts() == [(1, "bottle", "HELLO", 0.9)]
        store.clear()
        assert store.texts() == []


# ----------------------------------------------------------------------
# Trigger policy
# ----------------------------------------------------------------------

class TestOcrTrigger:
    def test_first_seen_is_new(self):
        trig = OcrTrigger()
        assert trig.decide(1, "bottle", (0, 0, 10, 10), now=0.0) == "new"

    def test_cooldown_blocks_retrigger(self):
        trig = OcrTrigger(cooldown_s=3.0)
        trig.decide(1, "bottle", (0, 0, 10, 10), now=0.0)
        assert trig.decide(1, "bottle", (0, 0, 10, 10), now=1.0) is None

    def test_moved_triggers_after_cooldown(self):
        trig = OcrTrigger(cooldown_s=3.0, move_px=40)
        trig.decide(1, "bottle", (0, 0, 10, 10), now=0.0)
        assert trig.decide(1, "bottle", (100, 0, 10, 10), now=4.0) == "moved"

    def test_stale_triggers_when_unchanged(self):
        trig = OcrTrigger(cooldown_s=3.0, stale_after_s=5.0, move_px=40)
        trig.decide(1, "bottle", (0, 0, 10, 10), now=0.0)
        assert trig.decide(1, "bottle", (0, 0, 10, 10), now=10.0) == "stale"

    def test_touched_refreshes_clock(self):
        trig = OcrTrigger(cooldown_s=3.0)
        trig.decide(1, "bottle", (0, 0, 10, 10), now=0.0)
        trig.touched(1, now=2.0)
        assert trig.decide(1, "bottle", (0, 0, 10, 10), now=3.0) is None

    def test_prune_and_reset(self):
        trig = OcrTrigger()
        trig.decide(1, "bottle", (0, 0, 10, 10), now=0.0)
        trig.decide(2, "cup", (0, 0, 10, 10), now=0.0)
        trig.prune([1])
        assert 2 not in trig._last
        trig.reset()
        assert trig._last == {}


# ----------------------------------------------------------------------
# Target ranking
# ----------------------------------------------------------------------

class TestRankTargets:
    def _track(self, tid, label, box, conf):
        return SimpleNamespace(
            track_id=tid, label=label, box=box, confidence=conf,
            area=box[2] * box[3],
        )

    def test_orders_by_tier_then_area_then_conf(self):
        policy = OcrPolicy.defaults()
        book = self._track(1, "book", (0, 0, 50, 50), 0.9)    # high, small
        bottle = self._track(2, "bottle", (0, 0, 200, 200), 0.5)  # high, big
        cup = self._track(3, "cup", (0, 0, 100, 100), 0.9)    # medium
        ranked = rank_targets([cup, bottle, book], policy.rank)
        assert [t.track_id for t in ranked] == [2, 1, 3]


# ----------------------------------------------------------------------
# Object OCR worker
# ----------------------------------------------------------------------

class _FakeEngine:
    """Deterministic engine returning fixed lines / delays."""

    def __init__(self, lines=None, delay=0.0):
        self.lines = (list(lines) if lines is not None
                      else [_res("HELLO WORLD", 0.95)])
        self.delay = delay
        self.calls = 0

    def read_text(self, image):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return list(self.lines)


def _text_like_image():
    img = np.full((100, 300, 3), 200, dtype=np.uint8)
    cv2.putText(img, "SOME TEXT HERE", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    return img


class TestObjectOcrWorker:
    def test_ok_result_via_callback(self):
        engine = _FakeEngine()
        out = []
        worker = ObjectOcrWorker(engine, text_presence=False,
                                 on_result=out.append)
        worker.start()
        try:
            worker.submit(_text_like_image(), track_id=7, label="bottle")
            worker.join(5.0)
            assert len(out) == 1
            result = out[0]
            assert result.status == "ok"
            assert result.text == "HELLO WORLD"
            assert result.track_id == 7
            assert result.label == "bottle"
            assert result.confidence > 0.5
        finally:
            worker.stop()

    def test_text_presence_gate_skips_engine(self):
        engine = _FakeEngine()
        worker = ObjectOcrWorker(engine, text_presence=True)
        worker.start()
        try:
            blank = np.full((100, 200, 3), 128, dtype=np.uint8)
            worker.submit(blank, track_id=1, label="bottle")
            worker.join(5.0)
            assert engine.calls == 0
            assert worker.latest().status == "no_text"
        finally:
            worker.stop()

    def test_timeout_marks_result(self):
        engine = _FakeEngine(delay=0.05)
        worker = ObjectOcrWorker(engine, text_presence=False, timeout_ms=10)
        worker.start()
        try:
            worker.submit(_text_like_image(), track_id=1, label="bottle")
            worker.join(5.0)
            result = worker.latest()
            assert result.status == "timeout"
            assert worker.stats()["timeouts"] >= 1
        finally:
            worker.stop()

    def test_newest_request_replaces_pending(self):
        engine = _FakeEngine()
        worker = ObjectOcrWorker(engine, text_presence=False)
        # Before starting: two submissions hit the single pending slot.
        worker.submit(_text_like_image(), track_id=1, label="bottle")
        worker.submit(_text_like_image(), track_id=2, label="bottle")
        assert worker.stats()["replaced"] == 1
        worker.start()
        try:
            worker.join(5.0)
            assert worker.runs == 1
            assert worker.latest().track_id == 2
        finally:
            worker.stop()

    def test_empty_result_when_no_text(self):
        engine = _FakeEngine(lines=[])
        worker = ObjectOcrWorker(engine, text_presence=False)
        worker.start()
        try:
            worker.submit(_text_like_image(), track_id=1, label="bottle")
            worker.join(5.0)
            assert worker.latest().status == "empty"
        finally:
            worker.stop()

    def test_clear_drops_pending_and_latest(self):
        worker = ObjectOcrWorker(_FakeEngine(), text_presence=False)
        worker.submit(_text_like_image(), track_id=1)
        worker.clear()
        assert worker.latest() is None
        worker.start()
        try:
            worker.join(1.0)
            assert worker.runs == 0
        finally:
            worker.stop()

    def test_stats_snapshot(self):
        engine = _FakeEngine()
        worker = ObjectOcrWorker(engine, text_presence=False)
        stats = worker.stats()
        assert stats["runs"] == 0 and stats["replaced"] == 0