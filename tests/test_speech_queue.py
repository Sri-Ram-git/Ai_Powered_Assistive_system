"""Speech queue tests (Phases 18-20).

Proves the prioritised, deduplicated, rate-limited bridge behaves:
    - CRITICAL always speaks before NORMAL
    - identical phrases are rejected within the dedup window
    - the rate limit prevents chatter
    - reset clears the dedup memory
"""
import time

from src.audio.speech_queue import SpeechQueue, SpeechTier


class _Recorder:
    def __init__(self):
        self.spoken = []
        self.call_times = []

    def speak(self, text):
        self.spoken.append(text)
        self.call_times.append(time.monotonic())


class TestPriority:
    def test_critical_speaks_before_normal(self):
        recorder = _Recorder()
        queue = SpeechQueue(recorder, min_interval=0.0, dedupe_window=0.0)
        queue.start()
        try:
            queue.enqueue("person ahead", SpeechTier.NORMAL)
            queue.enqueue("car approaching stop", SpeechTier.CRITICAL)
            time.sleep(0.5)
            assert recorder.spoken[0] == "car approaching stop"
            assert recorder.spoken[1] == "person ahead"
        finally:
            queue.shutdown()


class TestDeduplication:
    def test_identical_phrase_rejected_within_window(self):
        recorder = _Recorder()
        queue = SpeechQueue(recorder, min_interval=0.0, dedupe_window=60.0)
        assert queue.enqueue("person ahead, about 3 metres", SpeechTier.NORMAL)
        assert not queue.enqueue("person ahead, about 3 metres",
                                 SpeechTier.NORMAL)

    def test_distance_jitter_does_not_duplicate(self):
        """Same object, different estimated distance -> same identity."""
        recorder = _Recorder()
        queue = SpeechQueue(recorder, min_interval=0.0, dedupe_window=60.0)
        assert queue.enqueue("Person ahead, about 3 metres")
        # Distance suffix stripped by cue_identity -> duplicate rejected.
        assert not queue.enqueue("person ahead, about 4 metres")

    def test_different_phrase_accepted(self):
        recorder = _Recorder()
        queue = SpeechQueue(recorder, min_interval=0.0, dedupe_window=60.0)
        assert queue.enqueue("person ahead")
        assert queue.enqueue("chair on the left")


class TestRateLimit:
    def test_worker_respects_min_interval(self):
        recorder = _Recorder()
        queue = SpeechQueue(recorder, min_interval=0.6, dedupe_window=0.0)
        queue.start()
        try:
            queue.enqueue("one", SpeechTier.CRITICAL)
            queue.enqueue("two", SpeechTier.CRITICAL)
            time.sleep(1.5)
            assert len(recorder.spoken) >= 2
            gap = recorder.call_times[1] - recorder.call_times[0]
            assert gap >= 0.5  # ~0.6s apart
        finally:
            queue.shutdown()


class TestMisc:
    def test_empty_text_rejected(self):
        queue = SpeechQueue(_Recorder())
        assert not queue.enqueue("")
        assert not queue.enqueue("   ")

    def test_reset_clears_dedup_memory(self):
        recorder = _Recorder()
        queue = SpeechQueue(recorder, min_interval=0.0, dedupe_window=60.0)
        assert queue.enqueue("person ahead")
        assert not queue.enqueue("person ahead")
        queue.reset()
        assert queue.enqueue("person ahead")  # accepted again

    def test_pending_count_reflects_backlog(self):
        queue = SpeechQueue(_Recorder(), min_interval=99.0, dedupe_window=0.0)
        queue.enqueue("one")
        queue.enqueue("two")
        assert queue.pending_count() == 2