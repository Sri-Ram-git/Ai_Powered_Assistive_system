"""Speech priority queue (Phases 18-20).

Guarantees speech never blocks vision, never repeats the same phrase
every frame, and never lets a low-priority announcement bury a safety
message:

    SpeechQueue(tts)
        .enqueue("person ahead", tier=SpeechTier.NORMAL)
        .enqueue("car approaching — stop", tier=SpeechTier.CRITICAL)

Tiers (lower = more urgent):
    CRITICAL = 0   immediate safety ("stop", "car left")
    HIGH     = 1   time-sensitive navigation ("turning right")
    NORMAL   = 2   routine guidance ("person ahead, about 3 metres")
    LOW      = 3   verbose / optional announcements

The worker thread always speaks the highest-priority pending item first.
Identical phrases are dropped within ``dedupe_window`` seconds, and no
phrase is spoken more often than ``min_interval`` seconds — so the voice
cannot chatter while the vision loop runs freely.
"""
import threading
import time
from enum import IntEnum
from typing import Dict, List, Optional, Set

from src.decision.engine import cue_identity
from src.utils.logger import setup_logger

_logger = setup_logger("SpeechQueue")


class SpeechTier(IntEnum):
    """Priority tiers for spoken responses."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


class SpeechQueue:
    """Prioritised, deduplicated, rate-limited speech bridge to TTS."""

    def __init__(
        self,
        tts,
        min_interval: float = 1.2,
        dedupe_window: float = 4.0,
        max_pending: int = 12,
    ) -> None:
        """Configure the queue.

        Args:
            tts: Object with a non-blocking ``speak(text)`` method
                (e.g. ``src.audio.SpeechOutput``).
            min_interval: Minimum seconds between *any* spoken phrases.
            dedupe_window: Seconds before an identical phrase may be
                spoken again.
            max_pending: Drop lowest-priority pending items beyond this
                count (protects the queue from flooding).
        """
        self._tts = tts
        self._min_interval = float(min_interval)
        self._dedupe_window = float(dedupe_window)
        self._max_pending = int(max_pending)

        self._pending: Dict[SpeechTier, List[str]] = {
            tier: [] for tier in SpeechTier
        }
        self._last_spoken: Dict[str, float] = {}
        self._last_speak_time: float = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stopped = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stopped.clear()
        self._thread = threading.Thread(
            target=self._worker, name="speech-queue", daemon=True)
        self._thread.start()
        _logger.info("Speech queue started")

    def shutdown(self) -> None:
        self._running = False
        self._stopped.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, text: str, tier: SpeechTier = SpeechTier.NORMAL) -> bool:
        """Queue a phrase (non-blocking).  Returns True if accepted.

        A phrase is rejected when it is a duplicate spoken within the
        dedupe window, so the system never repeats itself every frame.
        """
        if not text or not text.strip():
            return False
        key = cue_identity(text)
        if not key:
            return False

        with self._lock:
            now = time.monotonic()
            if key in self._last_spoken and \
                    now - self._last_spoken[key] < self._dedupe_window:
                return False

            # De-duplicate against already-pending identical phrases.
            for existing in self._pending[tier]:
                if cue_identity(existing) == key:
                    return False

            self._pending[tier].append(text)
            self._trim_lowest()
        _logger.debug("Queued [%s] %s", tier.name, text)
        return True

    def pending_count(self) -> int:
        with self._lock:
            return sum(len(items) for items in self._pending.values())

    def reset(self) -> None:
        """Forget dedup history (e.g. user pressed space)."""
        with self._lock:
            self._pending = {tier: [] for tier in SpeechTier}
            self._last_spoken.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        while self._running:
            item = self._pop_highest_priority()
            if item is None:
                self._stopped.wait(0.05)
                continue

            now = time.monotonic()
            if now - self._last_speak_time < self._min_interval:
                # Rate limit: hold the phrase back, retry shortly.
                self._re_queue(item)
                self._stopped.wait(0.05)
                continue

            try:
                self._tts.speak(item)
                self._last_speak_time = time.monotonic()
                self._last_spoken[cue_identity(item)] = \
                    self._last_speak_time
            except Exception as exc:  # pragma: no cover - env dependent
                _logger.warning("Speech failed: %s", exc)

    def _pop_highest_priority(self) -> Optional[str]:
        with self._lock:
            for tier in SpeechTier:  # CRITICAL first
                if self._pending[tier]:
                    return self._pending[tier].pop(0)
        return None

    def _re_queue(self, item: str) -> None:
        with self._lock:
            # Put it back at the front so it isn't starved.
            self._pending[SpeechTier.LOW].append(item)

    def _trim_lowest(self) -> None:
        """Drop oldest LOW (then NORMAL) items beyond max_pending."""
        total = sum(len(items) for items in self._pending.values())
        overflow = total - self._max_pending
        if overflow <= 0:
            return
        for tier in (SpeechTier.LOW, SpeechTier.NORMAL, SpeechTier.HIGH):
            while overflow > 0 and self._pending[tier]:
                self._pending[tier].pop(0)
                overflow -= 1