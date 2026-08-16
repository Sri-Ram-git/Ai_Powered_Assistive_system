"""Asynchronous OCR worker.

OCR is the slowest stage (~2.8-4.7 s/frame on CPU).  Running it on a
dedicated worker thread with a bounded queue means:

* camera capture is never blocked by OCR;
* object detection/tracking runs independently;
* the UI reads the *latest completed* OCR result via
  :meth:`OcrWorker.latest_result`, never waiting synchronously.

Design: a single-slot "latest frame" slot.  If OCR is busy and a newer
frame arrives, the newest frame simply replaces the pending one — a slow
OCR never builds a backlog.  This is the "OCR every N frames" + "OCR
worker thread" + "OCR queue" strategy from the roadmap combined.
"""
import threading
import time
from typing import List, Optional

from src.ocr.ocr_engine import OcrEngine, OcrResult
from src.ocr.preprocess import preprocess
from src.utils.logger import setup_logger

_logger = setup_logger("OcrWorker")


class OcrWorker:
    """Background OCR with latest-result semantics."""

    def __init__(
        self,
        engine: OcrEngine,
        preprocess_strategy: str = "none",
        poll_interval: float = 0.05,
        timeout_ms: int = 0,
    ) -> None:
        """Configure the worker.

        Args:
            engine: The OCR engine to run in the background thread.
            preprocess_strategy: Preprocessing to apply before OCR
                (see src.ocr.preprocess).  "none" = full frame.
            poll_interval: Sleep between queue polls (seconds).
            timeout_ms: Hard cap on a single OCR call in milliseconds.
                0 = no timeout (RapidOCR is not interruptible mid-call,
                so the cap only skips *scheduling* new work when a call
                is already running longer than the cap).
        """
        self._engine = engine
        self._strategy = preprocess_strategy
        self._poll_interval = float(poll_interval)
        self._timeout_ms = int(timeout_ms)

        self._frame_slot: Optional[object] = None
        self._slot_lock = threading.Lock()
        self._result: List[OcrResult] = []
        self._result_lock = threading.Lock()
        self._result_ready = threading.Event()

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._busy = False
        self._runs = 0
        self._calls = 0
        self._last_latency_ms: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, frame) -> None:
        """Queue a frame for OCR (non-blocking; newest wins)."""
        with self._slot_lock:
            self._frame_slot = frame

    def latest_result(self) -> List[OcrResult]:
        """Return the most recently completed OCR result (non-blocking)."""
        with self._result_lock:
            return list(self._result)

    @property
    def has_result(self) -> bool:
        return self._result_ready.is_set()

    @property
    def is_busy(self) -> bool:
        return self._busy

    @property
    def runs(self) -> int:
        """Number of OCR calls completed."""
        return self._runs

    @property
    def last_latency_ms(self) -> float:
        return self._last_latency_ms

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="ocr-worker", daemon=True,
        )
        self._thread.start()
        _logger.info("OCR worker started (preprocess=%s)", self._strategy)

    def stop(self, join_timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
            self._thread = None

    def join(self, timeout: float = 30.0) -> None:
        """Block until a result is available (used by tests)."""
        self._result_ready.wait(timeout)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._slot_lock:
                frame = self._frame_slot
                self._frame_slot = None
            if frame is None:
                self._stop.wait(self._poll_interval)
                continue

            self._busy = True
            started = time.monotonic()
            try:
                processed = preprocess(frame, self._strategy)
                items = self._engine.read_text(processed)
            except Exception as exc:  # pragma: no cover - env dependent
                _logger.warning("OCR failed: %s", exc)
                items = []
            self._busy = False
            self._runs += 1
            self._last_latency_ms = (time.monotonic() - started) * 1000.0
            with self._result_lock:
                self._result = items
            self._result_ready.set()
            _logger.debug(
                "OCR done: %d line(s) in %.0f ms",
                len(items), self._last_latency_ms,
            )

    def _clear(self) -> None:  # pragma: no cover - tests use public API
        """Reset internal state (used by unit tests)."""
        with self._slot_lock:
            self._frame_slot = None
        with self._result_lock:
            self._result = []
        self._result_ready.clear()