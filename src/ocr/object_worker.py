"""Object-aware OCR worker.

Runs OCR for object ROIs on a dedicated thread so camera capture, YOLO,
tracking, the UI, and TTS are never blocked.

Design (matches the "latest-request / replace-oldest" requirement):

    * a single pending-request slot — if OCR is busy and a newer request
      arrives, the newer request simply replaces the pending one, so a
      backlog can never form and latency never builds up;
    * the cheap text-presence gate runs first; when the ROI has no
      plausible text the request finishes instantly (status="no_text")
      without touching the OCR net;
    * preprocessing variants are tried and the best result is kept;
    * a request that exceeds ``timeout_ms`` is marked ``timeout`` and
      logged — OCR can never appear to hang forever;
    * failures surface as status="error", never as exceptions in callers.
"""
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from src.ocr.object_ocr import (
    DEFAULT_VARIANTS,
    ObjectOcrResult,
    combine_results,
    run_variants,
)
from src.ocr.text_presence import has_text
from src.utils.logger import setup_logger

_logger = setup_logger("ObjectOcrWorker")


@dataclass
class ObjectOcrRequest:
    """A single object OCR request (or a manual full-ROI request)."""

    roi: np.ndarray                       # (possibly upscaled) ROI image
    track_id: Optional[int] = None
    label: Optional[str] = None
    roi_box: Optional[Tuple[int, int, int, int]] = None
    scale: float = 1.0
    trigger: str = "new"
    source: str = "object"
    request_id: int = 0
    created: float = field(default_factory=time.time)


class ObjectOcrWorker:
    """Background object OCR with newest-request-wins semantics."""

    def __init__(
        self,
        engine,
        variants: Sequence[str] = DEFAULT_VARIANTS,
        min_confidence: float = 0.3,
        stop_confidence: float = 0.92,
        text_presence: bool = True,
        presence_threshold: float = 0.35,
        timeout_ms: int = 2000,
        poll_interval: float = 0.02,
        on_result: Optional[Callable[[ObjectOcrResult], None]] = None,
    ) -> None:
        """Configure the worker.

        Args:
            engine: Object exposing ``read_text(image) -> [OcrResult]``.
            variants: Preprocessing strategies to try per request.
            min_confidence: Drop OCR lines below this confidence.
            stop_confidence: Skip remaining variants when reached.
            text_presence: Gate requests behind the cheap heuristic.
            presence_threshold: Score threshold for the gate.
            timeout_ms: Mark a request as ``timeout`` when a single OCR
                call exceeds this (RapidOCR is not interruptible mid-call,
                so the marker is applied once the call returns).
            poll_interval: Sleep between queue polls (seconds).
            on_result: Optional callback invoked with each finished
                ObjectOcrResult (called on the worker thread).
        """
        self._engine = engine
        self._variants = list(variants)
        self._min_conf = float(min_confidence)
        self._stop_conf = float(stop_confidence)
        self._text_presence = bool(text_presence)
        self._presence_threshold = float(presence_threshold)
        self._timeout_ms = int(timeout_ms)
        self._poll_interval = float(poll_interval)
        self._on_result = on_result

        self._pending: Optional[ObjectOcrRequest] = None
        self._slot_lock = threading.Lock()
        self._latest: Optional[ObjectOcrResult] = None
        self._latest_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._busy = False
        self._request_seq = 0
        self._stats = {
            "runs": 0, "no_text": 0, "timeouts": 0, "errors": 0,
            "replaced": 0, "empty": 0, "last_latency_ms": 0.0,
            "total_latency_ms": 0.0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self,
        roi: np.ndarray,
        track_id: Optional[int] = None,
        label: Optional[str] = None,
        roi_box: Optional[Tuple[int, int, int, int]] = None,
        scale: float = 1.0,
        trigger: str = "new",
        source: str = "object",
    ) -> ObjectOcrRequest:
        """Queue an object ROI for OCR (non-blocking; newest wins)."""
        with self._slot_lock:
            if self._pending is not None:
                self._stats["replaced"] += 1
            self._request_seq += 1
            req = ObjectOcrRequest(
                roi=roi,
                track_id=track_id,
                label=label,
                roi_box=roi_box,
                scale=scale,
                trigger=trigger,
                source=source,
                request_id=self._request_seq,
            )
            self._pending = req
        return req

    def latest(self) -> Optional[ObjectOcrResult]:
        """The most recently finished result (non-blocking)."""
        with self._latest_lock:
            return self._latest

    def latest_result(self):
        """Back-compat alias mirroring OcrWorker.latest_result()."""
        result = self.latest()
        if result is None:
            return []
        return result

    @property
    def is_busy(self) -> bool:
        return self._busy

    @property
    def last_latency_ms(self) -> float:
        return self._stats["last_latency_ms"]

    @property
    def runs(self) -> int:
        return self._stats["runs"]

    def stats(self) -> Dict:
        """Snapshot of worker counters."""
        out = dict(self._stats)
        out["pending"] = self._pending is not None
        return out

    def clear(self) -> None:
        """Drop any pending request and the latest result."""
        with self._slot_lock:
            self._pending = None
        with self._latest_lock:
            self._latest = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="object-ocr-worker", daemon=True,
        )
        self._thread.start()
        _logger.info(
            "Object OCR worker started (variants=%d, text_presence=%s)",
            len(self._variants), self._text_presence,
        )

    def stop(self, join_timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=join_timeout)
            self._thread = None

    def join(self, timeout: float = 30.0) -> None:
        """Block until the pending request is processed (tests)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._pending is None and not self._busy:
                return
            time.sleep(0.02)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._slot_lock:
                req = self._pending
                self._pending = None
            if req is None:
                self._stop.wait(self._poll_interval)
                continue

            self._busy = True
            result = self._process(req)
            self._busy = False
            self._stats["runs"] += 1
            self._stats["last_latency_ms"] = result.latency_ms
            self._stats["total_latency_ms"] += result.latency_ms

            with self._latest_lock:
                self._latest = result
            if self._on_result is not None:
                try:
                    self._on_result(result)
                except Exception:  # pragma: no cover - callback safety
                    _logger.warning("OCR result callback failed", exc_info=True)

            _logger.debug(
                "Object OCR done: status=%s track=%s lat=%.0fms",
                result.status, result.track_id, result.latency_ms,
            )

    def _process(self, req: ObjectOcrRequest) -> ObjectOcrResult:
        started = time.monotonic()

        if req.roi is None or req.roi.size == 0:
            return self._finish(req, started, status="error")

        if self._text_presence and not has_text(
                req.roi, self._presence_threshold):
            self._stats["no_text"] += 1
            return self._finish(req, started, status="no_text")

        try:
            variant, items, _latency = run_variants(
                self._engine,
                req.roi,
                self._variants,
                stop_confidence=self._stop_conf,
                min_confidence=self._min_conf,
            )
            text, confidence = combine_results(items)
        except Exception as exc:  # pragma: no cover - env dependent
            _logger.warning("Object OCR failed: %s", exc)
            self._stats["errors"] += 1
            return self._finish(req, started, status="error")

        elapsed_ms = (time.monotonic() - started) * 1000.0
        if not text:
            self._stats["empty"] += 1
            return ObjectOcrResult(
                track_id=req.track_id, label=req.label, text="",
                confidence=0.0, roi_box=req.roi_box, variant=variant,
                scale=req.scale, timestamp=time.time(),
                latency_ms=elapsed_ms, status="empty",
                trigger=req.trigger, source=req.source,
            )

        if self._timeout_ms and elapsed_ms > self._timeout_ms:
            self._stats["timeouts"] += 1
            _logger.warning(
                "OCR_TIMEOUT track=%s label=%s elapsed=%.0fms (cap=%dms)",
                req.track_id, req.label, elapsed_ms, self._timeout_ms,
            )

        status = "timeout" if (
            self._timeout_ms and elapsed_ms > self._timeout_ms) else "ok"
        return ObjectOcrResult(
            track_id=req.track_id,
            label=req.label,
            text=text,
            confidence=confidence,
            raw_text=text,
            roi_box=req.roi_box,
            variant=variant,
            scale=req.scale,
            timestamp=time.time(),
            latency_ms=elapsed_ms,
            status=status,
            trigger=req.trigger,
            source=req.source,
        )

    def _finish(self, req: ObjectOcrRequest, started: float,
                status: str) -> ObjectOcrResult:
        elapsed_ms = (time.monotonic() - started) * 1000.0
        return ObjectOcrResult(
            track_id=req.track_id, label=req.label, text="",
            confidence=0.0, roi_box=req.roi_box, scale=req.scale,
            timestamp=time.time(), latency_ms=elapsed_ms, status=status,
            trigger=req.trigger, source=req.source,
        )