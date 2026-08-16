"""Text-to-speech output using pyttsx3 (Windows SAPI5 / macOS / Linux).

pyttsx3 drives the operating system's native speech engine, so no extra
models or cloud calls are required.  Speech is enqueued on a background
worker so the vision loop never blocks on audio.

Important: the pyttsx3 SAPI5 driver only produces sound for the first
`runAndWait()` per engine instance, so we never use runAndWait().
Instead the engine is created *inside* the worker thread, driven with
`startLoop(False)` and an explicit `iterate()` pump loop that runs until
the engine reports not busy — every phrase audibly plays.

Usage:
    tts = SpeechOutput(rate=170)
    tts.speak("Person on your left")
    tts.shutdown()
"""
import queue
import threading
from typing import Optional

import pyttsx3

from src.utils.exceptions import SpeechError
from src.utils.logger import setup_logger

_logger = setup_logger("SpeechOutput")


class SpeechOutput:
    """A small non-blocking TTS wrapper built on pyttsx3.

    `speak()` is non-blocking: text is enqueued and spoken by a single
    background worker thread, in order.
    """

    def __init__(
        self,
        rate: int = 165,
        volume: float = 1.0,
        voice_id: Optional[str] = None,
    ) -> None:
        """Configure the speech engine.

        Args:
            rate: Words-per-minute speaking rate.
            volume: 0.0 .. 1.0 volume.
            voice_id: Optional system voice identifier; None = default.

        Raises:
            SpeechError: If the platform speech engine cannot start.
        """
        self._rate = int(rate)
        self._volume = float(volume)
        self._voice_id = voice_id
        self._queue: "queue.Queue[tuple]" = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._engine = None  # created on the worker thread
        self._ready = threading.Event()
        self._init_error: Optional[Exception] = None

        self._running = True
        self._thread = threading.Thread(
            target=self._worker, name="tts-worker", daemon=True,
        )
        self._thread.start()

        # Wait for the worker to create the engine (on its own thread) so
        # broken setups still raise SpeechError early.
        if not self._ready.wait(timeout=10.0):
            self._running = False
            raise SpeechError("Timed out waiting for the speech engine to start")
        if self._init_error is not None:
            self._running = False
            raise SpeechError(
                f"Failed to initialise TTS: {self._init_error}"
            ) from self._init_error

        _logger.info("Speech engine ready (rate=%d, volume=%.1f)",
                     rate, volume)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str) -> None:
        """Say `text`.  Returns immediately; audio plays in background."""
        if not text or not text.strip():
            return
        self._queue.put((text.strip(), None))

    def say_now(self, text: str) -> None:
        """Say `text` synchronously by enqueuing and waiting for it."""
        if not text or not text.strip():
            return
        done = threading.Event()
        self._queue.put((text.strip(), done))
        done.wait(timeout=60.0)

    def set_rate(self, rate: int) -> None:
        self._rate = int(rate)
        if self._engine is not None:
            try:
                self._engine.setProperty("rate", self._rate)
            except Exception:  # pragma: no cover - engine may be busy
                pass

    def shutdown(self) -> None:
        """Stop the worker thread and release the engine."""
        self._running = False
        self._queue.put(("", None))  # unblock the worker
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        _logger.info("Speech engine shut down")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        # Create the engine inside the worker thread so every say() runs
        # on the same thread that owns the engine.
        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self._rate)
            self._engine.setProperty("volume", self._volume)
            if self._voice_id:
                self._engine.setProperty("voice", self._voice_id)
        except Exception as exc:  # pragma: no cover - env dependent
            _logger.error("TTS init failed on worker: %s", exc)
            self._init_error = exc
            self._running = False
            self._ready.set()
            return

        self._ready.set()

        # Use the manual event loop.  pyttsx3's SAPI5 driver has a known
        # bug where runAndWait() only produces sound for the first phrase
        # per process; startLoop(False) + iterate() avoids that.
        self._engine.startLoop(False)

        while self._running:
            text, done = self._queue.get()
            if not text:
                try:
                    if done is not None:
                        done.set()
                finally:
                    self._queue.task_done()
                continue
            try:
                self._engine.say(text)
                self._pump_loop(self._engine)
            except Exception as exc:  # pragma: no cover - env dependent
                _logger.warning("TTS error: %s", exc)
            finally:
                if done is not None:
                    done.set()
                self._queue.task_done()

        try:
            self._engine.endLoop()
        except Exception:  # pragma: no cover - engine may be gone
            pass

    @staticmethod
    def _pump_loop(engine, engine_timeout: float = 8.0) -> None:
        """Service the SAPI event loop until the phrase finishes.

        `iterate()` returns as soon as the COM message pump is serviced;
        the SAPI voice is still rendering audio, so keep pumping until the
        engine's busy flag clears (real completion signal) or timeout.
        """
        import time as _time

        deadline = _time.time() + engine_timeout
        while _time.time() < deadline:
            try:
                engine.iterate()
            except Exception:
                break
            try:
                if not engine.isBusy():
                    break
            except Exception:
                pass
            _time.sleep(0.02)
