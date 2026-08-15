"""Text-to-speech output using pyttsx3 (Windows SAPI5 / macOS / Linux).

pyttsx3 drives the operating system's native speech engine, so no extra
models or cloud calls are required.  Speech is enqueued on a background
worker so the vision loop never blocks on audio.

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
    """A small non-blocking TTS wrapper built on pyttsx3."""

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
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", int(rate))
            self._engine.setProperty("volume", float(volume))
            if voice_id:
                self._engine.setProperty("voice", voice_id)
        except Exception as exc:
            raise SpeechError(f"Failed to initialise TTS: {exc}") from exc

        self._running = True
        self._thread = threading.Thread(
            target=self._worker, name="tts-worker", daemon=True,
        )
        self._thread.start()
        _logger.info("Speech engine ready (rate=%d, volume=%.1f)",
                     rate, volume)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str) -> None:
        """Say `text`.  Returns immediately; audio plays in background."""
        if not text or not text.strip():
            return
        self._queue.put(text.strip())

    def say_now(self, text: str) -> None:
        """Say `text` synchronously (blocks until spoken)."""
        if not text or not text.strip():
            return
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as exc:
            raise SpeechError(f"TTS playback failed: {exc}") from exc

    def set_rate(self, rate: int) -> None:
        self._engine.setProperty("rate", int(rate))

    def shutdown(self) -> None:
        """Stop the worker thread and release the engine."""
        self._running = False
        self._queue.put("")  # unblock the worker
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self._engine.stop()
        except Exception:  # pragma: no cover - engine may be gone
            pass
        _logger.info("Speech engine shut down")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        while self._running:
            text = self._queue.get()
            if not text:
                continue
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as exc:  # pragma: no cover - env dependent
                _logger.warning("TTS error: %s", exc)
            finally:
                self._queue.task_done()
