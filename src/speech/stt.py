"""Speech-to-text backends.

Two implementations:

* ``KeywordSTT`` — pure-Python, offline, no model.  It matches audio
  input against the command registry via keyword spotting.  Since real
  audio transcription requires an STT model (which we do not want to
  hard-depend on in the core), this backend accepts a transcript text
  and returns the matched command — enabling offline-first command
  control and hardware-free tests.
* ``FasterWhisperSTT`` — local, offline Whisper transcription via
  `faster-whisper` (optional dependency).  On a real device you would
  run this in a background thread capturing microphone audio.

``create_stt`` selects a backend from configuration so the rest of the
system never cares which backend is active.
"""
import threading
from typing import List, Optional

from src.speech.commands import CommandRegistry
from src.speech.command_parser import CommandParser, ParsedCommand
from src.utils.logger import setup_logger

_logger = setup_logger("STT")


class BaseSTT:
    """Interface every STT backend implements."""

    def transcribe(self, text_or_audio) -> str:
        """Return the recognised transcript for the input.

        ``text_or_audio`` is intentionally loosely typed: real audio
        backends accept a file path / numpy audio; the keyword backend
        accepts plain text.
        """
        raise NotImplementedError  # pragma: no cover


class KeywordSTT(BaseSTT):
    """Deterministic, model-free STT for the fixed command set.

    Takes a raw transcript (from any upstream audio-to-text source or a
    typed command) and resolves it to a parsed command.  This keeps the
    voice-control layer testable and offline-capable.
    """

    def __init__(self, registry: Optional[CommandRegistry] = None) -> None:
        self._parser = CommandParser(registry or CommandRegistry())

    def transcribe(self, text_or_audio: str) -> str:
        return str(text_or_audio)

    def parse(self, text: str) -> ParsedCommand:
        return self._parser.parse(text)

    def help_text(self) -> str:
        return self._parser._registry.help_text()


class FasterWhisperSTT(BaseSTT):
    """Local Whisper transcription via faster-whisper (optional).

    Requires the ``faster-whisper`` package and a Whisper model download
    (e.g. "small", "base").  Offline once the model is cached.  Only
    used when explicitly configured — the core never hard-depends on it.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        registry: Optional[CommandRegistry] = None,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "faster-whisper is not installed. "
                "pip install faster-whisper"
            ) from exc
        self._model = WhisperModel(model_size, device=device,
                                   compute_type=compute_type)
        self._parser = CommandParser(registry or CommandRegistry())
        _logger.info("Whisper STT ready (model=%s)", model_size)

    def transcribe(self, audio_path: str) -> str:
        segments, _info = self._model.transcribe(audio_path)
        return " ".join(seg.text.strip() for seg in segments).strip()

    def parse(self, text: str) -> ParsedCommand:
        return self._parser.parse(text)


def create_stt(
    backend: str = "keyword",
    model_size: str = "base",
) -> BaseSTT:
    """Create an STT backend from configuration.

    Args:
        backend: "keyword" (default, offline, no model) or "whisper".
        model_size: Whisper model size when backend is "whisper".

    Returns:
        A BaseSTT instance.

    Raises:
        ValueError: For an unknown backend.
    """
    if backend == "keyword":
        return KeywordSTT()
    if backend == "whisper":
        return FasterWhisperSTT(model_size=model_size)
    raise ValueError(f"Unknown STT backend: {backend}")