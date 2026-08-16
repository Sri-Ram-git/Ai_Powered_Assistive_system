"""Speech input module.

Microphone → Speech-to-Text → Command Parser → Action.

Design decisions:

* The command system is **deterministic** (keyword matching), so simple
  commands never go through an LLM.
* STT is pluggable: ``FasterWhisperSTT`` (local, offline, faster-whisper)
  is the reference implementation; ``KeywordSTT`` provides a pure-Python
  fallback that recognises the command set without any model (used for
  offline-first mode and for tests).  The active backend is chosen from
  configuration.
"""
from src.speech.commands import Command, CommandRegistry
from src.speech.command_parser import CommandParser
from src.speech.stt import (
    BaseSTT,
    FasterWhisperSTT,
    KeywordSTT,
    create_stt,
)

__all__ = [
    "BaseSTT",
    "Command",
    "CommandParser",
    "CommandRegistry",
    "FasterWhisperSTT",
    "KeywordSTT",
    "create_stt",
]