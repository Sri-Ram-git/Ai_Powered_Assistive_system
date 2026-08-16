"""Speech Output module.

tts:           SpeechOutput — non-blocking text-to-speech via the OS
               speech engine.
speech_queue:  SpeechQueue / SpeechTier — prioritised, deduplicated,
               rate-limited bridge so speech never blocks or repeats.
"""
from src.audio.tts import SpeechOutput

__all__ = ["SpeechOutput"]
