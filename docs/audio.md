# Speech Output Module (`src/audio`)

## Overview

Text-to-speech via **pyttsx3**, which drives the operating system's
native speech engine — SAPI5 on Windows, NSSpeechSynthesizer on macOS,
espeak on Linux. No cloud calls, no model downloads.

## Architecture

```
┌──────────────────────────────────────────────┐
│                 audio/                       │
│  tts.py                                      │
│    SpeechOutput                              │
│      speak(text)          (non-blocking)     │
│      say_now(text)        (blocking)         │
│      set_rate(n) / shutdown()                │
└──────────────────────────────────────────────┘
```

`SpeechOutput` owns a small worker thread. `speak()` puts text on a
queue and returns immediately; the worker calls the engine's `say` +
`runAndWait` in the background so the vision loop never blocks on audio.

## Data model

| Member | Description |
|---|---|
| `SpeechOutput(rate=165, volume=1.0, voice_id=None)` | Constructor |
| `speak(text)` | Enqueue text; returns immediately |
| `say_now(text)` | Speak synchronously (blocks) |
| `set_rate(rate)` | Change speaking rate live |
| `shutdown()` | Stop the worker and release the engine |

## Usage

```python
from src.audio import SpeechOutput

tts = SpeechOutput(rate=165)
tts.speak("Person on your left, three metres")
# ... keep processing frames ...
tts.shutdown()
```

## Execution flow

```
python src/assist/assist_app.py    # 'm' mutes / unmutes speech
```

## Dependencies

- Python 3.11+, `pyttsx3`
- `pywin32` / `comtypes` are pulled in automatically on Windows
- A system voice must be installed (Windows ships David & Zira by default)

## Limitations

- Voice and quality are those of the OS engine, not a neural TTS.
- `say_now` blocks the calling thread; prefer `speak` for live loops.
- No interruption API: a long phrase cannot be cut short by a new one
  (future work: per-phrase priority with stop()).

## Future extensions

- Interrupt / barge-in (call `stop()` before speaking a new phrase)
- Phoneme or SSML control for clearer object names
- Offline neural TTS backend as an optional provider
