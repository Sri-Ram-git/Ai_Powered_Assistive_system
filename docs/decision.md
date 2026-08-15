# Decision Engine Module (`src/decision`)

## Overview

A rule-based engine that turns a frame summary (object detections +
OCR results + frame geometry) into a **prioritised, spoken phrase**.
Its core logic is a pure function (`evaluate`); the `DecisionEngine`
wrapper adds temporal state so the user is not spammed with speech.

## Architecture

```
┌──────────────────────────────────────────────┐
│               decision/                      │
│  engine.py                                   │
│    FrameSummary  (detections, ocr, geometry) │
│    evaluate(summary) → List[Decision]        │
│    DecisionEngine                            │
│      decide(summary, now) → Optional[str]    │
└──────────────────────────────────────────────┘
```

Dependency direction: `decision → detection`, `decision → ocr`,
`decision → navigation`.

## Decision rules

`evaluate` produces navigation cues via `navigation.scene_cues`, then
assigns a numeric priority and sorts (0 = highest):

| Priority | Cue | Source |
|---|---|---|
| 0 | Traffic light ahead/left/right | detection |
| 1 | Stop sign ... | detection |
| 2 | Crosswalk / Do-not-walk sign ahead | OCR |
| 3 | Text says, <recognised text> | OCR (optional) |
| 3 | Person <direction>, <distance> | detection |
| 4 | <Vehicle> <direction>, <distance> | detection |
| 5 | Obstacle ahead | detection |

Reading recognised text aloud is optional: enable it with `read_ocr_text`
(config `decision.speak_ocr_text: true`, default on) or toggle at
runtime with the `t` key.  Text is capped at `max_ocr_chars` (80).

If the nearest obstacle is an `obstacle`-category object, an additional
"Obstacle ahead" decision is appended.

Distances use per-label reference heights (person 1.7 m, car 1.5 m, bus
3.2 m, ...) phrased naturally: "Person ahead, about 3 metres", "very
close", or "far away".

## Cooldown / rate limiting

`DecisionEngine.decide` speaks at most one phrase and applies:

1. **Priority gate** — skip decisions with `priority > min_priority`.
2. **First utterance** — always allowed.
3. **Global rate limit** — nothing is spoken until `cooldown_seconds`
   have elapsed since the last phrase.
4. **De-duplication** — the identical phrase is not repeated
   (`reset()` re-enables it, e.g. when the user changes scene).

## Data model

| Member | Description |
|---|---|
| `FrameSummary(detections, ocr_items, frame_w, frame_h, read_ocr_text, max_ocr_chars)` | Frame snapshot |
| `Decision(text, priority, source)` | One candidate phrase |
| `DecisionEngine(cooldown_seconds, min_priority, read_ocr_text, max_ocr_chars)` | Stateful wrapper |
| `decide(summary, now=None)` | → phrase or `None` |
| `set_read_ocr_text(enabled)` / `read_ocr_text` | Toggle OCR reading aloud |
| `reset()` | Clears cooldown / last-spoken state |

## Usage

```python
from src.decision import DecisionEngine, FrameSummary

engine = DecisionEngine(cooldown_seconds=4.0)
summary = FrameSummary(detections=dets, ocr_items=texts,
                       frame_w=w, frame_h=h)
phrase = engine.decide(summary)   # e.g. "Person ahead, 3 metres"
if phrase:
    tts.speak(phrase)
```

## Execution flow

```
python src/assist/assist_app.py    # decision engine drives TTS each frame
```

## Dependencies

- Python 3.11+; imports `detection`, `ocr`, `navigation`

## Limitations

- Rules are hand-tuned; priority ordering is opinionated and may need
  adjustment per use case.
- The engine considers only the single highest-priority cue per frame.
- Cooldown-based speech can miss transient events that occur mid-window.

## Future extensions

- Confidence-weighted speech (only speak when confidence is stable)
- Barge-in priority (higher-priority cues interrupt lower ones)
- Machine-learned severity scoring instead of fixed rules
