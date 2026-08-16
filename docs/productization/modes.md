# Product Modes

The device has **five user-facing modes** — different emphases of the
same underlying pipeline.  Modes are behavioural presets over the engine
(`src/modes/`), never a recompile or restart.

| Mode       | OCR | Navigation | Object chatter | Scene describe | Quiet |
|------------|-----|------------|----------------|----------------|-------|
| `object`   | on  | on         | on             | —              | no    |
| `reading`  | on  | off        | off            | —              | no    |
| `navigation` | off | on       | on             | —              | no    |
| `scene`    | on  | off        | off            | yes            | yes   |
| `voice`    | off | off        | off            | —              | yes   |

## How modes change behaviour

Switching modes applies knobs to the live config:

* `ocr_enabled` — whether the OCR worker is fed frames.
* `navigation_enabled` — whether tracking monitor / decision guidance
  phrases are generated.
* `announce_objects` — object proximity chatter on.
* `announce_text` — recognised text read aloud on.
* `scene_describe` — a higher-level (deterministic) scene description is
  spoken.
* `quiet` — minimal speaking.

**Safety is never mode-dependent.**  The SafetyEngine is assessed on
every frame regardless of mode, and an urgent hazard always wins the
response planner (it bypasses the cooldown).

## Switching modes

* **Web UI**: the dashboard has a Mode card with one button per mode.
* **API**: `POST /api/mode` `{"mode": "reading"}`.
* **Config**: `app.mode` in `configs/assist_config.yaml` sets the
  starting mode.

## Where modes are wired

* `src/modes/manager.py` — `MODES` table, `ModeManager.apply()`.
* `src/core/pipeline.py` — `set_mode()` applies behaviour; the detect
  loop honours `announce_objects` / `scene_describe`; `announce_text`
  is propagated to the DecisionEngine.
* `src/api/routes.py` — `POST /api/mode`.

## Extending modes

Add a `ModeBehavior` entry in `src/modes/manager.py` and (if needed) a
new knob.  Keep the invariant: a mode may never disable the safety
engine or the response planner's urgent-safety path.