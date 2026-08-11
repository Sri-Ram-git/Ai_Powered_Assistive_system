# Architecture

## System goal (Week 3)

```
Camera → Object Detection → OCR → Decision Engine → Speech Output
```

Week 1 delivers only the vision foundation; AI modules land in Weeks 2-3.

## Week 1 module map

```
┌─────────────────────────────────────────────────────────────────┐
│                        assistive-vision-system                   │
├───────────────┬───────────────┬───────────────┬─────────────────┤
│  src/camera   │ image_        │ image_        │  morphology/    │
│               │ fundamentals  │ processing    │                 │
│  acquisition  │  utilities    │  filters/      │  shapes &       │
│  HUD, record  │  I/O, colour  │  edges/noise   │  contours       │
├───────────────┴───────────────┴───────────────┴─────────────────┤
│                      src/playground (integration)                │
│            src/utils (logger, exceptions) — shared               │
│            tests/ | docs/ | configs/ | assets/ | logs/           │
└─────────────────────────────────────────────────────────────────┘
```

## Layer rules

- **Stateless functions** in `image_fundamentals` and `image_processing`:
  input → output, no side effects, no global state.
- **Stateful objects** only where required: `Camera` (device handle),
  `HUD` (presentation state), `VideoRecorder` (background thread),
  `ShapeDetector` (tuning parameters).
- **No cross-module circular imports.** Dependency direction:
  `utils ← camera`, `utils ← image_fundamentals ← image_processing ←
  morphology ← playground`.
- **Presentation vs logic:** HUD only draws; it never touches the
  camera or pipeline.  The playground orchestrates.

## Data flow (live feed)

```
Camera.read() ──▶ [filter] ─▶ [gray] ─▶ [edge] ─▶ [thresh]
                                    │
                    scale_to_fit ───┘
                                    │
                              HUD.render() ──▶ cv2.imshow
```

## Future module slots (Weeks 2-3)

Each new capability becomes a sibling package under `src/`:

| Package | Slot | Feeds |
|---|---|---|
| `src/detection` | YOLO object detection | post-processing |
| `src/ocr` | text recognition (Tesseract/PaddleOCR) | decision engine |
| `src/decision` | rule-based decisions | speech output |
| `src/audio` | TTS / speech synthesis | user |
| `src/navigation` | environment guidance | audio |

## Configuration & observability

- All settings are externalised to `configs/*.yaml`.
- All modules log through `src.utils.logger.setup_logger` to console +
  `logs/app.log`.
- Custom exception hierarchy in `src/utils/exceptions.py`.
