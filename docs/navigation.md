# Navigation & Guidance Module (`src/navigation`)

## Overview

Stateless helpers that turn detections + OCR into the coarse cues a
visually-impaired user can act on: relative direction, rough distance,
and situational warnings (traffic signals, stop signs, crosswalks).

All functions are pure — image geometry + detections in, plain guidance
values out.  No state, no side effects.

## Architecture

```
┌──────────────────────────────────────────────┐
│              navigation/                     │
│  guidance.py                                 │
│    direction_of(box, frame_w)  → str         │
│    distance_estimate(box, frame_h) → float   │
│    nearest_obstacle(dets, frame_w) → det?    │
│    scene_cues(dets, ocr, w, h) → List[str]   │
└──────────────────────────────────────────────┘
```

## Guidance rules

### Direction

The frame width is split into three equal thirds:

| Box centre | Result |
|---|---|
| `< 1/3 · w` | `left` |
| `1/3 · w … 2/3 · w` | `ahead` |
| `> 2/3 · w` | `right` |

### Distance

Pinhole scale model:

```
focal = (frame_h / 2) / tan(VFOV / 2)         # VFOV = 60°
distance = reference_height · focal / box_h   # reference = 1.7 m
```

Uncalibrated but monotonic — a taller box ⇒ closer object.  Clamped to
a minimum of 0.2 m.

### Cues

`scene_cues` emits, in order of appearance:

- Traffic-light warning (largest such box): `Traffic light <dir>`
- Stop-sign warning: `Stop sign <dir>`
- Crosswalk / Do-not-walk warning from OCR keywords
- Per-person: `Person <dir>, <n> metres`
- Per-vehicle: `<Vehicle> <dir>, <n> metres`

## Function reference

| Function | Description |
|---|---|
| `direction_of(box, frame_w)` | `left` / `ahead` / `right` |
| `distance_estimate(box, frame_h, reference_metres=1.7)` | Metres (heuristic) |
| `nearest_obstacle(detections, frame_w)` | Largest box, or `None` |
| `scene_cues(detections, ocr_items, frame_w, frame_h)` | List of cue strings |

## Usage

```python
from src.navigation import scene_cues

cues = scene_cues(detections, ocr_items, frame.shape[1], frame.shape[0])
```

## Execution flow

```
python src/assist/assist_app.py    # cues feed the decision engine
```

## Dependencies

- Python 3.11+; imports `detection` and `ocr` data types

## Limitations

- Distance is a heuristic scale model, not calibrated to any lens.
- Direction uses only the horizontal position of the box centre.
- Crosswalk detection depends on OCR reading English keywords.

## Future extensions

- Calibrated depth via stereo / depth camera
- Vertical position analysis (curb vs. eye-level obstacles)
- Multilingual crosswalk keyword sets
