# Live Vision Playground

## Overview

The Week 1 integration application.  It wires the camera module, the
image-processing module, and the HUD together into one interactive,
fullscreen webcam tool controlled entirely from the keyboard and mouse.

## Architecture

```
                  ┌────────────────────────────────────────┐
  camera.py ─────▶│              playground.py            │
                  │                                        │
  processing.py ─▶│  pipeline: filter → gray → edge → thresh│
                  │                                        │
  hud.py ────────▶│  HUD overlay + draggable bars          │
                  └────────────────────────────────────────┘
                        │            │            │
                        ▼            ▼            ▼
                   fullscreen    screenshots    recording
                    display      /processed    (VideoRecorder)
```

## Execution

```
python src/playground/playground.py [--camera 0]
```

## Controls

| Key | Action |
|---|---|
| `1`–`7` | Select filter: original / gaussian / median / bilateral / sharpen / sobel / laplacian |
| `g` | Toggle grayscale |
| `e` | Toggle edge detection (Canny) |
| `t` | Toggle binary threshold |
| `s` | Save screenshot (raw frame) |
| `v` | Save processed image |
| `r` | Record 5-second clip (UI stays live, REC pill shown) |
| `space` | Reset all toggles |
| `q` | Quit |
| **mouse** | Drag the menu bar and dashboard anywhere |

The HUD mode label reflects the active pipeline, e.g.
`GAUSSIAN | GRAY | EDGE`.  Toasts confirm every save and recording.

## Pipeline order

```
raw frame
  → base filter (1-7)
  → grayscale (if g)
  → Canny edge (if e)
  → threshold (if t)
  → display (letterboxed to screen) with HUD
```

## Dependencies

- `src.camera` (Camera, CameraManager, HUD, VideoRecorder, display helpers)
- `src.image_processing` (filters, edges, threshold)

## Limitations

- Requires a webcam; runs a blocking fullscreen window (desktop only).
- Edge/threshold stages reduce output to single channel.

## Future extensions

- Object-detection overlay (Week 2: YOLO)
- OCR region selection (Week 3)
- Sliders for live filter parameters
- Multi-camera switching at runtime
