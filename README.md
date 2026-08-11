# AI-Powered Assistive Vision System

A modular computer-vision foundation for an assistive device that will
eventually help visually impaired people navigate their surroundings
(Camera → Object Detection → OCR → Decision Engine → Speech Output).

**This repository currently contains the Week 1 vision foundation only** —
no AI models, no OCR, no speech.  Those arrive in Weeks 2-3.

## Modules

| Module | Package | Purpose |
|---|---|---|
| Camera System | `src/camera` | Webcam init/selection, mirror feed, FPS, screenshots, threaded recording, professional draggable HUD |
| Image Fundamentals | `src/image_fundamentals` | Read/save, resize, crop, rotate, flip, colour-space conversions, pixel inspection, histograms |
| Image Processing | `src/image_processing` | Blur, threshold, Canny/Sobel/Laplacian edges, sharpen, brightness/contrast, noise add/remove |
| Morphology & Shapes | `src/morphology` | Erode/dilate/open/close, contours, geometry metrics, circle/rectangle/triangle detection |
| Vision Playground | `src/playground` | Live fullscreen webcam app with filter switching, toggles, recording, draggable HUD |
| Infrastructure | — | `tests/`, `docs/`, `configs/`, `assets/`, `logs/` |

## Quickstart

```bash
pip install -r requirements.txt

# Test the camera (mirror feed, fullscreen, draggable HUD)
python src/camera/camera_test.py

# Playground: live filters + toggles + recording
python src/playground/playground.py

# Demos (no webcam needed)
python src/image_fundamentals/image_demo.py
python src/image_processing/processing_demo.py
python src/morphology/demo.py

# Run the test suite
python -m pytest tests -q
```

## Repository layout

```
├── src/
│   ├── camera/                # camera.py, camera_manager.py,
│   │                          # camera_utils.py, hud.py, camera_test.py
│   ├── image_fundamentals/    # image_utils.py, image_demo.py
│   ├── image_processing/      # processing.py, demos
│   ├── morphology/            # contour_utils.py, shape_detector.py
│   ├── playground/            # playground.py
│   └── utils/                 # logger.py, exceptions.py
├── tests/                     # pytest suite (hardware-free)
├── docs/                      # per-module + architecture docs
├── configs/                   # YAML configuration
├── assets/                    # local media (never pushed — see .gitignore)
└── logs/                      # runtime logs
```

## Security & privacy

The `.gitignore` is configured to **never push personal media**: the
`assets/` tree (screenshots, recordings, photos) is blocked except for
placeholder `.gitkeep` files.  Only the synthetic, generated test scene
under `src/image_fundamentals/sample_images/` is versioned.

## Coding standards

- Python 3.11+, OpenCV, NumPy, Pillow (HUD text), pytest, PyYAML
- Type hints, docstrings, logging, explicit exception handling
- No global variables; stateless functions; PEP 8 style
- Per-module docs include architecture, function reference, dependencies,
  limitations, and future extensions

## Documentation

- `docs/architecture.md` — system design and module map
- `docs/camera.md`, `docs/image_fundamentals.md`,
  `docs/image_processing.md`, `docs/morphology.md`, `docs/playground.md`

## License

MIT — see [LICENSE](LICENSE).
