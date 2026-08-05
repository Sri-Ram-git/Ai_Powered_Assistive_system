# Image Fundamentals Module

## Overview

Stateless, pure utility functions for the basics of computer vision:
loading, saving, resizing, cropping, rotation, flipping, colour-space
conversion, pixel inspection, image metadata, and histograms.

These functions form the foundation every later module builds on
(processing, morphology, detection, OCR).

## Architecture

```
              ┌──────────────────────────────────────┐
              │           image_utils.py             │
              │                                      │
   path ─────▶│  I/O        read_image, save_image    │
              │  Geometry   resize, crop, flip,       │
              │             rotate                    │
   ndarray ──▶│  Colour     to_grayscale, to_rgb,     │
              │             to_bgr, to_hsv            │
              │  Inspect    image_info, image_stats,  │
              │             pixel_value               │
              │  Histogram  histogram,                │
              │             histogram_image           │
              └──────────────────────────────────────┘
              │
              ▼
   ndarray / dict / tuple        (pure — never mutates input)
```

`image_demo.py` is the only entry point with side effects (file I/O and
optional window display); the library itself is stateless.

## Files

| File | Responsibility |
|---|---|
| `image_utils.py` | All utility functions (pure, typed, logged) |
| `image_demo.py` | Generates a sample scene and demos every function |
| `sample_images/test_scene.png` | Synthetic test scene (generated, no personal media) |

## Function reference

### I/O

| Function | Input | Output | Description |
|---|---|---|---|
| `read_image(path, grayscale=False)` | str, bool | ndarray | Load image (BGR or gray) |
| `save_image(image, path, jpeg_quality=95)` | ndarray, str, int | str | Save image; returns absolute path |

### Geometric transforms

| Function | Input | Output | Description |
|---|---|---|---|
| `resize(image, width, height, scale, interpolation)` | ndarray, ints/None | ndarray | Resize by dims or scale |
| `crop(image, x, y, width, height)` | ndarray, 4×int | ndarray | Crop ROI (bounds checked) |
| `flip(image, flip_code)` | ndarray, int | ndarray | 0=vertical, 1=horizontal |
| `rotate(image, angle, center, scale, border)` | ndarray, float | ndarray | Rotate CCW by degrees |

### Colour space

| Function | Input | Output | Description |
|---|---|---|---|
| `to_grayscale(image)` | ndarray | ndarray (1ch) | Grayscale conversion |
| `to_rgb(image)` | ndarray | ndarray | BGR → RGB |
| `to_bgr(image)` | ndarray | ndarray | Gray/RGB/RGBA → BGR |
| `to_hsv(image)` | ndarray | ndarray | BGR → HSV |
| `to_bgr_from_hsv(image)` | ndarray | ndarray | HSV → BGR |

### Inspection & histograms

| Function | Input | Output | Description |
|---|---|---|---|
| `image_info(image)` | ndarray | dict | height, width, channels, dtype, bytes |
| `image_stats(image)` | ndarray | dict | per-channel min/max/mean/std |
| `pixel_value(image, x, y)` | ndarray, int, int | scalar/tuple | Single pixel value |
| `histogram(image, channel, bins, ranges)` | ndarray | (ndarray, ndarray) | Histogram + bin centers |
| `histogram_image(image, bins, size)` | ndarray | ndarray | Rendered histogram image |

## Execution flow (demo)

1. Load `sample_images/test_scene.png` (generate it if missing).
2. Print metadata and statistics.
3. Run each transform/conversion and print the resulting shape.
4. With `--show`, display every result in its own window.

```
python src/image_fundamentals/image_demo.py          # report only
python src/image_fundamentals/image_demo.py --show   # + windows
```

## Dependencies

- Python 3.11+
- OpenCV (`cv2`)
- NumPy

## Limitations

- Pixel coordinate conventions are OpenCV's: `(x, y)` = (column, row).
- All operations are in-memory; no lazy evaluation or caching.
- Histogram rendering uses simple bar drawing, not a plotting engine.

## Future extensions

- Batch processing over a directory of images
- EXIF metadata extraction (Pillow)
- Colour-balance and white-point analysis
- `numpy.asarray` compatible lazy loaders for very large images
