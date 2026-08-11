# Image Processing Module

## Overview

Stateless operations for filtering, thresholding, edge detection,
enhancement, and noise handling.  Every function takes an image and
returns a new image; inputs are never mutated.

## Architecture

```
        ┌──────────────────────────────────────────────┐
 ndarray│               processing.py                 │
  ─────▶│  Filters      blur_gaussian / median /       │
        │               bilateral                       │
        │  Threshold    threshold, adaptive_threshold  │
        │  Edges        canny, sobel_x/y/magnitude,    │
        │               laplacian                      │
        │  Enhance      sharpen, brightness, contrast  │
        │  Noise        add_noise, remove_noise        │
        └──────────────────────────────────────────────┘
                          │
                          ▼
                        ndarray        (pure, typed, logged)
```

`processing_demo.py` runs every operation headlessly; `interactive_processing.py` provides live trackbars.

## Function reference

| Function | Input | Output | Notes |
|---|---|---|---|
| `blur_gaussian(image, kernel_size, sigma)` | ndarray, int, float | ndarray | kernel must be odd |
| `blur_median(image, kernel_size)` | ndarray, int | ndarray | kernel must be odd |
| `blur_bilateral(image, d, sc, ss)` | ndarray, int | ndarray | edge-preserving denoise |
| `threshold(image, thresh, maxval, method)` | ndarray | ndarray (2ch) | fixed threshold |
| `adaptive_threshold(image, maxval, block_size, c)` | ndarray | ndarray (2ch) | per-region threshold |
| `canny(image, low, high)` | ndarray | ndarray (2ch) | edge detection |
| `sobel_x / sobel_y / sobel_magnitude(image, ksize)` | ndarray | ndarray (2ch) | normalised to uint8 |
| `laplacian(image, ksize)` | ndarray | ndarray (2ch) | second derivative |
| `sharpen(image, amount)` | ndarray, float | ndarray | unsharp mask |
| `adjust_brightness(image, delta)` | ndarray, float | ndarray | clips to [0,255] |
| `adjust_contrast(image, alpha)` | ndarray, float | ndarray | alpha > 0 |
| `add_noise(image, type, amount)` | ndarray, str, float | ndarray | gaussian / salt_pepper |
| `remove_noise(image, method, kernel_size)` | ndarray, str | ndarray | median / bilateral / gaussian |

## Execution flow (demo)

```
python src/image_processing/processing_demo.py          # report
python src/image_processing/processing_demo.py --show   # windows
python src/image_processing/interactive_processing.py   # trackbars
```

## Dependencies

- Python 3.11+, OpenCV, NumPy
- Depends on `image_fundamentals.image_utils.to_grayscale`

## Limitations

- Grayscale-converting operations (threshold, edges) discard colour.
- Sobel/Laplacian outputs are normalised to [0,255] for display, losing
  sign information (edges are shown as magnitude).
- `interactive_processing.py` runs a blocking window loop (desktop only).

## Future extensions

- Hough line/circle transforms
- Histogram equalisation and CLAHE
- Frequency-domain filtering (DFT)
- Morphological gradient / top-hat transforms
