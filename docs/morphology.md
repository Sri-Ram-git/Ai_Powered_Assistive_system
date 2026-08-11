# Morphology & Shape Analysis Module

## Overview

Morphological operations for binary images plus contour geometry and
geometric shape classification (circle / rectangle / triangle).

## Architecture

```
          ┌──────────────────────────────────────────────┐
 ndarray  │                morphology/                    │
 ───────▶ │  contour_utils.py                             │
          │    erode, dilate, opening, closing            │
          │    find_contours, area, perimeter, bbox,      │
          │    convex_hull, centroid, draw, report        │
          │                                              │
          │  shape_detector.py                            │
          │    ShapeDetector.detect() → List[ShapeResult] │
          │    classify_shape(), label_image()            │
          └──────────────────────────────────────────────┘
                          │
                          ▼
              labels / annotated image
```

## Detection algorithm

1. Convert to binary (threshold colour input).
2. `findContours` (external, simple approximation).
3. Filter contours by `min_area`.
4. Approximate each contour with `approxPolyDP` (~2% of perimeter).
5. Classify: 3 vertices → triangle; 4 → rectangle; ≥5 with high
   circularity (`4π·area/perimeter²`) → circle; otherwise `polygon-N`.

## Function reference

### contour_utils

| Function | Description |
|---|---|
| `erode / dilate / opening / closing(image, kernel_size, iterations)` | Morphological ops |
| `morph_open / morph_close` | Aliases |
| `find_contours(image, mode, method)` | → (contours, hierarchy) |
| `contour_area / contour_perimeter(contour)` | Geometry metrics |
| `bounding_rect(contour)` | (x, y, w, h) |
| `convex_hull(contour)` | Hull points |
| `convexity_defects(contour)` | Defect indices or None |
| `center_of_mass(contour)` | Centroid (x, y) |
| `draw_contours(image, contours, ...)` | Annotated copy |
| `contours_report(contours)` | Human-readable summary |

### shape_detector

| Member | Description |
|---|---|
| `ShapeDetector(min_area, approx_epsilon_ratio, circularity_threshold)` | Constructor |
| `detect(image)` | → `List[ShapeResult]`, largest first |
| `classify_shape(contour, vertices)` | → shape name |
| `label_image(image, results)` | Annotated copy |

`ShapeResult`: `shape, contour, area, perimeter, bounding_box, center, vertices`.

## Execution flow (demo)

```
python src/morphology/demo.py            # report
python src/morphology/demo.py --show     # annotated image
```

The demo builds a synthetic binary scene (circle + rectangle + triangle
+ noise), cleans it with opening, lists contours, then detects shapes.

## Dependencies

- Python 3.11+, OpenCV, NumPy
- Depends on `image_processing` (thresholding) and `image_fundamentals`

## Limitations

- Shape classification is heuristic and assumes clean silhouettes;
  overlapping/occluded shapes may be merged or misclassified.
- Circles are approximated by vertex count; large polygons with high
  circularity are also classed as circles.
- Morphological ops assume binary (or near-binary) input.

## Future extensions

- Hough circle detection for noisy/deformed circles
- Minimum bounding (rotated) rectangles
- Shape matching against templates (Hu moments)
- Polygon regularisation and corner refinement
