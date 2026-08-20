"""Demo for morphology & shape analysis.

Draws synthetic shapes, applies morphological operations, then detects
and annotates circles/rectangles/triangles.

Usage:
    python src/morphology/demo.py [--show]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.morphology import contour_utils as C  # noqa: E402
from src.morphology.shape_detector import (  # noqa: E402
    ShapeDetector,
    label_image,
)


def build_scene() -> np.ndarray:
    """Synthetic binary scene with circle, rectangle, and triangle."""
    scene = np.zeros((400, 600), dtype=np.uint8)
    cv2.circle(scene, (140, 130), 60, 255, -1)
    cv2.rectangle(scene, (300, 60), (450, 210), 255, -1)
    pts = np.array([[120, 320], [60, 400], [180, 400]], np.int32)
    cv2.fillPoly(scene, [pts], 255)
    # noise speckle
    np.random.seed(0)
    mask = np.random.random(scene.shape) < 0.01
    scene[mask] = 255
    return scene


def main() -> None:
    parser = argparse.ArgumentParser(description="Morphology & shape demo")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    binary = build_scene()
    print(f"Binary scene: {binary.shape}")

    print("\n=== Morphological operations ===")
    steps = {
        "erode": C.erode(binary, 3),
        "dilate": C.dilate(binary, 3),
        "opening": C.opening(binary, 3),
        "closing": C.closing(binary, 3),
    }
    for name, result in steps.items():
        diff = int(np.count_nonzero(binary != result))
        print(f"  {name:<8} pixels changed: {diff}")

    # Use opening to clean the noise, then detect
    cleaned = C.opening(binary, 3)

    print("\n=== Contours ===")
    contours, _ = C.find_contours(cleaned)
    print(C.contours_report(contours))

    print("\n=== Shape detection ===")
    detector = ShapeDetector(min_area=200)
    results = detector.detect(cleaned)
    for r in results:
        print(f"  {r.shape:<10} area={r.area:7.1f} "
              f"bbox={r.bounding_box}")

    if args.show:
        color = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        cv2.imshow("shapes", label_image(color, results))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("\nMorphology demo complete.")


if __name__ == "__main__":
    main()
