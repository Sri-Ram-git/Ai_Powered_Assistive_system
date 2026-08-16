"""Processing demo: runs every operation on the sample scene and prints
the output shape, plus optional window display.

Usage:
    python src/image_processing/processing_demo.py
    python src/image_processing/processing_demo.py --show
"""
import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.image_fundamentals.image_utils import read_image  # noqa: E402
from src.image_processing import processing as P  # noqa: E402

SCENE = PROJECT_ROOT / "src" / "image_fundamentals" / "sample_images" \
    / "test_scene.png"


def _run(image, label: str, result, show: bool) -> None:
    print(f"  {label:<24} -> {result.shape}")
    if show:
        cv2.imshow(label, result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo image processing")
    parser.add_argument("--show", action="store_true",
                        help="Show each result in a window.")
    parser.add_argument("--input", type=str, default=None)
    args = parser.parse_args()

    scene = read_image(args.input) if args.input else read_image(SCENE)
    print(f"Loaded: {scene.shape}")

    print("\n=== Filters ===")
    _run(scene, "gaussian_blur", P.blur_gaussian(scene), args.show)
    _run(scene, "median_blur", P.blur_median(scene), args.show)
    _run(scene, "bilateral", P.blur_bilateral(scene), args.show)

    print("\n=== Thresholding ===")
    _run(scene, "threshold", P.threshold(scene), args.show)
    _run(scene, "adaptive_threshold",
         P.adaptive_threshold(scene), args.show)

    print("\n=== Edge detection ===")
    _run(scene, "canny", P.canny(scene), args.show)
    _run(scene, "sobel_x", P.sobel_x(scene), args.show)
    _run(scene, "sobel_y", P.sobel_y(scene), args.show)
    _run(scene, "sobel_magnitude", P.sobel_magnitude(scene), args.show)
    _run(scene, "laplacian", P.laplacian(scene), args.show)

    print("\n=== Enhance ===")
    _run(scene, "sharpen", P.sharpen(scene), args.show)
    _run(scene, "brightness+40", P.adjust_brightness(scene, 40), args.show)
    _run(scene, "contrast x1.5", P.adjust_contrast(scene, 1.5), args.show)

    print("\n=== Noise ===")
    _run(scene, "gaussian_noise", P.add_noise(scene, "gaussian"), args.show)
    _run(scene, "s&p_noise", P.add_noise(scene, "salt_pepper"), args.show)
    _run(scene, "median_denoise",
         P.remove_noise(P.add_noise(scene, "salt_pepper")), args.show)

    if args.show:
        print("\nPress any key to close all windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print("\nProcessing demo complete.")


if __name__ == "__main__":
    main()
