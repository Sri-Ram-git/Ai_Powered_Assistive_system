"""Interactive demo for the Image Fundamentals module.

Usage:
    python src/image_fundamentals/image_demo.py            # print report
    python src/image_fundamentals/image_demo.py --show     # also show windows

The demo generates a synthetic test scene (no personal media needed),
then exercises every image_utils operation and prints a resample of the
results.  With ``--show`` each transformed image is displayed in its own
window.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure the project root is on sys.path so that "src.image_fundamentals"
# resolves.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.image_fundamentals import image_utils as U  # noqa: E402

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_images"
SAMPLE_FILE = SAMPLE_DIR / "test_scene.png"


def create_sample_scene(path: str | Path) -> np.ndarray:
    """Create a synthetic test scene with assorted shapes and colours.

    Args:
        path: Where to save the generated scene image.

    Returns:
        The generated BGR image.
    """
    image = np.full((400, 600, 3), 245, dtype=np.uint8)

    # background gradient
    for i in range(400):
        tint = int(20 + 235 * (i / 399.0))
        cv2.line(image, (0, i), (600, i), (tint // 2, tint, 200), 1)

    # red circle
    cv2.circle(image, (150, 120), 70, (60, 60, 220), -1)
    # green rectangle
    cv2.rectangle(image, (300, 50), (460, 200), (80, 200, 90), -1)
    # blue triangle
    pts = np.array([[120, 300], [60, 390], [180, 390]], np.int32)
    cv2.fillPoly(image, [pts], (220, 140, 80))
    # small white square (target for crop/ROI demos)
    cv2.rectangle(image, (420, 300), (500, 380), (30, 30, 30), -1)
    cv2.putText(image, "sample", (470, 76), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (240, 240, 240), 2)

    U.save_image(image, path)
    return image


def _open(image: np.ndarray, name: str, show: bool) -> None:
    print(f"  {name:<28} shape={image.shape}")
    if show:
        cv2.imshow(name, image)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo the image_fundamentals module."
    )
    parser.add_argument("--show", action="store_true",
                        help="Display each result in a window.")
    parser.add_argument("--input", type=str, default=None,
                        help="Use an external image instead of the "
                             "generated sample scene.")
    args = parser.parse_args()

    if args.input:
        scene = U.read_image(args.input)
        print(f"Loaded external image: {args.input}")
    else:
        if SAMPLE_FILE.is_file():
            scene = U.read_image(SAMPLE_FILE)
            print(f"Loaded existing scene: {SAMPLE_FILE}")
        else:
            scene = create_sample_scene(SAMPLE_FILE)
            print(f"Generated scene: {SAMPLE_FILE}")

    print("\n=== Metadata ===")
    info = U.image_info(scene)
    for key, value in info.items():
        print(f"  {key:<14} {value}")
    stats = U.image_stats(scene)
    print(f"  {'mean':<14} {np.round(stats['mean'], 1)}")

    print("\n=== Demo operations ===")
    _open(scene, "1_original", args.show)

    # transforms
    scaled = U.resize(scene, scale=0.5)
    _open(scaled, "2_resized_half", args.show)
    cropped = U.crop(scene, 420, 300, 80, 80)
    _open(cropped, "3_cropped_roi", args.show)
    flipped = U.flip(scene, flip_code=1)
    _open(flipped, "4_flipped_H", args.show)
    rotated = U.rotate(scene, 30)
    _open(rotated, "5_rotated_30", args.show)

    # colour space
    gray = U.to_grayscale(scene)
    _open(gray, "6_grayscale", args.show)
    hsv = U.to_hsv(scene)
    _open(hsv, "7_hsv", args.show)
    rgb = U.to_rgb(cropped)
    _open(rgb, "8_rgb(crop)", args.show)

    # inspection
    px = U.pixel_value(scene, 150, 120)
    print(f"  pixel @ (150,120)        = {px}")
    hst = U.histogram_image(scene)
    _open(hst, "9_histogram_image", args.show)

    if args.show:
        print("\nPress any key in a window to close all...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("\nImage fundamentals demo complete.")


if __name__ == "__main__":
    main()