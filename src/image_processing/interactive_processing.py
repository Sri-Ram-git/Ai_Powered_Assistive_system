"""Interactive processing playground.

Displays the sample scene with live-adjustable Gaussian blur and Canny
edge thresholds using OpenCV trackbars.

Usage:
    python src/image_processing/interactive_processing.py

Keys:  q  quit    space  reset trackbars
"""
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.image_fundamentals.image_utils import read_image  # noqa: E402
from src.image_processing import processing as P  # noqa: E402

SCENE = PROJECT_ROOT / "src" / "image_fundamentals" / "sample_images" \
    / "test_scene.png"

WINDOW = "Interactive Processing"
BLUR_MAX = 20
CANNY_LOW_MAX = 255
CANNY_HIGH_MAX = 255


def _on_trackbar(*_) -> None:
    pass  # values are read in the loop


def main() -> None:
    scene = read_image(SCENE)
    cv2.namedWindow(WINDOW)

    cv2.createTrackbar("Blur ksize", WINDOW, 1, BLUR_MAX, _on_trackbar)
    cv2.createTrackbar("Canny low", WINDOW, 50, CANNY_LOW_MAX, _on_trackbar)
    cv2.createTrackbar("Canny high", WINDOW, 150, CANNY_HIGH_MAX, _on_trackbar)

    print("Move trackbars; press q to quit, space to reset.")

    while True:
        ksize = cv2.getTrackbarPos("Blur ksize", WINDOW)
        ksize = max(1, ksize | 1)  # force odd, >= 1
        low = cv2.getTrackbarPos("Canny low", WINDOW)
        high = max(low, cv2.getTrackbarPos("Canny high", WINDOW))

        blurred = P.blur_gaussian(scene, ksize)
        edges = P.canny(blurred, low, high)

        display = np.hstack([
            cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY),
            edges,
        ])
        cv2.imshow(WINDOW, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            cv2.setTrackbarPos("Blur ksize", WINDOW, 1)
            cv2.setTrackbarPos("Canny low", WINDOW, 50)
            cv2.setTrackbarPos("Canny high", WINDOW, 150)

    cv2.destroyAllWindows()
    print("Interactive processing closed.")


if __name__ == "__main__":
    main()
