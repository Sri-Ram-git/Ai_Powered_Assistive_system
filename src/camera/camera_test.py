"""Interactive test for the Camera module.

Run directly to verify camera initialisation, frame capture, overlays,
screenshot capture, and graceful shutdown.

Controls:
    q  — quit
    s  — take screenshot
"""
import argparse
import sys
from pathlib import Path

import cv2

# Ensure the project root is on sys.path so that "src.camera" resolves
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera import Camera, CameraManager, draw_fps, draw_timestamp, take_screenshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the Camera module")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    args = parser.parse_args()

    # 1. Discover cameras
    manager = CameraManager()
    cameras = manager.list_cameras()
    print(f"Available cameras: {[c.id for c in cameras]}")
    if not cameras:
        print("ERROR: No cameras found. Exiting.")
        sys.exit(1)

    # 2. Open and stream
    with Camera(camera_id=args.camera, resolution=(args.width, args.height)) as cam:
        print(f"Camera {args.camera} started | resolution={cam.resolution}")
        print("Controls:  [s] screenshot  [q] quit")

        while True:
            frame = cam.read()

            display = draw_fps(frame.copy(), cam.actual_fps)
            display = draw_timestamp(display)
            cv2.putText(
                display,
                f"Res: {cam.resolution[0]}x{cam.resolution[1]}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

            cv2.imshow("Camera Test", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quitting...")
                break
            elif key == ord("s"):
                path = take_screenshot(frame)
                print(f"Screenshot saved: {path}")

    cv2.destroyAllWindows()
    print("Camera test complete.")


if __name__ == "__main__":
    main()
