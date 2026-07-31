"""Interactive test for the Camera module.

Runs the camera in fullscreen with the best supported resolution and a
minimal professional HUD overlay.

Controls:
    q  — quit
    s  — take screenshot (toast confirms where it was saved)
    r  — record a 5-second video clip (UI keeps running with a REC
         indicator; a toast confirms when the file is saved)
"""
import argparse
import sys
from pathlib import Path

import cv2

# Ensure the project root is on sys.path so that "src.camera" resolves
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera import (
    Camera,
    CameraManager,
    HUD,
    VideoRecorder,
    auto_select_resolution,
    get_screen_size,
    open_fullscreen_window,
    scale_to_fit,
    take_screenshot,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the Camera module")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=None, help="Frame width (auto if omitted)")
    parser.add_argument("--height", type=int, default=None, help="Frame height (auto if omitted)")
    args = parser.parse_args()

    # 1. Discover cameras
    manager = CameraManager()
    cameras = manager.list_cameras()
    print(f"Available cameras: {[c.id for c in cameras]}")
    if not cameras:
        print("ERROR: No cameras found. Exiting.")
        sys.exit(1)

    # 2. Open the camera at its best supported resolution
    screen_w, screen_h = get_screen_size()
    with Camera(camera_id=args.camera, resolution=(640, 480)) as cam:
        if args.width and args.height:
            cam.set_resolution(args.width, args.height)
        else:
            auto_select_resolution(cam)
        print(f"Camera {args.camera} started | resolution={cam.resolution}")
        print(f"Screen: {screen_w}x{screen_h} | font: Segoe UI")
        print("Controls:  [s] screenshot  [r] record  [q] quit")

        window = "Assistive Vision"
        open_fullscreen_window(window)
        hud = HUD()

        recorder: VideoRecorder | None = None

        while True:
            # During recording the recorder thread owns frame reads, so
            # display its latest frame instead of reading twice.
            if recorder is not None and recorder.is_recording:
                frame = recorder.latest_frame
                if frame is None:
                    continue
            else:
                frame = cam.read()

            hud.tick(cam.actual_fps)
            display = scale_to_fit(frame, screen_w, screen_h)
            display = hud.render(display, camera=cam, mode="LIVE", status="")

            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                if recorder is not None:
                    recorder.stop()
                print("Quitting...")
                break
            elif key == ord("s"):
                path = take_screenshot(frame)
                hud.show_toast(f"Screenshot saved: {Path(path).name}")
                print(f"Screenshot saved: {path}")
            elif key == ord("r") and (
                recorder is None or not recorder.is_recording
            ):
                recorder = VideoRecorder(cam, duration=5)
                recorder.start()
                hud.set_recording(True)
                hud.show_toast("Recording...")
                print("Recording started...")

            # Recording finished in the background: confirm with a toast
            if (
                recorder is not None
                and not recorder.is_recording
                and recorder.saved_path is not None
            ):
                hud.set_recording(False)
                hud.show_toast(f"Saved: {Path(recorder.saved_path).name}")
                print(f"Recording saved: {recorder.saved_path}")
                recorder = None

    cv2.destroyAllWindows()
    print("Camera test complete.")


if __name__ == "__main__":
    main()
