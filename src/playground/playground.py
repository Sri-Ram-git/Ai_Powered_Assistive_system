"""Live Vision Playground — the Week 1 integration application.

Opens the webcam and lets the user switch processing filters, toggle
edge detection / grayscale / threshold, record, and save frames — all
from the keyboard.  The menu bar and dashboard are draggable.

Usage:
    python src/playground/playground.py [--camera 0]

Keys:
    1-7        select base filter (original/gaussian/median/bilateral/
               sharpen/sobel/laplacian)
    g          toggle grayscale
    e          toggle edge detection (Canny)
    t          toggle threshold (binary)
    s          save screenshot (raw frame)
    v          save processed image
    r          record 5-second clip
    space      reset all toggles
    q          quit

Mouse: drag the menu bar and dashboard anywhere.
"""
import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera import (  # noqa: E402
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
from src.image_processing import processing as P  # noqa: E402


# Filter id -> (name, function(frame)->frame)
FILTERS = {
    ord("1"): ("original", lambda f: f),
    ord("2"): ("gaussian", lambda f: P.blur_gaussian(f, 5)),
    ord("3"): ("median", lambda f: P.blur_median(f, 5)),
    ord("4"): ("bilateral", lambda f: P.blur_bilateral(f)),
    ord("5"): ("sharpen", lambda f: P.sharpen(f, 0.8)),
    ord("6"): ("sobel", lambda f: P.sobel_magnitude(f)),
    ord("7"): ("laplacian", lambda f: P.laplacian(f)),
}


def _make_drag_handler(hud, canvas_w: int, canvas_h: int):
    drag = {"widget": None, "offset": (0, 0)}

    def on_mouse(event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            widget = hud.hit_test(x, y, canvas_w, canvas_h)
            if widget is not None:
                rx, ry, _, _ = hud.widget_rect(widget, canvas_w, canvas_h)
                drag["widget"] = widget
                drag["offset"] = (x - rx, y - ry)
        elif event == cv2.EVENT_MOUSEMOVE and drag["widget"] is not None:
            ox, oy = drag["offset"]
            hud.set_widget_pos(drag["widget"], x - ox, y - oy,
                               canvas_w, canvas_h)
        elif event == cv2.EVENT_LBUTTONUP:
            drag["widget"] = None

    return on_mouse


def build_mode(filter_name: str, gray: bool, edge: bool, thresh: bool) -> str:
    """Compose the HUD mode string from the active pipeline stages."""
    stages = [filter_name.upper()]
    if gray:
        stages.append("GRAY")
    if edge:
        stages.append("EDGE")
    if thresh:
        stages.append("THRESH")
    return " | ".join(stages)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Vision Playground")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index")
    args = parser.parse_args()

    manager = CameraManager()
    cameras = manager.list_cameras()
    print(f"Available cameras: {[c.id for c in cameras]}")
    if not cameras:
        print("ERROR: No cameras found. Exiting.")
        sys.exit(1)

    screen_w, screen_h = get_screen_size()
    with Camera(camera_id=args.camera, resolution=(640, 480)) as cam:
        auto_select_resolution(cam)
        print(f"Camera {args.camera} | resolution={cam.resolution}")

        window = "Vision Playground"
        open_fullscreen_window(window)
        hud = HUD()
        cv2.setMouseCallback(
            window, _make_drag_handler(hud, screen_w, screen_h), hud,
        )
        hud.show_toast("1-7 filter | g gray | e edge | t thresh | s/v save")

        state = {
            "filter": ord("1"),
            "gray": False,
            "edge": False,
            "thresh": False,
        }
        recorder: VideoRecorder | None = None

        while True:
            if recorder is not None and recorder.is_recording:
                frame = recorder.latest_frame
                if frame is None:
                    continue
            else:
                frame = cam.read()

            # ---- processing pipeline ----
            filter_name, func = FILTERS[state["filter"]]
            processed = func(frame)
            if state["gray"]:
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY) \
                    if processed.ndim == 3 else processed
            if state["edge"]:
                processed = P.canny(processed, 60, 160)
            if state["thresh"]:
                processed = P.threshold(processed, 127)

            # HUD requires a BGR image
            display = scale_to_fit(frame, screen_w, screen_h)
            hud.tick(cam.actual_fps)
            display = hud.render(
                display, camera=cam,
                mode=build_mode(filter_name, state["gray"],
                                state["edge"], state["thresh"]),
            )
            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                if recorder is not None:
                    recorder.stop()
                break
            elif key in FILTERS:
                state["filter"] = key
            elif key == ord("g"):
                state["gray"] = not state["gray"]
            elif key == ord("e"):
                state["edge"] = not state["edge"]
            elif key == ord("t"):
                state["thresh"] = not state["thresh"]
            elif key == ord(" "):
                state.update(filter=ord("1"), gray=False,
                             edge=False, thresh=False)
            elif key == ord("s"):
                path = take_screenshot(frame)
                hud.show_toast(f"Screenshot: {Path(path).name}")
            elif key == ord("v"):
                path = take_screenshot(processed,
                                       "assets/screenshots/processed")
                hud.show_toast(f"Processed: {Path(path).name}")
            elif key == ord("r") and (
                recorder is None or not recorder.is_recording
            ):
                recorder = VideoRecorder(cam, duration=5)
                recorder.start()
                hud.set_recording(True)
                hud.show_toast("Recording...")

            if (
                recorder is not None
                and not recorder.is_recording
                and recorder.saved_path is not None
            ):
                hud.set_recording(False)
                hud.show_toast(f"Saved: {Path(recorder.saved_path).name}")
                recorder = None

    cv2.destroyAllWindows()
    print("Playground closed.")


if __name__ == "__main__":
    main()
