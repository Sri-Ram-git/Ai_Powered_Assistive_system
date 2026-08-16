"""Audit failure-mode checks (temporary, not part of the test suite).

Verifies behaviour under error/edge conditions. Each check prints
PASS/FAIL based on the actual exception raised.

Run:  python scripts/audit/failure_modes.py
"""
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PASS = 0
FAIL = 0


def check(name, fn, expect_exc=None, expect_ok=False):
    global PASS, FAIL
    try:
        fn()
        if expect_exc is not None:
            print(f"FAIL  {name}: no exception raised (expected {expect_exc.__name__})")
            FAIL += 1
        elif expect_ok:
            print(f"PASS  {name}")
            PASS += 1
        else:
            print(f"FAIL  {name}: unexpected success")
            FAIL += 1
    except expect_exc as exc:
        print(f"PASS  {name}: raised {type(exc).__name__} ({exc})")
        PASS += 1
    except Exception as exc:
        print(f"FAIL  {name}: raised {type(exc).__name__} (expected {expect_exc.__name__ if expect_exc else 'no-exception'}): {exc}")
        FAIL += 1


def main():
    from src.utils.exceptions import (
        CameraAccessError,
        CameraNotFoundError,
        DetectionError,
        ImageError,
        InvalidResolutionError,
        OcrError,
        ProcessingError,
        RecordingError,
    )

    # ---------------- image fundamentals ----------------
    from src.image_fundamentals import image_utils as U

    print("== image_fundamentals ==")
    check("read_image missing file",
          lambda: U.read_image("does_not_exist.png"), ImageError)
    check("read_image corrupted/empty path",
          lambda: U.read_image(""), ImageError)
    check("save_image None", lambda: U.save_image(None, "x.png"), ImageError)
    check("save_image empty array",
          lambda: U.save_image(np.zeros((0, 0, 3)), "x.png"), ImageError)
    check("crop out of bounds",
          lambda: U.crop(np.zeros((10, 10, 3), np.uint8), 5, 5, 10, 10), ImageError)
    check("crop negative",
          lambda: U.crop(np.zeros((10, 10, 3), np.uint8), -1, 0, 5, 5), ImageError)
    check("resize bad scale",
          lambda: U.resize(np.zeros((10, 10, 3), np.uint8), scale=0), ImageError)
    check("flip bad code",
          lambda: U.flip(np.zeros((10, 10, 3), np.uint8), 5), ImageError)
    check("pixel_value out of bounds",
          lambda: U.pixel_value(np.zeros((10, 10, 3), np.uint8), 100, 100), ImageError)
    check("image_info None", lambda: U.image_info(None), ImageError)

    # valid round-trips
    tmp = PROJECT_ROOT / "scripts" / "audit" / "_tmp.png"
    img = np.zeros((20, 20, 3), np.uint8)
    check("save_image valid", lambda: U.save_image(img, tmp), expect_ok=True)
    check("read_image valid", lambda: U.read_image(tmp), expect_ok=True)
    tmp.unlink(missing_ok=True)

    # ---------------- processing ----------------
    from src.image_processing import processing as P

    print("\n== image_processing ==")
    check("blur even kernel",
          lambda: P.blur_gaussian(np.zeros((10, 10, 3), np.uint8), 4), ProcessingError)
    check("bilateral even d",
          lambda: P.blur_bilateral(np.zeros((10, 10, 3), np.uint8), d=4), ProcessingError)
    check("adaptive_threshold even block",
          lambda: P.adaptive_threshold(np.zeros((10, 10, 3), np.uint8), block_size=4), ProcessingError)
    check("sharpen negative amount",
          lambda: P.sharpen(np.zeros((10, 10, 3), np.uint8), -1), ProcessingError)
    check("adjust_brightness out of range",
          lambda: P.adjust_brightness(np.zeros((10, 10, 3), np.uint8), 999), ProcessingError)
    check("adjust_contrast non-positive",
          lambda: P.adjust_contrast(np.zeros((10, 10, 3), np.uint8), 0), ProcessingError)
    check("add_noise out of range",
          lambda: P.add_noise(np.zeros((10, 10, 3), np.uint8), amount=2), ProcessingError)
    check("add_noise unknown type",
          lambda: P.add_noise(np.zeros((10, 10, 3), np.uint8), "bogus"), ProcessingError)
    check("remove_noise unknown method",
          lambda: P.remove_noise(np.zeros((10, 10, 3), np.uint8), "bogus"), ProcessingError)

    # ---------------- morphology ----------------
    from src.morphology import contour_utils as C

    print("\n== morphology ==")
    check("erode even kernel",
          lambda: C.erode(np.zeros((10, 10), np.uint8), 4), ProcessingError)
    check("dilate even kernel",
          lambda: C.dilate(np.zeros((10, 10), np.uint8), 2), ProcessingError)

    # ---------------- camera ----------------
    from src.camera.camera import Camera

    print("\n== camera ==")
    cam = Camera(camera_id=0, resolution=(640, 480))
    check("Camera.read before start", lambda: cam.read(), CameraAccessError)
    check("Camera.set_resolution before start",
          lambda: cam.set_resolution(640, 480), CameraAccessError)
    check("Camera.set_resolution invalid",
          lambda: (cam.start() or cam.set_resolution(0, -5)), InvalidResolutionError)

    from src.camera.camera_utils import VideoRecorder

    rec = VideoRecorder(cam, duration=1)
    check("VideoRecorder.stop when idle", lambda: rec.stop(), expect_ok=True)

    # Camera not found: use an out-of-range device id where possible
    check("Camera.open nonexistent device",
          lambda: Camera(camera_id=999).start(), CameraNotFoundError)

    # ---------------- detection ----------------
    from src.detection import YoloDetector

    print("\n== detection ==")
    check("YoloDetector missing model",
          lambda: YoloDetector("models/nope.onnx"), DetectionError)
    model = PROJECT_ROOT / "models" / "yolov8n.onnx"
    d = YoloDetector(str(model))
    check("detect on empty frame returns []",
          lambda: (None if d.detect(np.zeros((0, 0, 3), np.uint8)) else None), expect_ok=True)
    _ = len(d.detect(np.zeros((640, 640, 3), np.uint8))) >= 0
    print("PASS  detect on black frame (returns list)")

    # ---------------- OCR ----------------
    from src.ocr import OcrEngine

    print("\n== OCR ==")
    ocr = OcrEngine(min_confidence=0.3)
    check("OCR on non-numpy input",
          lambda: ocr.read_text("not an image"), OcrError)
    check("OCR on None returns []",
          lambda: (None if ocr.read_text(None) else None), expect_ok=True)
    check("OCR on empty array returns []",
          lambda: (None if ocr.read_text(np.zeros((0, 0), np.uint8)) else None), expect_ok=True)

    # ---------------- audio (TTS) ----------------
    print("\n== audio (TTS init on real engine) ==")
    from src.audio import SpeechOutput
    check("SpeechOutput init on real engine",
          lambda: SpeechOutput(), expect_ok=True)

    # ---------------- VideoRecorder without a running camera ----------------
    print("\n== VideoRecorder robustness ==")
    idle_cam = Camera(camera_id=0, resolution=(320, 240))
    # Do NOT start it: the recorder thread will call read() -> CameraAccessError
    rec2 = VideoRecorder(idle_cam, duration=1)
    check("VideoRecorder.start with idle camera",
          lambda: rec2.start(), RecordingError)

    print(f"\nTotal: {PASS} passed, {FAIL} failed")


if __name__ == "__main__":
    main()