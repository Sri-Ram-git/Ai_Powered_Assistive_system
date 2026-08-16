"""Assistive Vision App — end-to-end Week 2-3 integration.

Live pipeline: Camera → Object Detection (YOLOv8 ONNX) → Tracking (IoU
association) → OCR (RapidOCR) → Decision Engine + Tracking Monitor →
Speech Output, with tracked boxes, stable IDs, and per-object distance
drawn on a draggable HUD.

The tracker gives each object a stable ID and the monitor re-announces
the distance as the object moves, so guidance is continuous — not a
one-off.

Usage:
    python src/assist/assist_app.py [--camera 0] [--config configs/assist_config.yaml]

Keys:
    m          mute / unmute speech
    t          toggle OCR mode (auto-read vs ask-before-read)
    r          read the most recently detected text
    s          save annotated screenshot
    space      reset tracking + decision cooldown
    q          quit
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio import SpeechOutput  # noqa: E402
from src.camera import (  # noqa: E402
    Camera,
    CameraManager,
    HUD,
    get_screen_size,
    open_fullscreen_window,
    scale_to_fit,
    take_screenshot,
)
from src.decision import DecisionEngine, FrameSummary  # noqa: E402
from src.detection import YoloDetector  # noqa: E402
from src.ocr import OcrEngine, draw_text_boxes  # noqa: E402
from src.tracking import IoUTracker, TrackingMonitor  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402

_logger = setup_logger("AssistApp")


def load_config(path: str) -> dict:
    """Load the YAML assist configuration."""
    config_path = Path(path)
    if not config_path.exists():
        _logger.warning("Config not found: %s — using defaults", path)
        return {}
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def draw_tracks(
    frame,
    tracks,
    ocr_items,
    frame_h,
    vfov_deg: float = 55.0,
):
    """Draw tracked boxes with stable IDs + distance, and OCR text."""
    display = frame.copy()
    for track in tracks:
        x, y, w, h = track.box
        color = _track_color(track.track_id)
        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
        dist = track_distance(track, frame_h, vfov_deg)
        text = f"#{track.track_id} {track.label} {dist}"
        label_y = y - 8 if y - 8 > 10 else y + h + 18
        cv2.putText(display, text, (x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return draw_text_boxes(display, ocr_items)


def track_distance(track, frame_h: int, vfov_deg: float = 55.0) -> str:
    from src.navigation.guidance import (
        distance_estimate,
        reference_height,
    )

    d = distance_estimate(track.box, frame_h,
                          reference_height(track.label),
                          vfov_deg=vfov_deg)
    if d >= 15.0:
        return "far"
    if d <= 0.5:
        return "close"
    return f"{d:.0f}m"


def _apply_reference_heights(overrides: dict) -> None:
    """Push optional per-class height overrides into the distance model."""
    if not overrides:
        return
    from src.navigation import guidance as _g

    _g._REFERENCE_HEIGHTS.update({
        str(k): float(v) for k, v in overrides.items()
    })


def _track_color(track_id: int):
    palette = [
        (0, 255, 0), (0, 165, 255), (255, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 0),
    ]
    return palette[track_id % len(palette)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Assistive Vision App")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--config", default=str(
        PROJECT_ROOT / "configs" / "assist_config.yaml"))
    parser.add_argument("--model", default=None,
                        help="Override the YOLO ONNX model path")
    args = parser.parse_args()

    cfg = load_config(args.config)

    det_cfg = cfg.get("detection", {})
    ocr_cfg = cfg.get("ocr", {})
    speech_cfg = cfg.get("speech", {})
    decision_cfg = cfg.get("decision", {})
    tracking_cfg = cfg.get("tracking", {})
    cam_cfg = cfg.get("camera", {})
    app_cfg = cfg.get("app", {})
    nav_cfg = cfg.get("navigation", {})

    # Distance model knobs: vertical FOV + optional reference-height
    # overrides make the pinhole distance estimate accurate to your
    # actual webcam.
    vfov_deg = float(nav_cfg.get("vertical_fov", 55.0))
    height_overrides = dict(nav_cfg.get("reference_heights", {}) or {})
    _apply_reference_heights(height_overrides)

    # OCR behaviour: "ask" mode extracts text and prompts the user to
    # press R to hear it; "auto" mode reads it aloud immediately.
    ask_before_reading = bool(
        ocr_cfg.get("ask_before_reading",
                    decision_cfg.get("ask_before_reading", False)))

    camera_id = args.camera if args.camera is not None else cam_cfg.get("id", 0)
    model_path = args.model or det_cfg.get("model_path", "models/yolov8n.onnx")
    if not Path(model_path).is_absolute():
        model_path = str(PROJECT_ROOT / model_path)
    detect_every = max(1, int(det_cfg.get("every_n_frames", 2)))
    ocr_every = max(1, int(ocr_cfg.get("every_n_frames", 10)))

    manager = CameraManager()
    cameras = manager.list_cameras()
    print(f"Available cameras: {[c.id for c in cameras]}")
    if not cameras:
        print("ERROR: No cameras found. Exiting.")
        sys.exit(1)

    _logger.info("Loading detection model %s ...", model_path)
    detector = YoloDetector(
        model_path,
        input_size=det_cfg.get("input_size", 640),
        conf_threshold=det_cfg.get("conf_threshold", 0.35),
        iou_threshold=det_cfg.get("iou_threshold", 0.45),
    )
    ocr = OcrEngine(
        min_confidence=ocr_cfg.get("min_confidence", 0.3),
        max_boxes=ocr_cfg.get("max_boxes", 50),
    )
    tts = SpeechOutput(
        rate=speech_cfg.get("rate", 165),
        volume=speech_cfg.get("volume", 1.0),
    )
    engine = DecisionEngine(
        cooldown_seconds=decision_cfg.get("cooldown_seconds", 4.0),
        min_priority=decision_cfg.get("min_priority", 5),
        read_ocr_text=decision_cfg.get("speak_ocr_text", True),
        max_ocr_chars=decision_cfg.get("max_ocr_chars", 80),
    )
    tracker = IoUTracker(
        iou_threshold=tracking_cfg.get("iou_threshold", 0.3),
        max_missed=tracking_cfg.get("max_missed", 8),
    )
    monitor = TrackingMonitor(
        distance_change_metres=tracking_cfg.get("distance_change_metres", 1.0),
        min_announce_interval=tracking_cfg.get("min_announce_interval", 3.0),
        vfov_deg=vfov_deg,
    )

    screen_w, screen_h = get_screen_size()
    res = tuple(cam_cfg.get("resolution", [1280, 720]))
    with Camera(camera_id=camera_id, resolution=res) as cam:
        print(f"Camera {camera_id} | resolution={cam.resolution}")

        window = "Assistive Vision"
        open_fullscreen_window(window)
        hud = HUD()
        hud.show_toast("m mute | t ocr mode | r read text | s save | space reset | q quit")

        muted = False
        frame_index = 0
        last_ocr: list = []
        tracks: list = []
        pending_text: str = ""       # most recently extracted text
        last_prompted_text: str = ""
        ask_read = ask_before_reading

        while True:
            frame = cam.read()
            frame_index += 1

            # ---- object detection (throttled) ----
            if frame_index % detect_every == 0:
                detections = detector.detect(frame)
                tracks = tracker.update(detections)
            # ---- OCR (throttled for CPU) ----
            if frame_index % ocr_every == 0:
                try:
                    last_ocr = ocr.read_text(frame)
                except Exception as exc:  # pragma: no cover - env dependent
                    _logger.warning("OCR failed: %s", exc)
                    last_ocr = []
                pending_text = " ".join(r.text for r in last_ocr).strip()

            # ---- continuous tracking guidance ----
            phrases = monitor.events(
                tracks, frame.shape[1], frame.shape[0],
            )
            for phrase in phrases:
                if not muted:
                    tts.speak(phrase)

            # ---- OCR: ask-before-read prompt ----
            if pending_text and ask_read:
                # Tell the user text was found; they press R to hear it.
                if pending_text != last_prompted_text:
                    if not muted:
                        tts.speak("Text detected. Press R to hear it read.")
                    last_prompted_text = pending_text
            elif not pending_text:
                last_prompted_text = ""

            # ---- decision engine (signs / obstacles / auto-read OCR) ----
            summary = FrameSummary(
                detections=[t for t in tracks],
                ocr_items=last_ocr if not ask_read else [],
                frame_w=frame.shape[1],
                frame_h=frame.shape[0],
            )
            phrase = engine.decide(summary, already_spoken=phrases)
            if phrase and not muted:
                tts.speak(phrase)

            # ---- annotation + HUD ----
            display = draw_tracks(frame, tracks, last_ocr,
                                  frame.shape[0], vfov_deg)
            display = scale_to_fit(display, screen_w, screen_h)
            ocr_text = pending_text[:40] if pending_text else ""
            mode = "ASK" if ask_read else "READ"
            status = "MUTED" if muted else (
                f"{len(tracks)} tracked | OCR:{mode}"
                + (f" | {ocr_text}" if ocr_text else "")
            )
            hud.tick(cam.actual_fps)
            display = hud.render(display, camera=cam,
                                 mode="ASSIST", status=status)
            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                muted = not muted
                hud.show_toast("Speech muted" if muted else "Speech on")
            elif key == ord("t"):
                ask_read = not ask_read
                hud.show_toast(
                    "OCR: ask before reading" if ask_read else
                    "OCR: read text aloud"
                )
            elif key == ord("r"):
                if pending_text:
                    text = pending_text[:80]
                    if not muted:
                        tts.speak(f"Text says, {text}")
                    hud.show_toast(f"Text: {text}")
                else:
                    hud.show_toast("No text detected yet")
            elif key == ord("s"):
                path = take_screenshot(display, "assets/screenshots/assist")
                hud.show_toast(f"Screenshot: {Path(path).name}")
            elif key == ord(" "):
                engine.reset()
                tracker.reset()
                monitor.reset()
                hud.show_toast("Tracking + cooldown reset")

    cv2.destroyAllWindows()
    tts.shutdown()
    print("Assistive Vision app closed.")


if __name__ == "__main__":
    main()
