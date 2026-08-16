"""Assistive Vision App — real-time desktop interface (Phase 6+).

Runs the async engine (``src.core.AsyncVisionPipeline``) so no slow
stage can freeze the view:

    grab thread    camera → latest-frame store (never blocks)
    detect thread  YOLO → hardened IoU tracker (detect every N frames,
                   track at its own rate, latest results published)
    speech         engine phrases → SpeechQueue → non-blocking TTS

This app is a thin display layer: it shows the *latest* camera frame
with the *latest* tracking overlay at camera FPS, and never blocks on
detection, tracking, OCR, or speech.  OCR is disabled by default
(Phase 21); the worker only loads when ``ocr.enabled: true``.

Usage:
    python src/assist/assist_app.py [--camera 0] [--config configs/assist_config.yaml]

Keys:
    m          mute / unmute speech
    d          toggle debug overlay (fps, latency, raw vs smoothed boxes)
    r          read the most recently detected text (needs ocr.enabled)
    s          save annotated screenshot
    space      reset tracking + decision + speech memory
    q          quit
"""
import argparse
import sys
from pathlib import Path

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio import SpeechOutput  # noqa: E402
from src.audio.speech_queue import SpeechQueue, SpeechTier  # noqa: E402
from src.camera import (  # noqa: E402
    HUD,
    get_screen_size,
    open_fullscreen_window,
    scale_to_fit,
    take_screenshot,
)
from src.core.config import PipelineConfig  # noqa: E402
from src.core.pipeline import AsyncVisionPipeline  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402

_logger = setup_logger("AssistApp")


class _CameraInfo:
    """Lightweight camera facade for the HUD (id + resolution only)."""

    def __init__(self, camera_id: int, resolution) -> None:
        self.camera_id = camera_id
        self.resolution = resolution


class _Mute:
    """Mutable boolean so the speech callback sees the live value."""

    def __init__(self) -> None:
        self.on = False

    def __bool__(self) -> bool:
        return self.on


def draw_tracks(frame, tracks, frame_h, vfov_deg: float = 55.0,
                debug: bool = False):
    """Draw tracked boxes with stable IDs + distance (smooth == solid).

    In debug mode the raw (un-smoothed) box from the last detection is
    drawn in cyan so box smoothing is visible.
    """
    display = frame.copy()
    for track in tracks:
        x, y, w, h = track.box
        color = _track_color(track.track_id)
        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)

        if debug:
            rx, ry, rw, rh = track.raw_box
            cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh),
                          (255, 255, 0), 1)

        dist = track_distance(track, frame_h, vfov_deg)
        conf = f"{track.confidence:.2f}" if debug else ""
        text = f"#{track.track_id} {track.label} {dist}{conf}"
        label_y = y - 8 if y - 8 > 10 else y + h + 18
        cv2.putText(display, text, (x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return display


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


def _debug_overlay(display, state, results, extra: str = ""):
    """Draw a small diagnostics panel (Phase 23)."""
    lines = [
        f"cam {state.get('fps', 0.0):.1f} fps",
        f"yolo {results['latencies'].get('yolo_ms', 0.0):.1f} ms",
        f"tracks {len(results['tracks'])} | pending speech {extra}",
    ]
    y = 8
    for line in lines:
        cv2.putText(display, line, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        y += 22
    return display


def _speech_tier_for(text: str) -> SpeechTier:
    """Map engine output to a speech tier (safety phrases are critical)."""
    lowered = (text or "").lower()
    if "stop" in lowered or "emergency" in lowered:
        return SpeechTier.CRITICAL
    if "turn" in lowered or "now" in lowered:
        return SpeechTier.HIGH
    return SpeechTier.NORMAL


def main() -> None:
    parser = argparse.ArgumentParser(description="Assistive Vision App")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--config", default=str(
        PROJECT_ROOT / "configs" / "assist_config.yaml"))
    parser.add_argument("--model", default=None,
                        help="Override the YOLO ONNX model path")
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    if args.camera is not None:
        cfg.camera_id = args.camera
    if args.model:
        cfg.model_path = args.model
    if not Path(cfg.model_path).is_absolute():
        cfg.model_path = str(PROJECT_ROOT / cfg.model_path)

    # Desktop path: no JPEG encode (web only), OCR off unless configured.
    cfg.encode_jpeg = False

    raw_cfg = {}
    if Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as fh:
            raw_cfg = yaml.safe_load(fh) or {}
    speech_cfg = raw_cfg.get("speech", {}) or {}

    tts = SpeechOutput(
        rate=speech_cfg.get("rate", 165),
        volume=speech_cfg.get("volume", 1.0),
    )
    queue = SpeechQueue(
        tts,
        min_interval=speech_cfg.get("min_interval", 1.2),
        dedupe_window=speech_cfg.get("dedupe_window", 4.0),
    )
    queue.start()
    mute = _Mute()

    def speak(text: str) -> None:
        if not mute:
            queue.enqueue(text, tier=_speech_tier_for(text))

    pipe = AsyncVisionPipeline(config=cfg, speech_callback=speak)
    pipe.start(timeout=8.0)
    state = pipe.state_snapshot()
    if not state.get("running"):
        print(f"ERROR: pipeline failed to start: {state.get('error')}")
        queue.shutdown()
        tts.shutdown()
        sys.exit(1)

    print(f"Pipeline running | camera {cfg.camera_id} | "
          f"model {Path(cfg.model_path).name} | OCR "
          f"{'enabled' if cfg.ocr_enabled else 'disabled'}")

    screen_w, screen_h = get_screen_size()
    window = "Assistive Vision"
    open_fullscreen_window(window)
    hud = HUD()
    hud.show_toast("m mute | d debug | r read | s save | space reset | q quit")

    debug = False
    cam_info = _CameraInfo(cfg.camera_id, cfg.camera_resolution)

    try:
        while True:
            frame = pipe.latest_frame
            if frame is None:
                cv2.waitKey(20)
                continue

            results = pipe.latest_results.snapshot()
            state = pipe.state_snapshot()

            display = draw_tracks(frame, results["tracks"],
                                  frame.shape[0], cfg.vfov_deg,
                                  debug=debug)
            display = scale_to_fit(display, screen_w, screen_h)
            if debug:
                display = _debug_overlay(display, state, results)

            ocr_text = state.get("ocr_text") or ""
            status = "MUTED" if mute else (
                f"{len(results['tracks'])} tracked"
                + (f" | {ocr_text[:40]}" if ocr_text else "")
            )
            hud.tick(state.get("fps", 0.0))
            display = hud.render(display, camera=cam_info,
                                 mode="LIVE", status=status)
            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                mute.on = not mute.on
                hud.show_toast("Speech muted" if mute else "Speech on")
            elif key == ord("d"):
                debug = not debug
                hud.show_toast("Debug overlay on" if debug
                               else "Debug overlay off")
            elif key == ord("r"):
                if not cfg.ocr_enabled:
                    hud.show_toast(
                        "OCR disabled (set ocr.enabled: true, then restart)")
                elif ocr_text:
                    tts.speak(f"Text says, {ocr_text[:80]}")
                    hud.show_toast(f"Text: {ocr_text[:60]}")
                else:
                    hud.show_toast("No text detected yet")
            elif key == ord("s"):
                path = take_screenshot(display, "assets/screenshots/assist")
                hud.show_toast(f"Screenshot: {Path(path).name}")
            elif key == ord(" "):
                pipe.reset()
                queue.reset()
                hud.show_toast("Tracking + speech memory reset")
    finally:
        pipe.stop()
        queue.shutdown()
        tts.shutdown()
        cv2.destroyAllWindows()
        print("Assistive Vision app closed.")


if __name__ == "__main__":
    main()
