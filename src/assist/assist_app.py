"""Assistive Vision App — real-time desktop interface (Phase 6+).

Runs the async engine (``src.core.AsyncVisionPipeline``) so no slow
stage can freeze the view:

    grab thread    camera → latest-frame store (never blocks)
    detect thread  YOLO → hardened IoU tracker (detect every N frames,
                   track at its own rate, latest results published)
    speech         engine phrases → SpeechQueue → non-blocking TTS

This app is a thin display layer: it shows the *latest* camera frame
with the *latest* tracking overlay at camera FPS, and never blocks on
detection, tracking, OCR, or speech.  OCR runs on its own worker thread
(non-blocking); it is enabled by default so text is read aloud.

Speech uses the object vocabulary (1551 words) to pick an announcement
tier and to vary repeated phrasing so the same object is not announced
with one fixed sentence.

Usage:
    python src/assist/assist_app.py [--camera 0] [--config configs/assist_config.yaml]

Keys:
    m          mute / unmute speech
    d          toggle debug overlay (fps, latency, raw vs smoothed boxes)
    r          read the most recently recognised text aloud (READ)
    c          copy the latest recognised text to the clipboard
    n          request OCR now on the best text-bearing object
    x          clear the recognised-text history
    s          save annotated screenshot
    space      reset tracking + decision + speech memory
    q          quit

Mouse:
    click the TEXT panel buttons on the right (READ / NOW / COPY / CLEAR)
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio import SpeechOutput  # noqa: E402
from src.audio.speech_queue import SpeechQueue, SpeechTier  # noqa: E402
from src.audio.variety import SpeechVariety  # noqa: E402
from src.vocabulary import ObjectVocabulary  # noqa: E402
from src.camera import (  # noqa: E402
    HUD,
    get_screen_size,
    open_fullscreen_window,
    scale_to_fit,
    take_screenshot,
)
from src.assist.text_panel import TextPanel  # noqa: E402
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
                debug: bool = False, mirror_x: bool = False):
    """Draw tracked boxes with stable IDs + distance (smooth == solid).

    ``mirror_x`` mirrors the box coordinates so overlays align with a
    selfie-style (flipped) preview of the *same* raw vision frame — the
    tracking itself runs on the unmirrored frame, so OCR text stays in
    true reading orientation.

    In debug mode the raw (un-smoothed) box from the last detection is
    drawn in cyan so box smoothing is visible.
    """
    display = frame.copy()
    w = display.shape[1]
    for track in tracks:
        x, y, bw, bh = track.box
        if mirror_x:
            x = w - (x + bw)
        color = _track_color(track.track_id)
        cv2.rectangle(display, (x, y), (x + bw, y + bh), color, 2)

        if debug:
            rx, ry, rw, rh = track.raw_box
            if mirror_x:
                rx = w - (rx + rw)
            cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh),
                          (255, 255, 0), 1)

        dist = track_distance(track, frame_h, vfov_deg)
        conf = f"{track.confidence:.2f}" if debug else ""
        text = f"#{track.track_id} {track.label} {dist}{conf}"
        label_y = y - 8 if y - 8 > 10 else y + bh + 18
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


def _debug_overlay(display, state, results, extra: str = "",
                   ocr=None, geom=None):
    """Draw a small diagnostics panel (Phase 23 + OCR accuracy work).

    ``ocr`` is the pipeline's ``latest_track_ocr()`` dict (or None);
    ``geom`` is ``pipe.camera_geometry()`` (mirror / rotation).
    """
    lines = [
        f"cam {state.get('fps', 0.0):.1f} fps",
        f"yolo {results['latencies'].get('yolo_ms', 0.0):.1f} ms",
        f"tracks {len(results['tracks'])} | pending speech {extra}",
    ]
    if geom is not None:
        lines.append(
            f"mirror {geom.get('mirror', False)} "
            f"| rotate {geom.get('rotate', 0)}deg")
    if ocr:
        lines.append(
            f"ocr {ocr.get('variant', '')} "
            f"{ocr.get('latency_ms', 0.0):.0f}ms "
            f"conf {ocr.get('confidence', 0.0) * 100:.0f}%")
        lines.append(
            f"stable {ocr.get('stable', False)} track#{ocr.get('track_id')}")
    y = 8
    for line in lines:
        cv2.putText(display, line, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        y += 22
    return display


def _speech_tier_for(text: str, vocab: Optional[ObjectVocabulary] = None) -> SpeechTier:
    """Map engine output to a speech tier.

    Vocabulary tier wins (a detected "car"/"person" is critical), then
    safety keyword cues, then normal.
    """
    lowered = (text or "").lower()
    if vocab is not None:
        for word in lowered.split():
            tier = vocab.tier_for(word.strip(".,;:!?"))
            if tier == "critical":
                return SpeechTier.CRITICAL
            if tier == "high":
                return SpeechTier.HIGH
            if tier == "low":
                return SpeechTier.LOW
    if "stop" in lowered or "emergency" in lowered:
        return SpeechTier.CRITICAL
    if "turn" in lowered or "now" in lowered:
        return SpeechTier.HIGH
    return SpeechTier.NORMAL


def _copy_to_clipboard(text: str) -> bool:
    """Copy text to the Windows clipboard (via the `clip` helper)."""
    if not text:
        return False
    try:
        import subprocess

        subprocess.run(["clip"], input=text, text=True, check=True)
        return True
    except Exception:
        _logger.warning("Clipboard copy failed", exc_info=True)
        return False


def _current_text(pipe) -> str:
    latest = pipe.latest_track_ocr()
    return (latest or {}).get("text", "")


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
    variety = SpeechVariety()
    try:
        vocab = ObjectVocabulary.load()
    except Exception:
        vocab = None

    def speak(text: str) -> None:
        if not mute:
            varied = variety.render(text)
            queue.enqueue(varied, tier=_speech_tier_for(varied, vocab))

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
    panel = TextPanel()
    hud.show_toast(
        "m mute | d debug | r read | c copy | n read now | "
        "x clear text | s save | space reset | q quit")

    debug = False
    cam_info = _CameraInfo(cfg.camera_id, cfg.camera_resolution)

    def _dispatch(action: str) -> None:
        if action == "read":
            if not cfg.ocr_enabled:
                hud.show_toast("OCR disabled (set ocr.enabled: true)")
            elif pipe.read_latest_text():
                hud.show_toast("Reading text...")
            else:
                hud.show_toast("No text detected yet")
        elif action == "now":
            if cfg.ocr_enabled and pipe.request_manual_ocr():
                hud.show_toast("Reading now...")
            else:
                hud.show_toast("OCR disabled or no frame yet")
        elif action == "copy":
            text = _current_text(pipe)
            if _copy_to_clipboard(text):
                hud.show_toast(f"Copied: {text[:40]}")
            else:
                hud.show_toast("Nothing to copy")
        elif action == "clear":
            pipe.clear_track_ocr()
            hud.show_toast("Text history cleared")

    def _on_mouse(event, x, y, flags, param) -> None:
        if event == cv2.EVENT_MOUSEMOVE:
            panel.on_motion(x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            action = panel.hit_test(x, y)
            if action:
                _dispatch(action)

    cv2.setMouseCallback(window, _on_mouse)

    try:
        while True:
            frame = pipe.latest_frame
            if frame is None:
                cv2.waitKey(20)
                continue

            results = pipe.latest_results.snapshot()
            state = pipe.state_snapshot()

            # Front-camera selfie preview may be mirrored for the USER,
            # but OCR/YOLO/tracking always run on the pipeline's raw,
            # geometrically-correct vision frame.  Only the display copy
            # is flipped here, and track boxes are mirrored to match.
            preview = cv2.flip(frame, 1) if cfg.preview_mirror else frame
            display = draw_tracks(preview, results["tracks"],
                                  frame.shape[0], cfg.vfov_deg,
                                  debug=debug, mirror_x=cfg.preview_mirror)
            display = scale_to_fit(display, screen_w, screen_h)
            display = panel.render(
                display,
                latest=pipe.latest_track_ocr(),
                history=pipe.track_ocr_history(),
                stats=pipe.ocr_stats(),
                busy=pipe.ocr_busy,
                status=pipe.ocr_status(),
                debug=debug,
            )
            if debug:
                display = _debug_overlay(
                    display, state, results,
                    ocr=pipe.latest_track_ocr(),
                    geom=pipe.camera_geometry(),
                )

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
                _dispatch("read")
            elif key == ord("c"):
                _dispatch("copy")
            elif key == ord("n"):
                _dispatch("now")
            elif key == ord("x"):
                _dispatch("clear")
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
