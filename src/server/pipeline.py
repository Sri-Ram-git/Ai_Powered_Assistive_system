"""Vision pipeline server — runs the AI pipeline in a background thread
and exposes it to the web dashboard.

The pipeline thread captures camera frames, runs (throttled) detection +
tracking + OCR, produces the annotated JPEG for the MJPEG stream, and
maintains a small JSON "state" (detections with distances, AI guidance
phrase, FPS, latency) that the dashboard polls via /api/state.

Architecture:
    Camera ─▶ detect ─▶ track ─▶ monitor/decision ─▶ guidance + speech
        │                                              │
        └── annotate ─▶ JPEG ─▶ /video_feed (MJPEG)     │
                             └▶ state dict ─▶ /api/state
"""
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import yaml

from src.camera.camera import Camera
from src.decision import DecisionEngine, FrameSummary
from src.detection import YoloDetector
from src.ocr import OcrEngine
from src.tracking import IoUTracker, TrackingMonitor
from src.utils.logger import setup_logger

_logger = setup_logger("PipelineServer")


@dataclass
class PipelineConfig:
    """All tunable pipeline settings."""

    camera_id: int = 0
    camera_resolution: tuple = (1280, 720)
    model_path: str = "models/yolov8n.onnx"
    detect_every: int = 2
    ocr_every: int = 10
    ocr_min_conf: float = 0.3
    iou_threshold: float = 0.3
    max_missed: int = 8
    distance_delta: float = 1.0
    min_announce: float = 3.0
    cooldown: float = 4.0
    min_priority: int = 5
    speak_ocr_text: bool = True
    max_ocr_chars: int = 80
    jpeg_quality: int = 70
    jpeg_width: int = 960
    vfov_deg: float = 55.0

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        cfg = cls()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except FileNotFoundError:
            return cfg

        det = data.get("detection", {})
        trk = data.get("tracking", {})
        ocr = data.get("ocr", {})
        dec = data.get("decision", {})
        cam = data.get("camera", {})
        app = data.get("app", {})
        nav = data.get("navigation", {})

        cfg.camera_id = cam.get("id", cfg.camera_id)
        cfg.camera_resolution = tuple(cam.get("resolution",
                                              cfg.camera_resolution))
        cfg.model_path = det.get("model_path", cfg.model_path)
        cfg.detect_every = max(1, int(det.get("every_n_frames", cfg.detect_every)))
        cfg.ocr_every = max(1, int(ocr.get("every_n_frames", cfg.ocr_every)))
        cfg.ocr_min_conf = float(ocr.get("min_confidence", cfg.ocr_min_conf))
        cfg.iou_threshold = float(trk.get("iou_threshold", cfg.iou_threshold))
        cfg.max_missed = int(trk.get("max_missed", cfg.max_missed))
        cfg.distance_delta = float(
            trk.get("distance_change_metres", cfg.distance_delta))
        cfg.min_announce = float(
            trk.get("min_announce_interval", cfg.min_announce))
        cfg.cooldown = float(dec.get("cooldown_seconds", cfg.cooldown))
        cfg.min_priority = int(dec.get("min_priority", cfg.min_priority))
        cfg.speak_ocr_text = bool(dec.get("speak_ocr_text",
                                          cfg.speak_ocr_text))
        cfg.max_ocr_chars = int(dec.get("max_ocr_chars", cfg.max_ocr_chars))
        cfg.jpeg_width = int(app.get("jpeg_width", cfg.jpeg_width))
        cfg.vfov_deg = float(nav.get("vertical_fov", cfg.vfov_deg))
        heights = dict(nav.get("reference_heights", {}) or {})
        if heights:
            from src.navigation import guidance as _g

            _g._REFERENCE_HEIGHTS.update({
                str(k): float(v) for k, v in heights.items()
            })
        return cfg


class PipelineServer:
    """Runs the vision pipeline and serves frames + state to the web UI."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self._cfg = config or PipelineConfig()
        self._latest_jpeg: Optional[bytes] = None
        self._latest_lock = threading.Lock()
        self._state: Dict = {"running": False}
        self._state_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._grab_thread: Optional[threading.Thread] = None
        self._infer_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._speech_callback = None
        self._engine = None
        self._camera = None

        # Shared between the grab thread (writes) and inference thread
        # (reads/writes the latest detection results).
        self._latest_frame = None
        self._latest_frame_lock = threading.Lock()
        self._latest_tracks: List = []
        self._latest_ocr: List = []
        self._results_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(
        self,
        speech_callback=None,
        timeout: Optional[float] = None,
    ) -> None:
        """Start the pipeline in a background thread.

        Args:
            speech_callback: Optional callable(text) invoked for each
                spoken phrase (e.g. wires up the TTS engine).
            timeout: Block up to this many seconds waiting for the
                pipeline to boot (used in tests).  None = no wait.
        """
        self._speech_callback = speech_callback
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="pipeline", daemon=True,
        )
        self._thread.start()
        if timeout:
            deadline = time.time() + timeout
            while time.time() < deadline:
                if self._state.get("running"):
                    return
                time.sleep(0.1)

    def stop(self) -> None:
        self._stop.set()
        if self._grab_thread is not None:
            self._grab_thread.join(timeout=3.0)
        if self._infer_thread is not None:
            self._infer_thread.join(timeout=3.0)
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._camera is not None:
            self._camera.stop()

    # ------------------------------------------------------------------
    # Accessors for the Flask layer
    # ------------------------------------------------------------------

    @property
    def latest_jpeg(self) -> Optional[bytes]:
        with self._latest_lock:
            return self._latest_jpeg

    def state_snapshot(self) -> Dict:
        with self._state_lock:
            state = dict(self._state)
        if "detections" in state:
            state["detections"] = [dict(d) for d in state["detections"]]
        if isinstance(state.get("ocr_text"), str):
            state["ocr_text"] = state["ocr_text"]
        return state

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Start the camera, then run grab + inference worker threads.

        The grab thread reads frames as fast as the camera provides them
        and encodes the annotated JPEG for the MJPEG stream — it never
        runs AI, so the feed stays live regardless of inference speed.
        The inference thread runs detection/tracking/OCR/decision on the
        latest frame and publishes state.
        """
        cfg = self._cfg

        camera = Camera(camera_id=cfg.camera_id,
                        resolution=cfg.camera_resolution)
        try:
            camera.start()
        except Exception as exc:
            self._set_state(running=False, error=str(exc))
            _logger.error("Camera failed: %s", exc)
            return
        self._camera = camera

        self._set_state(running=True, error=None,
                        resolution=list(camera.resolution))
        _logger.info("Pipeline running at %dx%d",
                     *camera.resolution)

        self._grab_thread = threading.Thread(
            target=self._grab_loop, name="pipeline-grab", daemon=True,
        )
        self._grab_thread.start()

        self._infer_thread = threading.Thread(
            target=self._infer_loop, name="pipeline-infer", daemon=True,
        )
        self._infer_thread.start()

        # Keep the coordinator thread alive until stop().
        while not self._stop.wait(0.25):
            pass

        self._set_state(running=False)
        camera.stop()
        _logger.info("Pipeline stopped")

    # ------------------------------------------------------------------
    # Worker: grab + encode (never runs AI)
    # ------------------------------------------------------------------

    def _grab_loop(self) -> None:
        cfg = self._cfg
        last_fps_time = time.time()
        while not self._stop.is_set():
            try:
                frame = self._camera.read()
            except Exception as exc:
                self._set_state(error=f"Frame grab failed: {exc}")
                self._stop.wait(0.5)
                continue

            # Share the raw frame for the inference thread.
            with self._latest_frame_lock:
                self._latest_frame = frame

            # Annotate using the *latest* AI results and encode — cheap
            # (draw + JPEG), so it never blocks the camera.
            with self._results_lock:
                tracks = list(self._latest_tracks)
                ocr_items = list(self._latest_ocr)
            annotated = self._annotate(frame, tracks, ocr_items)
            jpeg = self._to_jpeg(annotated, cfg.jpeg_width, cfg.jpeg_quality)
            with self._latest_lock:
                self._latest_jpeg = jpeg

            # Report the *feed* FPS (grab rate), not inference rate.
            now = time.time()
            if now - last_fps_time >= 1.0:
                last_fps_time = now
                with self._state_lock:
                    self._state["fps"] = self._camera.actual_fps

    # ------------------------------------------------------------------
    # Worker: inference (never touches the JPEG feed)
    # ------------------------------------------------------------------

    def _infer_loop(self) -> None:
        cfg = self._cfg

        model_path = cfg.model_path
        if not Path(model_path).is_absolute():
            model_path = str(Path(__file__).resolve().parents[2] / model_path)

        detector = YoloDetector(
            model_path,
            input_size=640,
            conf_threshold=0.35,
            iou_threshold=0.45,
        )
        ocr = OcrEngine(min_confidence=cfg.ocr_min_conf)
        tracker = IoUTracker(iou_threshold=cfg.iou_threshold,
                             max_missed=cfg.max_missed)
        monitor = TrackingMonitor(
            distance_change_metres=cfg.distance_delta,
            min_announce_interval=cfg.min_announce,
            vfov_deg=cfg.vfov_deg,
        )
        engine = DecisionEngine(
            cooldown_seconds=cfg.cooldown,
            min_priority=cfg.min_priority,
            read_ocr_text=cfg.speak_ocr_text,
            max_ocr_chars=cfg.max_ocr_chars,
        )

        frame_index = 0
        last_ocr: List = []
        tracks: List = []

        while not self._stop.is_set():
            with self._latest_frame_lock:
                frame = self._latest_frame
                if frame is None:
                    frame = None

            # No frame yet (camera still warming up) — wait and retry.
            if frame is None:
                self._stop.wait(0.05)
                continue

            frame_index += 1
            start = time.time()

            if frame_index % cfg.detect_every == 0:
                try:
                    detections = detector.detect(frame)
                    tracks = tracker.update(detections)
                except Exception as exc:
                    _logger.error("Detection failed: %s", exc)

            if frame_index % cfg.ocr_every == 0:
                try:
                    last_ocr = ocr.read_text(frame)
                except Exception as exc:
                    _logger.warning("OCR failed: %s", exc)
                    last_ocr = []

            # Publish the newest results for the grab thread to draw.
            with self._results_lock:
                self._latest_tracks = list(tracks)
                self._latest_ocr = list(last_ocr)

            phrases = monitor.events(tracks, frame.shape[1], frame.shape[0])
            summary = FrameSummary(
                detections=[t for t in tracks],
                ocr_items=last_ocr,
                frame_w=frame.shape[1],
                frame_h=frame.shape[0],
            )
            phrase = engine.decide(summary)
            if phrase:
                phrases.append(phrase)

            for p in phrases:
                _logger.info("Speak: %s", p)
                if self._speech_callback is not None:
                    try:
                        self._speech_callback(p)
                    except Exception:  # pragma: no cover
                        pass

            self._update_state(
                tracks=tracks,
                ocr_items=last_ocr,
                phrases=phrases,
                frame_w=frame.shape[1],
                frame_h=frame.shape[0],
                latency_ms=(time.time() - start) * 1000.0,
            )

            # Leave a little CPU for the camera/grab thread between
            # inference ticks (throttled by the every_n_frames knobs).
            self._stop.wait(0.02)

        _logger.info("Inference loop stopped")

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _set_state(self, **kwargs) -> None:
        with self._state_lock:
            self._state.update(kwargs)

    def _update_state(
        self,
        tracks,
        ocr_items,
        phrases: List[str],
        frame_w: int,
        frame_h: int,
        latency_ms: float,
    ) -> None:
        items = []
        for t in tracks:
            items.append({
                "track_id": t.track_id,
                "label": t.label,
                "confidence": round(t.confidence, 2),
                "distance": round(_track_distance_m(t, frame_h,
                                                    self._cfg.vfov_deg), 1),
                "direction": _track_direction(t, frame_w),
            })
        ocr_text = " ".join(r.text for r in ocr_items)
        with self._state_lock:
            self._state.update(
                detections=items,
                ocr_text=ocr_text,
                guidance=" ".join(phrases) or None,
                latency_ms=round(latency_ms, 1),
            )

    # ------------------------------------------------------------------
    # Drawing / encoding
    # ------------------------------------------------------------------

    def _annotate(self, frame, tracks, ocr_items):
        display = frame.copy()
        for t in tracks:
            x, y, w, h = t.box
            color = _track_color(t.track_id)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            dist = _track_distance_m(t, display.shape[0], self._cfg.vfov_deg)
            text = f"#{t.track_id} {t.label} {dist:.1f}m"
            label_y = y - 8 if y - 8 > 10 else y + h + 18
            cv2.putText(display, text, (x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        for r in ocr_items:
            bx, by, bw, bh = r.box
            cv2.rectangle(display, (bx, by), (bx + bw, by + bh),
                          (0, 165, 255), 2)
            cv2.putText(display, r.text, (bx, by - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        return display

    def _to_jpeg(self, frame, width: int, quality: int) -> bytes:
        if frame.shape[1] > width:
            scale = width / frame.shape[1]
            new_w = int(frame.shape[1] * scale)
            new_h = int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (new_w, new_h),
                               interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [
            cv2.IMWRITE_JPEG_QUALITY, int(quality),
        ])
        if not ok:
            return b""
        return buf.tobytes()


def _track_distance_m(track, frame_h: int, vfov_deg: float = 55.0) -> float:
    from src.navigation.guidance import distance_estimate, reference_height

    return distance_estimate(track.box, frame_h,
                             reference_height(track.label),
                             vfov_deg=vfov_deg)


def _track_direction(track, frame_w: int) -> str:
    from src.navigation.guidance import direction_of

    return direction_of(track.box, frame_w)
