"""Asynchronous vision pipeline — the core engine.

Decouples camera capture from AI inference so no slow stage (OCR, YOLO)
can freeze the camera feed, the UI, or speech.

Threads:
    grab    — reads camera frames, publishes to the FrameManager, and
              annotates/encodes the latest JPEG for the UI (never AI).
    detect  — reads the newest frame, runs YOLO + IoU tracking +
              decision/monitor guidance, publishes latest results.
    ocr     — OcrWorker: runs OCR on its own thread; consumers read the
              latest completed OCR result (never wait synchronously).

The core engine has no Flask dependency: the web server and the desktop
app are thin consumers of ``AsyncVisionPipeline``.
"""
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from src.camera.camera import Camera
from src.core.config import PipelineConfig
from src.core.frame_manager import FrameManager
from src.core.results import LatestResults
from src.decision import DecisionEngine, FrameSummary
from src.detection import YoloDetector
from src.metrics import MetricsRegistry
from src.ocr import OcrEngine
from src.ocr.worker import OcrWorker
from src.tracking import IoUTracker, TrackingMonitor
from src.utils.logger import setup_logger

_logger = setup_logger("AsyncPipeline")


class AsyncVisionPipeline:
    """Grab + detect + OCR engine with latest-result semantics."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        camera: Optional[Camera] = None,
        speech_callback: Optional[Callable[[str], None]] = None,
        camera_factory: Optional[Callable[..., Camera]] = None,
    ) -> None:
        """Configure the pipeline.

        Args:
            config: Pipeline settings; defaults to PipelineConfig().
            camera: Pre-built camera (e.g. a stub in tests).  If None a
                camera is opened from ``camera_factory`` (defaults to
                src.camera.Camera).
            speech_callback: Optional callable(text) for each phrase.
            camera_factory: Callable(camera_id=..., resolution=...) used
                to build the camera lazily.  Overridable so tests can
                inject a stub camera.
        """
        self._cfg = config or PipelineConfig()
        self._camera = camera
        self._camera_factory = camera_factory or Camera
        self._speech_callback = speech_callback

        self._frames = FrameManager()
        self._results = LatestResults()

        self._latest_jpeg: Optional[bytes] = None
        self._jpeg_lock = threading.Lock()
        self._state: Dict = {
            "running": False,
            "detections": [],
            "ocr_text": "",
            "guidance": None,
            "fps": 0.0,
        }
        self._state_lock = threading.Lock()

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._grab_thread: Optional[threading.Thread] = None
        self._detect_thread: Optional[threading.Thread] = None
        self._ocr_worker: Optional[OcrWorker] = None

        # Created lazily so the constructor never touches hardware/models.
        self._detector = None
        self._engine = None
        self._tracker = None
        self._monitor = None
        self._decision = None
        self._model_path = None
        self._depth = None
        self._latest_scene = None
        self._safety = None
        self._latest_risk = None
        self._planner = None

        # Observability (Prometheus-style metrics).
        self._metrics = MetricsRegistry() if self._cfg.metrics else None

        self._mode_manager = None
        self._decision_engine = None
        self._tracker = None
        self._monitor = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(
        self,
        timeout: Optional[float] = None,
        speech_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Start the pipeline; optionally block until ready."""
        self._stop.clear()
        if speech_callback is not None:
            self._speech_callback = speech_callback

        if self._camera is None:
            self._camera = self._camera_factory(
                camera_id=self._cfg.camera_id,
                resolution=self._cfg.camera_resolution,
            )
            try:
                self._camera.start()
            except Exception as exc:
                self._set_state(running=False, error=str(exc))
                _logger.error("Camera failed: %s", exc)
                return
        self._set_state(running=True, error=None,
                        resolution=list(self._camera.resolution))
        _logger.info("Pipeline running at %dx%d",
                     *self._camera.resolution)

        self._thread = threading.Thread(
            target=self._coordinator, name="pipeline", daemon=True,
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
        if self._ocr_worker is not None:
            self._ocr_worker.stop()
        for t in (self._grab_thread, self._detect_thread, self._thread):
            if t is not None and t.is_alive():
                t.join(timeout=3.0)
        if self._camera is not None:
            self._camera.stop()
        self._set_state(running=False)
        _logger.info("Pipeline stopped")

    # ------------------------------------------------------------------
    # Consumers (UI/API)
    # ------------------------------------------------------------------

    @property
    def latest_jpeg(self) -> Optional[bytes]:
        with self._jpeg_lock:
            return self._latest_jpeg

    @property
    def latest_frame(self) -> Optional[np.ndarray]:
        """The most recent raw camera frame (no annotation, no JPEG).

        Consumers (e.g. the desktop display loop) read the newest frame
        and overlay the latest ``latest_results`` themselves, so the
        display never blocks on detection or encoding.  Do not mutate
        the returned array.
        """
        _, frame = self._frames.latest()
        return frame

    @property
    def latest_results(self) -> LatestResults:
        return self._results

    @property
    def config(self) -> PipelineConfig:
        """Effective pipeline configuration (read-only consumer view)."""
        return self._cfg

    @property
    def metrics(self):
        """Metrics registry (None when metrics disabled)."""
        return self._metrics

    @property
    def latest_scene(self):
        """The most recently built SceneContext (or None)."""
        return self._latest_scene

    @property
    def latest_risk(self):
        """The most recent SafetyEngine RiskAssessment (or None)."""
        return self._latest_risk

    def state_snapshot(self) -> Dict:
        with self._state_lock:
            state = dict(self._state)
        if "detections" in state:
            state["detections"] = [dict(d) for d in state["detections"]]
        return state

    # ------------------------------------------------------------------
    # API-facing controls
    # ------------------------------------------------------------------

    VALID_MODES = ("object", "reading", "navigation", "scene", "voice")

    @property
    def mode_behavior(self):
        """The active ModeBehavior (or None before first set_mode)."""
        from src.modes import ModeManager

        if self._mode_manager is None:
            self._mode_manager = ModeManager(self._cfg)
        return self._mode_manager.get(self._cfg.mode)

    def set_mode(self, mode: str) -> None:
        """Switch the product mode (object|reading|navigation|scene|voice).

        Applies the mode's behaviour knobs to the live config (OCR,
        navigation, announce toggles).  The safety engine is never
        disabled by a mode.
        """
        from src.modes import ModeManager

        if self._mode_manager is None:
            self._mode_manager = ModeManager(self._cfg)
        behavior = self._mode_manager.apply(mode)
        # OCR-text narration follows the mode's announce_text toggle.
        if self._decision_engine is not None:
            self._decision_engine.set_read_ocr_text(behavior.announce_text)
        self._set_state(mode=behavior.name)
        _logger.info("Mode -> %s (%s)", behavior.name, behavior.label)

    def reset(self) -> None:
        """Forget all temporal state (tracks, cooldowns, speech memory).

        Used when the user signals a scene change (space bar in the
        desktop app).  The detector/model itself is untouched.
        """
        if self._tracker is not None:
            self._tracker.reset()
        if self._monitor is not None:
            self._monitor.reset()
        if self._decision_engine is not None:
            self._decision_engine.reset()
        if self._planner is not None:
            self._planner.reset()
        _logger.info("Pipeline temporal state reset")

    def handle_command(self, parsed) -> bool:
        """Execute a parsed voice command (or the raw text).

        Returns True when the command was recognised and dispatched.
        """
        if parsed is None:
            return False
        if isinstance(parsed, str):  # raw utterance: parse first
            from src.speech.command_parser import parse_command

            parsed = parse_command(parsed)

        from src.speech.commands import Command

        command = getattr(parsed, "command", None)
        if command is None:
            return False
        name = command.value if isinstance(command, Command) else command
        if name is None:
            return False

        # Map recognized commands onto pipeline behaviour.
        if name == Command.READ_TEXT.value:
            text = self._state.get("ocr_text", "")
            if text and self._speech_callback:
                self._speech_callback(f"Text reads: {text}")
            return True
        if name == Command.WHAT_DO_YOU_SEE.value:
            dets = self._state.get("detections", [])
            if dets:
                labels = ", ".join(f"{d['label']} {d['direction']}"
                                   for d in dets[:5])
                phrase = f"You are near {labels}"
            else:
                phrase = "I do not see any objects"
            if self._speech_callback:
                self._speech_callback(phrase)
            return True
        if name == Command.DESCRIBE_SCENE.value:
            scene = self._latest_scene
            if scene is not None:
                from src.vision.vlm import DeterministicVLM

                text = DeterministicVLM().describe(scene)
            else:
                text = "I do not see any objects yet"
            if self._speech_callback:
                self._speech_callback(text)
            return True
        if name == Command.REPEAT.value:
            if self._planner is not None and self._planner.history:
                last = self._planner.history[-1]
                if self._speech_callback:
                    self._speech_callback(last)
            return True
        if name == Command.HELP.value:
            from src.speech.commands import CommandRegistry

            if self._speech_callback:
                self._speech_callback(CommandRegistry().help_text())
            return True
        if name in (Command.START_OCR.value, Command.STOP_OCR.value):
            self._cfg.ocr_enabled = (name == Command.START_OCR.value)
            return True
        if name in (Command.ENABLE_NAVIGATION.value,
                    Command.DISABLE_NAVIGATION.value):
            self._cfg.navigation_enabled = (
                name == Command.ENABLE_NAVIGATION.value)
            return True
        if name == Command.STOP_SPEAKING.value:
            self._speech_callback = None
            return True
        return False

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------

    def _coordinator(self) -> None:
        """Start the worker threads and keep the coordinator alive."""
        self._grab_thread = threading.Thread(
            target=self._grab_loop, name="pipeline-grab", daemon=True,
        )
        self._grab_thread.start()

        self._detect_thread = threading.Thread(
            target=self._detect_loop, name="pipeline-detect", daemon=True,
        )
        self._detect_thread.start()

        # Phase 21: OCR is a slow stage; when disabled at startup it is
        # never loaded (no RapidOCR model load, no worker thread).
        if self._cfg.ocr_enabled:
            self._ocr_worker = self._build_ocr_worker()
            self._ocr_worker.start()
        else:
            self._ocr_worker = None
            _logger.info("OCR disabled — worker not started")

        if self._cfg.depth_enabled:
            try:
                from src.depth import create_depth_estimator

                self._depth = create_depth_estimator(
                    backend=self._cfg.depth_backend,
                    model_path=self._cfg.depth_model_path or None,
                )
                _logger.info("Depth stage enabled (%s)",
                             self._cfg.depth_backend)
            except Exception as exc:
                _logger.warning(
                    "Depth stage unavailable (continuing without): %s", exc)
                self._depth = None

        while not self._stop.wait(0.25):
            pass

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

            self._frames.publish(frame)

            if cfg.encode_jpeg:
                results = self._results.snapshot()
                annotated = self._annotate(frame, results)
                jpeg = self._to_jpeg(annotated, cfg.jpeg_width,
                                     cfg.jpeg_quality)
                with self._jpeg_lock:
                    self._latest_jpeg = jpeg

            now = time.time()
            if now - last_fps_time >= 1.0:
                last_fps_time = now
                fps = self._frames.fps()
                with self._state_lock:
                    self._state["fps"] = fps
                if self._metrics is not None:
                    self._metrics.set("camera_fps", fps)

    def _detect_loop(self) -> None:
        cfg = self._cfg
        # A missing/unloadable model must not kill the pipeline: the grab,
        # depth and OCR stages stay alive and the loop keeps publishing
        # state (with empty detections) so the app reports the fault via
        # state rather than silently stopping detection.
        try:
            detector = self._load_detector(cfg)
        except Exception as exc:
            _logger.error(
                "Detection unavailable (no model?) — running without "
                "detections: %s", exc)
            self._set_state(detection_error=str(exc))
            detector = None
        tracker = IoUTracker(
            iou_threshold=cfg.iou_threshold,
            max_missed=cfg.max_missed,
            smoothing=cfg.tracking_smoothing,
            conf_smoothing=cfg.tracking_conf_smoothing,
            class_consistent=cfg.tracking_class_consistent,
            label_vote_window=cfg.tracking_label_vote_window,
        )
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
        self._decision_engine = engine
        self._tracker = tracker
        self._monitor = monitor

        frame_index = 0
        while not self._stop.is_set():
            _, frame = self._frames.latest()
            if frame is None:
                self._stop.wait(0.05)
                continue

            frame_index += 1
            started = time.time()
            detections: List = []
            if detector is not None and frame_index % cfg.detect_every == 0:
                try:
                    detections = detector.detect(frame)
                    tracks = tracker.update(detections)
                except Exception as exc:
                    _logger.error("Detection failed: %s", exc)
                    tracks = tracker.active_tracks
            else:
                tracks = tracker.active_tracks

            # Optional depth stage (independent, non-blocking on detect).
            depth_result = None
            if self._depth is not None and frame_index % 5 == 0:
                try:
                    depth_result = self._depth.estimate(
                        frame, boxes=[t.box for t in tracks])
                except Exception as exc:
                    _logger.warning("Depth failed: %s", exc)

            ocr_items = (
                self._ocr_worker.latest_result()
                if self._ocr_worker is not None and cfg.ocr_enabled else []
            )

            # Deterministic scene context (world model for downstream).
            from src.vision.scene_context import build_scene_context

            scene = build_scene_context(
                tracks=list(tracks),
                ocr_text=[r.text for r in ocr_items],
                frame_w=frame.shape[1],
                frame_h=frame.shape[0],
                distance_of=lambda t, fh: _track_distance_m(
                    t, fh, self._cfg.vfov_deg),
                depth_available=(depth_result is not None),
            )
            self._latest_scene = scene

            # Deterministic safety assessment (never an LLM).
            from src.safety import SafetyEngine

            if self._safety is None:
                self._safety = SafetyEngine()
            risk = self._safety.assess(scene)
            self._latest_risk = risk

            phrases: List[str] = []
            behavior = self.mode_behavior
            if cfg.navigation_enabled and behavior.announce_objects:
                phrases = monitor.events(
                    tracks, frame.shape[1], frame.shape[0])
                summary = FrameSummary(
                    detections=list(tracks),
                    ocr_items=ocr_items,
                    frame_w=frame.shape[1],
                    frame_h=frame.shape[0],
                )
                phrase = engine.decide(summary, already_spoken=phrases)
                if phrase:
                    phrases.append(phrase)
            else:
                # Navigation off (or quiet mode): no guidance, but the
                # SafetyEngine is still assessed (never mode-dependent).
                summary = FrameSummary(
                    detections=list(tracks),
                    ocr_items=ocr_items,
                    frame_w=frame.shape[1],
                    frame_h=frame.shape[0],
                )

            # Scene mode: add a higher-level description on demand
            # (deterministic; never blocks the loop).
            if behavior.scene_describe and scene.objects:
                from src.vision.vlm import DeterministicVLM

                phrases.append(DeterministicVLM().describe(scene))

            # Response planner arbitrates everything (priority/dedup/
            # cooldown); urgent safety bypasses the cooldown.
            from src.response import (
                Response,
                ResponsePlanner,
                ResponsePriority,
            )

            if self._planner is None:
                from src.response import PlannerConfig

                self._planner = ResponsePlanner(PlannerConfig(
                    cooldown_seconds=self._cfg.planner_cooldown,
                    dedupe=self._cfg.planner_dedupe,
                ))
            proposals = [
                Response(p, ResponsePriority.NAVIGATION, source="detect")
                for p in phrases
            ]
            if risk and risk.urgent:
                top = risk.hazards[0]
                proposals.insert(
                    0,
                    Response(
                        f"{top.label} {top.direction} — stop",
                        ResponsePriority.URGENT_SAFETY,
                        source="safety", urgent=True,
                    ),
                )
            chosen = self._planner.plan(proposals, risk=risk)
            if chosen is not None:
                _logger.info("Speak: %s", chosen.text)
                if self._speech_callback is not None:
                    try:
                        self._speech_callback(chosen.text)
                    except Exception:  # pragma: no cover
                        pass

            latencies = {
                "yolo_ms": round(
                    (time.time() - started) * 1000.0, 1),
                "ocr_ms": round(
                    self._ocr_worker.last_latency_ms, 1)
                if self._ocr_worker is not None else 0.0,
                "depth_ms": round(depth_result.latency_ms, 1)
                if depth_result else 0.0,
            }
            if self._metrics is not None:
                self._metrics.observe("yolo_latency_ms", latencies["yolo_ms"])
                self._metrics.observe("ocr_latency_ms", latencies["ocr_ms"])
                self._metrics.observe("depth_latency_ms", latencies["depth_ms"])
                self._metrics.inc("frames_processed")
                if detections:
                    self._metrics.inc("detections_found", amount=len(detections))

            self._results.update(
                detections=detections,
                tracks=tracks,
                ocr_items=ocr_items,
                depth_map=depth_result.map if depth_result else None,
                guidance=phrases,
                latencies=latencies,
            )
            self._update_state(tracks, ocr_items, phrases, frame)

            # Publish a fresh frame to the OCR worker every N frames.
            if (cfg.ocr_enabled and
                    frame_index % cfg.ocr_every == 0):
                self._ocr_worker.submit(frame)

            self._stop.wait(0.02)

        _logger.info("Detection loop stopped")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_ocr_worker(self) -> OcrWorker:
        ocr = OcrEngine(min_confidence=self._cfg.ocr_min_conf,
                        max_boxes=self._cfg.ocr_max_boxes)
        return OcrWorker(
            ocr,
            preprocess_strategy=self._cfg.ocr_preprocess,
            timeout_ms=self._cfg.inference_timeout_ms,
        )

    def _load_detector(self, cfg: PipelineConfig) -> YoloDetector:
        model_path = cfg.model_path
        if not Path(model_path).is_absolute():
            model_path = str(Path(__file__).resolve().parents[2] / model_path)
        return YoloDetector(
            model_path,
            input_size=640,
            conf_threshold=cfg.conf_threshold,
            iou_threshold=cfg.nms_iou_threshold,
            conf_overrides=cfg.conf_overrides,
            filter_tall_laptops=cfg.filter_tall_laptops,
        )

    def _update_state(self, tracks, ocr_items, phrases, frame) -> None:
        items = []
        for t in tracks:
            items.append({
                "track_id": t.track_id,
                "label": t.label,
                "confidence": round(t.confidence, 2),
                "distance": round(_track_distance_m(
                    t, frame.shape[0], self._cfg.vfov_deg), 1),
                "direction": _track_direction(t, frame.shape[1]),
            })
        ocr_text = " ".join(r.text for r in ocr_items)
        with self._state_lock:
            self._state.update(
                detections=items,
                ocr_text=ocr_text,
                guidance=" ".join(phrases) or None,
            )

    def _set_state(self, **kwargs) -> None:
        with self._state_lock:
            self._state.update(kwargs)

    def _annotate(self, frame, results) -> np.ndarray:
        display = frame.copy()
        for t in results["tracks"]:
            x, y, w, h = t.box
            color = _track_color(t.track_id)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            dist = _track_distance_m(t, display.shape[0], self._cfg.vfov_deg)
            text = f"#{t.track_id} {t.label} {dist:.1f}m"
            label_y = y - 8 if y - 8 > 10 else y + h + 18
            cv2.putText(display, text, (x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        for r in results["ocr_items"]:
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


def _track_color(track_id: int):
    palette = [
        (0, 255, 0), (0, 165, 255), (255, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 0),
    ]
    return palette[track_id % len(palette)]