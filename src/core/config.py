"""Pipeline configuration shared by the core engine and its consumers.

Moved out of ``src/server/pipeline.py`` so the core engine is usable
without Flask.  ``src.server.pipeline.PipelineConfig`` re-exports this
type so existing imports keep working.
"""
from dataclasses import dataclass, field
from typing import Dict, Tuple

import yaml


@dataclass
class PipelineConfig:
    """All tunable pipeline settings."""

    camera_id: int = 0
    camera_resolution: Tuple[int, int] = (1280, 720)
    # Camera geometry, corrected exactly once at capture.  mirror=True
    # horizontally flips the RAW frame — leave OFF so OCR/YOLO see true
    # text orientation (front-camera preview mirroring is a *display*
    # concern: `preview_mirror`).  rotate fixes a physically sideways
    # sensor (0/90/180/270 degrees).
    camera_mirror: bool = False
    camera_rotate: int = 0
    # Display-only selfie-style mirroring of the video feed (the OCR
    # panel and HUD are drawn on top and never flipped).
    preview_mirror: bool = True
    model_path: str = "models/yolov8n.onnx"
    conf_threshold: float = 0.4
    nms_iou_threshold: float = 0.45
    conf_overrides: Dict[str, float] = field(default_factory=dict)
    filter_tall_laptops: bool = False
    reject_box_shape: Dict[str, Dict[str, float]] = field(default_factory=dict)
    detect_every: int = 2
    ocr_every: int = 10
    ocr_min_conf: float = 0.3
    ocr_preprocess: str = "none"
    ocr_max_boxes: int = 50
    # Object-aware OCR (inspect text-bearing object ROIs instead of
    # scanning the whole frame).  Only active when ocr_enabled is true.
    object_ocr_enabled: bool = True
    ocr_policy_path: str = "configs/ocr_policy.yaml"
    ocr_roi_padding: float = 0.1
    ocr_roi_min_w: int = 24
    ocr_roi_min_h: int = 12
    ocr_max_upscale: float = 3.0
    ocr_text_presence: bool = True
    ocr_variants: int = 3
    ocr_cooldown_s: float = 3.0
    ocr_stale_after_s: float = 5.0
    ocr_move_px: int = 40
    ocr_auto_read: bool = False
    ocr_history_max: int = 20
    ocr_min_chars: int = 2
    ocr_timeout_ms: int = 2000
    ocr_debug_records: int = 8
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
    encode_jpeg: bool = True
    vfov_deg: float = 55.0
    mode: str = "object"
    inference_timeout_ms: int = 2000
    metrics: bool = True
    depth_enabled: bool = False
    depth_backend: str = "synthetic"
    depth_model_path: str = ""
    planner_cooldown: float = 2.5
    planner_dedupe: bool = True
    ocr_enabled: bool = False
    navigation_enabled: bool = True
    tracking_smoothing: float = 0.4
    tracking_conf_smoothing: float = 0.5
    tracking_label_vote_window: int = 5
    tracking_class_consistent: bool = True

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
        plan = data.get("planner", {})

        cfg.camera_id = cam.get("id", cfg.camera_id)
        cfg.camera_resolution = tuple(cam.get("resolution",
                                              cfg.camera_resolution))
        cfg.camera_mirror = bool(cam.get("mirror", cfg.camera_mirror))
        cfg.camera_rotate = int(cam.get("rotate", cfg.camera_rotate)) % 360
        cfg.preview_mirror = bool(
            cam.get("preview_mirror", cfg.preview_mirror))
        cfg.model_path = det.get("model_path", cfg.model_path)
        cfg.conf_threshold = float(
            det.get("conf_threshold", cfg.conf_threshold))
        cfg.nms_iou_threshold = float(
            det.get("iou_threshold", cfg.nms_iou_threshold))
        cfg.conf_overrides = dict(det.get("conf_overrides", {}) or {})
        cfg.filter_tall_laptops = bool(
            det.get("filter_tall_laptops", cfg.filter_tall_laptops))
        cfg.reject_box_shape = dict(det.get("reject_box_shape", {}) or {})
        cfg.detect_every = max(1, int(det.get("every_n_frames", cfg.detect_every)))
        cfg.ocr_every = max(1, int(ocr.get("every_n_frames", cfg.ocr_every)))
        cfg.ocr_enabled = bool(ocr.get("enabled", cfg.ocr_enabled))
        cfg.ocr_min_conf = float(ocr.get("min_confidence", cfg.ocr_min_conf))
        cfg.ocr_preprocess = str(ocr.get("preprocess", cfg.ocr_preprocess))
        cfg.ocr_max_boxes = int(ocr.get("max_boxes", cfg.ocr_max_boxes))
        cfg.object_ocr_enabled = bool(
            ocr.get("object_ocr_enabled", cfg.object_ocr_enabled))
        cfg.ocr_policy_path = str(ocr.get("policy_path",
                                          cfg.ocr_policy_path))
        cfg.ocr_roi_padding = float(ocr.get("roi_padding",
                                            cfg.ocr_roi_padding))
        cfg.ocr_roi_min_w = int(ocr.get("roi_min_w", cfg.ocr_roi_min_w))
        cfg.ocr_roi_min_h = int(ocr.get("roi_min_h", cfg.ocr_roi_min_h))
        cfg.ocr_max_upscale = float(ocr.get("max_upscale",
                                            cfg.ocr_max_upscale))
        cfg.ocr_text_presence = bool(ocr.get("text_presence",
                                             cfg.ocr_text_presence))
        cfg.ocr_variants = max(1, int(ocr.get("variants", cfg.ocr_variants)))
        cfg.ocr_cooldown_s = float(ocr.get("cooldown_s", cfg.ocr_cooldown_s))
        cfg.ocr_stale_after_s = float(
            ocr.get("stale_after_s", cfg.ocr_stale_after_s))
        cfg.ocr_move_px = int(ocr.get("move_px", cfg.ocr_move_px))
        cfg.ocr_auto_read = bool(ocr.get("auto_read", cfg.ocr_auto_read))
        cfg.ocr_history_max = int(ocr.get("history_max", cfg.ocr_history_max))
        cfg.ocr_min_chars = int(ocr.get("min_chars", cfg.ocr_min_chars))
        cfg.ocr_timeout_ms = int(ocr.get("timeout_ms", cfg.ocr_timeout_ms))
        cfg.ocr_debug_records = max(
            0, int(ocr.get("debug_records", cfg.ocr_debug_records)))
        cfg.iou_threshold = float(trk.get("iou_threshold", cfg.iou_threshold))
        cfg.max_missed = int(trk.get("max_missed", cfg.max_missed))
        cfg.tracking_smoothing = float(
            trk.get("smoothing", cfg.tracking_smoothing))
        cfg.tracking_conf_smoothing = float(
            trk.get("conf_smoothing", cfg.tracking_conf_smoothing))
        cfg.tracking_label_vote_window = int(
            trk.get("label_vote_window", cfg.tracking_label_vote_window))
        cfg.tracking_class_consistent = bool(
            trk.get("class_consistent", cfg.tracking_class_consistent))
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
        cfg.mode = str(app.get("mode", cfg.mode))
        depth = data.get("depth", {})
        cfg.depth_enabled = bool(depth.get("enabled", cfg.depth_enabled))
        cfg.depth_backend = str(depth.get("backend", cfg.depth_backend))
        cfg.depth_model_path = str(depth.get("model_path",
                                             cfg.depth_model_path))
        cfg.planner_cooldown = float(
            plan.get("cooldown_seconds", cfg.planner_cooldown))
        cfg.planner_dedupe = bool(plan.get("dedupe", cfg.planner_dedupe))
        heights = dict(nav.get("reference_heights", {}) or {})
        if heights:
            from src.navigation import guidance as _g

            _g._REFERENCE_HEIGHTS.update({
                str(k): float(v) for k, v in heights.items()
            })
        return cfg

    def to_yaml(self, path: str) -> None:
        """Write the config back to a YAML file (for tooling)."""
        from dataclasses import asdict

        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(asdict(self), fh, sort_keys=False)