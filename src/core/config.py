"""Pipeline configuration shared by the core engine and its consumers.

Moved out of ``src/server/pipeline.py`` so the core engine is usable
without Flask.  ``src.server.pipeline.PipelineConfig`` re-exports this
type so existing imports keep working.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import yaml


@dataclass
class PipelineConfig:
    """All tunable pipeline settings."""

    camera_id: int = 0
    camera_resolution: Tuple[int, int] = (1280, 720)
    model_path: str = "models/yolov8n.onnx"
    conf_threshold: float = 0.4
    nms_iou_threshold: float = 0.45
    conf_overrides: Dict[str, float] = field(default_factory=dict)
    filter_tall_laptops: bool = False
    detect_every: int = 2
    ocr_every: int = 10
    ocr_min_conf: float = 0.3
    ocr_preprocess: str = "none"
    ocr_max_boxes: int = 50
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
        cfg.model_path = det.get("model_path", cfg.model_path)
        cfg.conf_threshold = float(
            det.get("conf_threshold", cfg.conf_threshold))
        cfg.nms_iou_threshold = float(
            det.get("iou_threshold", cfg.nms_iou_threshold))
        cfg.conf_overrides = dict(det.get("conf_overrides", {}) or {})
        cfg.filter_tall_laptops = bool(
            det.get("filter_tall_laptops", cfg.filter_tall_laptops))
        cfg.detect_every = max(1, int(det.get("every_n_frames", cfg.detect_every)))
        cfg.ocr_every = max(1, int(ocr.get("every_n_frames", cfg.ocr_every)))
        cfg.ocr_enabled = bool(ocr.get("enabled", cfg.ocr_enabled))
        cfg.ocr_min_conf = float(ocr.get("min_confidence", cfg.ocr_min_conf))
        cfg.ocr_preprocess = str(ocr.get("preprocess", cfg.ocr_preprocess))
        cfg.ocr_max_boxes = int(ocr.get("max_boxes", cfg.ocr_max_boxes))
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