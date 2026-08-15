"""YOLOv8 object detection via OpenCV DNN and an ONNX export.

Loads a YOLOv8 detection model (.onnx) with OpenCV's DNN module — no
PyTorch or GPU required.  Handles letterboxing, forward pass,
confidence filtering, non-maximum suppression, and scale-back to the
original frame size.

Usage:
    detector = YoloDetector("models/yolov8n.onnx")
    results = detector.detect(frame)
    for r in results:
        print(r.label, r.confidence, r.box)
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.exceptions import DetectionError
from src.utils.logger import setup_logger

_logger = setup_logger("YoloDetector")

COCO_NAMES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Classes relevant to navigation guidance for a pedestrian assistant.
NAVIGATION_CLASSES: Dict[str, str] = {
    "person": "person",
    "bicycle": "bicycle",
    "car": "vehicle",
    "motorcycle": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "traffic light": "traffic signal",
    "stop sign": "stop sign",
    "chair": "obstacle",
    "couch": "obstacle",
    "potted plant": "obstacle",
    "bench": "obstacle",
    "fire hydrant": "obstacle",
}

# Classes that need a higher confidence bar before they're believed.
# YOLO is prone to labeling plain rectangles (doors, walls, screens)
# as "laptop", "tv", "book", etc. at low confidence.
_HIGH_CONF_CLASSES = {
    "laptop": 0.55,
    "tv": 0.5,
    "book": 0.55,
    "remote": 0.55,
    "mouse": 0.5,
    "cell phone": 0.5,
    "toilet": 0.5,
}

_INPUT_SIZE = 640


@dataclass
class DetectionResult:
    """One detected object."""

    label: str
    confidence: float
    box: Tuple[int, int, int, int]  # (x, y, w, h) in original frame coords

    @property
    def category(self) -> str:
        """Coarse group used by the decision engine (vehicle/person/...)."""
        return NAVIGATION_CLASSES.get(self.label, "object")

    @property
    def center(self) -> Tuple[float, float]:
        """Bounding-box centre (cx, cy)."""
        x, y, w, h = self.box
        return (x + w / 2.0, y + h / 2.0)

    @property
    def area(self) -> float:
        """Bounding-box area (w * h)."""
        _, _, w, h = self.box
        return float(w * h)


class YoloDetector:
    """OpenCV-DNN based YOLOv8 detector."""

    def __init__(
        self,
        model_path: str,
        input_size: int = _INPUT_SIZE,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
    ) -> None:
        """Configure the detector.

        Args:
            model_path: Path to the YOLOv8 .onnx export.
            input_size: Square size the model expects (default 640).
            conf_threshold: Minimum class confidence to keep a box.
            iou_threshold: NMS intersection-over-union cutoff.

        Raises:
            DetectionError: If the model cannot be loaded.
        """
        path = Path(model_path)
        if not path.exists():
            raise DetectionError(f"Model not found: {model_path}")
        try:
            self._net = cv2.dnn.readNetFromONNX(str(path))
        except cv2.error as exc:
            raise DetectionError(f"Failed to load ONNX model: {exc}") from exc

        self._input_size = int(input_size)
        self._conf = float(conf_threshold)
        self._iou = float(iou_threshold)
        self._names = list(COCO_NAMES)
        self._loaded = True
        _logger.info(
            "Loaded YOLO model %s (input=%d, conf=%.2f, iou=%.2f)",
            path.name, self._input_size, self._conf, self._iou,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def class_names(self) -> List[str]:
        """The class labels understood by the model."""
        return list(self._names)

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Detect objects in a BGR frame.

        Args:
            frame: BGR image (any size; letterboxed internally).

        Returns:
            List of DetectionResult sorted by confidence (highest first).
            Empty list when nothing is detected.
        """
        if frame is None or frame.size == 0:
            return []

        blob, ratio, pad_x, pad_y = self._letterbox(frame)
        self._net.setInput(blob)
        outputs = self._net.forward()

        detections = self._parse_outputs(
            outputs, ratio, pad_x, pad_y, frame.shape[1], frame.shape[0],
        )
        _logger.debug("Detected %d object(s)", len(detections))
        return detections

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _letterbox(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, float, int, int]:
        """Resize and pad a frame to the model's square input.

        Returns:
            (blob, ratio, pad_x, pad_y) where ratio scales model-space
            boxes back to original pixels and (pad_x, pad_y) is the
            letterbox offset in the padded coordinate system.
        """
        h, w = frame.shape[:2]
        size = self._input_size
        ratio = min(size / w, size / h)
        new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
        resized = cv2.resize(frame, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)

        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        pad_x = (size - new_w) // 2
        pad_y = (size - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        blob = cv2.dnn.blobFromImage(
            canvas, 1.0 / 255.0, (size, size),
            swapRB=True, crop=False,
        )
        return blob, ratio, pad_x, pad_y

    def _parse_outputs(
        self,
        outputs: np.ndarray,
        ratio: float,
        pad_x: int,
        pad_y: int,
        orig_w: int,
        orig_h: int,
    ) -> List[DetectionResult]:
        """Decode the raw [1, 84, 8400] tensor into detections."""
        pred = np.squeeze(outputs)  # (84, 8400)
        if pred.ndim == 2 and pred.shape[0] == 84:
            pred = pred.T  # (8400, 84)

        boxes = pred[:, :4]      # cx, cy, w, h (model-space)
        scores = pred[:, 4:]     # per-class probabilities
        class_ids = np.argmax(scores, axis=1)
        class_scores = scores[np.arange(scores.shape[0]), class_ids]

        # Per-class confidence: some classes need a higher bar to avoid
        # false positives (e.g. "laptop" firing on doors/rectangles).
        class_conf = np.array([
            _HIGH_CONF_CLASSES.get(self._names[int(i)], self._conf)
            for i in class_ids
        ])
        keep_conf = class_scores >= class_conf
        if not np.any(keep_conf):
            return []

        boxes = boxes[keep_conf]
        class_ids = class_ids[keep_conf]
        class_scores = class_scores[keep_conf]

        # Convert centre-wh to corner x1y1x2y2 (model-space, padded).
        x1 = boxes[:, 0] - boxes[:, 2] / 2.0
        y1 = boxes[:, 1] - boxes[:, 3] / 2.0
        x2 = boxes[:, 0] + boxes[:, 2] / 2.0
        y2 = boxes[:, 1] + boxes[:, 3] / 2.0

        rects = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        indices = cv2.dnn.NMSBoxes(
            rects.tolist(),
            class_scores.tolist(),
            self._conf,
            self._iou,
        )
        if isinstance(indices, np.ndarray):
            indices = indices.flatten().tolist()
        elif isinstance(indices, (list, tuple)) and len(indices) > 0:
            indices = [i[0] if isinstance(i, (list, tuple)) else i
                       for i in indices]

        results: List[DetectionResult] = []
        for i in indices:
            cx, cy, w, h = boxes[i]
            cls_id = int(class_ids[i])
            label = (
                self._names[cls_id]
                if 0 <= cls_id < len(self._names) else f"class-{cls_id}"
            )

            x1o = (cx - w / 2 - pad_x) / ratio
            y1o = (cy - h / 2 - pad_y) / ratio
            x2o = (cx + w / 2 - pad_x) / ratio
            y2o = (cy + h / 2 - pad_y) / ratio
            x1o = max(0.0, min(x1o, orig_w))
            y1o = max(0.0, min(y1o, orig_h))
            x2o = max(0.0, min(x2o, orig_w))
            y2o = max(0.0, min(y2o, orig_h))

            box = (int(round(x1o)), int(round(y1o)),
                   int(round(x2o - x1o)), int(round(y2o - y1o)))

            if _looks_like_false_laptop(label, box):
                continue

            results.append(DetectionResult(
                label=label,
                confidence=float(class_scores[i]),
                box=box,
            ))

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results


def _looks_like_false_laptop(label: str,
                             box: Tuple[int, int, int, int]) -> bool:
    """Heuristic: drop 'laptop' boxes that are much taller than wide.

    A real laptop is a wide, low slab (aspect ratio ~1.3-2.5 wide).  A
    door, poster, or wall panel is the opposite — tall and narrow.  When
    the model insists on calling those "laptop", the confidence bar in
    _HIGH_CONF_CLASSES already kills most; this catches the rest.
    """
    if label != "laptop":
        return False
    _, _, w, h = box
    if w <= 0 or h <= 0:
        return True
    # Taller than wide by a clear margin -> not a laptop.
    return h > w * 1.15


def label_detections(
    frame: np.ndarray,
    results: List[DetectionResult],
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Return a copy of the frame annotated with detection boxes/labels."""
    display = frame.copy()
    for r in results:
        x, y, w, h = r.box
        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
        text = f"{r.label} {r.confidence:.2f}"
        label_y = y - 8 if y - 8 > 10 else y + h + 18
        cv2.putText(display, text, (x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return display
