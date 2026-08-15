"""Object Detection module (YOLOv8 via OpenCV DNN).

detector: YoloDetector, DetectionResult, label_detections.  Runs a
          YOLOv8 ONNX export on the CPU with OpenCV's DNN module —
          no PyTorch, no GPU required.
"""
from src.detection.detector import (
    DetectionResult,
    YoloDetector,
    label_detections,
)

__all__ = [
    "DetectionResult",
    "YoloDetector",
    "label_detections",
]
