"""Text Recognition (OCR) module.

ocr_engine: OcrEngine, OcrResult, draw_text_boxes.  RapidOCR-based,
            CPU-only ONNX text detection + recognition.
"""
from src.ocr.ocr_engine import (
    OcrEngine,
    OcrResult,
    draw_text_boxes,
)

__all__ = [
    "OcrEngine",
    "OcrResult",
    "draw_text_boxes",
]
