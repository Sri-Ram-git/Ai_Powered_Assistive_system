"""Text Recognition (OCR) module.

ocr_engine:   OcrEngine, OcrResult, draw_text_boxes.  RapidOCR-based,
              CPU-only ONNX text detection + recognition.
preprocess:   Preprocessing strategies (none/gray/threshold/contrast/
              adaptive/sharpen/downscale).
policy:       Object -> OCR eligibility policy (configs/ocr_policy.yaml).
roi:          Padded, validated object ROI extraction + smart upscaling.
text_presence: Cheap heuristic gate deciding whether an ROI has text.
object_ocr:   ObjectOcrResult, validation, variants, TrackOcrStore,
              OcrTrigger.
object_worker: ObjectOcrWorker (async, newest-request-wins, timeout).
"""
from src.ocr.ocr_engine import (
    OcrEngine,
    OcrResult,
    draw_text_boxes,
)
from src.ocr.object_ocr import (
    ObjectOcrResult,
    TrackOcrEntry,
    TrackOcrStore,
)
from src.ocr.object_worker import ObjectOcrWorker
from src.ocr.policy import OcrPolicy

__all__ = [
    "OcrEngine",
    "OcrResult",
    "ObjectOcrResult",
    "ObjectOcrWorker",
    "OcrPolicy",
    "TrackOcrEntry",
    "TrackOcrStore",
    "draw_text_boxes",
]