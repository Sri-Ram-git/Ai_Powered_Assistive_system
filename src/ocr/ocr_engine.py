"""Text recognition (OCR) using RapidOCR (ONNX, CPU-only).

RapidOCR bundles PaddleOCR's detection + recognition models as ONNX
graphs, so no heavy PaddlePaddle or PyTorch runtime is required.  The
wrapper below provides typed results and automatic input validation.

Usage:
    engine = OcrEngine()
    items = engine.read_text(frame)          # list of OcrResult
    text = engine.text_of(frame)             # " ".join of all text
"""
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from src.utils.exceptions import OcrError
from src.utils.logger import setup_logger

_logger = setup_logger("OcrEngine")


@dataclass
class OcrResult:
    """One recognised text line."""

    text: str
    confidence: float
    box: Tuple[int, int, int, int]  # (x, y, w, h) axis-aligned


class OcrEngine:
    """Wraps RapidOCR for the assistive vision pipeline."""

    def __init__(
        self,
        min_confidence: float = 0.4,
        max_boxes: int = 50,
    ) -> None:
        """Configure the OCR engine.

        Args:
            min_confidence: Drop text lines below this confidence.
            max_boxes: Hard cap on the number of returned lines.

        Raises:
            OcrError: If the OCR engine cannot be initialised.
        """
        self._min_conf = float(min_confidence)
        self._max_boxes = int(max_boxes)
        try:
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR()
        except Exception as exc:  # pragma: no cover - env dependent
            raise OcrError(f"Failed to initialise RapidOCR: {exc}") from exc
        _logger.info("OCR engine ready (min_conf=%.2f)", self._min_conf)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_text(self, image: np.ndarray) -> List[OcrResult]:
        """Recognise text in an image.

        Args:
            image: BGR (or grayscale) numpy image.

        Returns:
            Sorted list of OcrResult, top-most first.

        Raises:
            OcrError: If `image` is not a numpy array or inference fails.
        """
        if image is None:
            return []
        if not isinstance(image, np.ndarray):
            raise OcrError(
                "OCR input must be a numpy image array, "
                f"got {type(image).__name__}"
            )
        if image.size == 0:
            return []

        try:
            raw, _ = self._engine(np.ascontiguousarray(image))
        except Exception as exc:
            raise OcrError(f"OCR inference failed: {exc}") from exc

        results: List[OcrResult] = []
        if not raw:
            return results

        for item in raw:
            box, text, confidence = item[0], item[1], item[2]
            if not text:
                continue
            if float(confidence) < self._min_conf:
                continue
            results.append(OcrResult(
                text=str(text).strip(),
                confidence=float(confidence),
                box=_axis_aligned_box(box),
            ))
            if len(results) >= self._max_boxes:
                break

        results.sort(key=lambda r: r.box[1])  # top-most first
        _logger.debug("OCR found %d text line(s)", len(results))
        return results

    def text_of(self, image: np.ndarray) -> str:
        """Return all recognised text joined into a single string."""
        return " ".join(r.text for r in self.read_text(image))


def _axis_aligned_box(
    quad: Sequence[Sequence[float]],
) -> Tuple[int, int, int, int]:
    """Convert a 4-point quadrilateral to an (x, y, w, h) rectangle."""
    pts = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
    return (int(round(x0)), int(round(y0)),
            int(round(x1 - x0)), int(round(y1 - y0)))


def draw_text_boxes(
    frame: np.ndarray,
    results: List[OcrResult],
    color: Tuple[int, int, int] = (0, 165, 255),
) -> np.ndarray:
    """Return a copy of the frame with OCR boxes and text drawn on."""
    display = frame.copy()
    for r in results:
        x, y, w, h = r.box
        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
        cv2.putText(display, r.text, (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return display
