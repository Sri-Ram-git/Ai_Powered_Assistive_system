# OCR Module (`src/ocr`)

## Overview

Text recognition using **RapidOCR**, which bundles PaddleOCR's
detection + recognition models as ONNX graphs. It runs on CPU with
ONNX Runtime — no PaddlePaddle or PyTorch installation required.

## Architecture

```
┌──────────────────────────────────────────────┐
│                 ocr/                         │
│  ocr_engine.py                               │
│    OcrEngine                                 │
│      read_text(image) → List[OcrResult]      │
│      text_of(image) → str                    │
│    draw_text_boxes(frame, results)           │
└──────────────────────────────────────────────┘
```

## Text pipeline

1. **Validate input** — must be a non-empty numpy array.
2. **Detect** — RapidOCR's DB (Differentiable Binarization) text
   detector finds text-line quadrilaterals.
3. **Recognise** — each cropped line is run through the CRNN
   recognition model.
4. **Filter & sort** — drop lines below `min_confidence`, cap at
   `max_boxes`, and sort top-most first.

## Data model

`OcrResult`:

| Field | Description |
|---|---|
| `text` | Recognised string (stripped) |
| `confidence` | Model confidence for the line |
| `box` | `(x, y, w, h)` axis-aligned bounding rectangle |

## Function reference

| Member | Description |
|---|---|
| `OcrEngine(min_confidence=0.4, max_boxes=50)` | Initialises RapidOCR |
| `read_text(image)` | → `List[OcrResult]` (top-most first) |
| `text_of(image)` | All recognised text joined with spaces |
| `draw_text_boxes(frame, results)` | Annotated copy with boxes + text |

## Usage

```python
from src.ocr import OcrEngine, draw_text_boxes

ocr = OcrEngine()
items = ocr.read_text(frame)          # List[OcrResult]
text = ocr.text_of(frame)             # " ".join(...)
display = draw_text_boxes(frame, items)
```

## Execution flow

```
python src/assist/assist_app.py    # live pipeline OCRs every N frames
```

## Dependencies

- Python 3.11+, NumPy, OpenCV (drawing)
- `onnxruntime` and `rapidocr-onnxruntime` (CPU)
- RapidOCR downloads/embeds small ONNX models on first use

## Limitations

- Recognition quality depends on font, contrast, and camera angle;
  stylised / skewed text (e.g. printed on buses) can be misread.
- CPU inference adds latency, so the assist app runs OCR only every
  Nth frame rather than on every frame.
- English is the default language; other languages require swapping the
  recognition model.

## Future extensions

- Region-of-interest OCR (only read signs within a zone)
- Language selection via config
- Re-read / confirm OCR after a stable frame (temporal smoothing)
