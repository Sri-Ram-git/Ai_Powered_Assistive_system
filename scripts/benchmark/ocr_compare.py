"""Compare OCR performance: previous engine config vs this version.

The previous version used RapidOCR's default detection resize policy
(``det_limit_type='min'``, limit 736): any image whose *smaller* side is
under 736 px is upscaled so the min side becomes 736.  For object-aware
OCR this is pathological: a 96x24 bottle label becomes a ~3270x736 image
and a single call takes ~10 s — far worse than the full 1280x720 frame.

This version configures ``det_limit_type='max'`` (only shrink images
whose *largest* side exceeds the limit), which keeps object ROIs near
their natural size.

The script measures, for both engine configs:

    * full 1280x720 frame     (the previous pipeline's per-scan input)
    * mid 320x72 ROI          (a bottle label / book cover)
    * small 96x24 ROI         (tiny label, needs smart upscaling)

and also runs the *object-aware pipeline path* (extract_roi -> presence
gate -> up to 3 preprocessing variants -> best result) to show the
end-to-end cost per accepted object.

Usage:
    python scripts/benchmark/ocr_compare.py [--runs 3]
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _text_image(w: int, h: int, text: str, scale: float) -> np.ndarray:
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    cv2.putText(img, text, (8, int(h * 0.72)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 2)
    return img


def _time_calls(engine, image, n: int) -> tuple:
    engine.read_text(image)  # warm-up
    times = []
    for _ in range(n):
        t = time.monotonic()
        engine.read_text(image)
        times.append((time.monotonic() - t) * 1000.0)
    return float(np.median(times)), float(np.mean(times)), times


def _build_engine(det_limit_type: str, limit: int):
    from rapidocr_onnxruntime import RapidOCR

    rapid = RapidOCR(
        det_model_path=None,
        det_limit_side_len=limit,
        det_limit_type=det_limit_type,
    )

    class _Wrapper:
        def read_text(self, image):
            rapid(np.ascontiguousarray(image))
            return []

    return _Wrapper()


def _object_path(frame, box, variants, engine) -> tuple:
    """Run the object-aware path once; return (status, latency_ms)."""
    from src.ocr.object_ocr import combine_results, run_variants
    from src.ocr.roi import extract_roi
    from src.ocr.text_presence import has_text

    t0 = time.monotonic()
    roi = extract_roi(frame, box, padding=0.1)
    if roi is None:
        return "roi_rejected", (time.monotonic() - t0) * 1000.0
    if not has_text(roi.image):
        return "no_text", (time.monotonic() - t0) * 1000.0
    variant, items, _ = run_variants(engine, roi.image, variants,
                                     stop_confidence=0.92)
    combine_results(items)
    return f"ok/{variant}", (time.monotonic() - t0) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    n = max(1, args.runs)

    full = _text_image(1280, 720, "EXIT PLEASE DO NOT ENTER", 2.0)
    mid = _text_image(320, 72, "COCA COLA", 1.2)
    small = _text_image(96, 24, "COLA", 0.5)

    print(f"=== OCR engine config comparison ({n} runs) ===\n")

    results = {}
    for name, limit_type in (("previous (min limit 736)", "min"),
                             ("this version (max limit 736)", "max")):
        engine = _build_engine(limit_type, 736)
        print(f"--- {name} ---")
        row = {}
        for label, img in (("full 1280x720", full),
                           ("mid 320x72", mid),
                           ("small 96x24", small)):
            med, mean, _ = _time_calls(engine, img, n)
            row[label] = round(med, 1)
            print(f"  {label:<14} median {med:>8.0f} ms  mean {mean:>8.0f} ms")
        results[name] = row
        print()

    print("=== speed-up (median, this version vs previous) ===")
    prev = results["previous (min limit 736)"]
    cur = results["this version (max limit 736)"]
    for label in ("full 1280x720", "mid 320x72", "small 96x24"):
        p, c = prev[label], cur[label]
        speedup = p / c if c > 0 else float("inf")
        print(f"  {label:<14} {p:>8.0f} -> {c:>8.0f} ms  "
              f"({speedup:>6.1f}x faster)")

    print("\n=== object-aware pipeline path (this version) ===")
    from src.ocr import OcrEngine
    from src.ocr.object_ocr import DEFAULT_VARIANTS

    ocr = OcrEngine(min_confidence=0.3)
    variants = DEFAULT_VARIANTS[:3]
    samples = [
        ("label ROI", _text_image(320, 72, "COCA COLA", 1.2), (40, 200, 320, 72)),
        ("small ROI", _text_image(96, 24, "COLA", 0.5), (100, 200, 96, 24)),
        ("blank ROI", np.full((72, 320, 3), 128, dtype=np.uint8), (40, 200, 320, 72)),
    ]
    for label, img, box in samples:
        times = []
        statuses = set()
        for _ in range(n):
            frame = np.full((480, 640, 3), 90, dtype=np.uint8)
            frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]] = img
            status, latency = _object_path(frame, box, variants, ocr)
            statuses.add(status)
            times.append(latency)
        print(f"  {label:<12} median {np.median(times):>6.0f} ms  "
              f"status {sorted(statuses)}")


if __name__ == "__main__":
    main()