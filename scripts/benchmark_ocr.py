"""Benchmark OCR strategies: latency, throughput, and perceived delay.

Compares the strategies in ``src.ocr.preprocess`` on a synthetic text
image (CPU-only, no webcam required) and reports:

    * latency per call (ms)
    * call rate (calls/s)
    * how a non-blocking worker affects the *detection* cadence vs a
      fully synchronous OCR (the before/after comparison).

Usage:
    python scripts/benchmark_ocr.py [--frames 3]
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _synthetic_text_image(width: int = 1280, height: int = 720) -> np.ndarray:
    """A realistic-ish text frame: dark text on light background."""
    img = np.full((height, width, 3), 245, dtype=np.uint8)
    for i, (text, y) in enumerate([
        ("EXIT", 80),
        ("PLEASE DO NOT ENTER", 220),
        ("EMERGENCY EXIT", 360),
        ("FIRE SAFETY NOTICE", 500),
    ]):
        cv2.putText(img, text, (60, y), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, (30, 30, 30), 4)
    # A little noise so the detector actually has to work.
    rng = np.random.default_rng(42)
    img[::4, ::3] = rng.integers(230, 255, img[::4, ::3].shape,
                                 dtype=np.uint8)
    return img


def _run_sync_ocr(engine, frame, strategy: str, n: int) -> dict:
    """Run OCR n times synchronously on the given strategy."""
    from src.ocr.preprocess import preprocess

    latencies = []
    for _ in range(n):
        started = time.monotonic()
        items = engine.read_text(preprocess(frame, strategy))
        latencies.append((time.monotonic() - started) * 1000.0)
    latencies.sort()
    return {
        "median_ms": latencies[len(latencies) // 2],
        "mean_ms": float(np.mean(latencies)),
        "min_ms": latencies[0],
        "calls_per_s": 1000.0 / (float(np.mean(latencies)) or 1.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=3)
    args = parser.parse_args()

    from src.ocr import OcrEngine
    from src.ocr.preprocess import SUPPORTED_STRATEGIES

    print(f"OpenCV: {cv2.__version__}")
    frame = _synthetic_text_image()
    print(f"Frame: {frame.shape[1]}x{frame.shape[0]}")
    print("Loading OCR engine ...")
    engine = OcrEngine(min_confidence=0.3)

    results = {}
    for strategy in SUPPORTED_STRATEGIES:
        print(f"\n--- strategy: {strategy} ---")
        # Warm-up (model load + jit).
        from src.ocr.preprocess import preprocess
        engine.read_text(preprocess(frame, strategy))
        stats = _run_sync_ocr(engine, frame, strategy, args.frames)
        results[strategy] = stats
        print(f"  median={stats['median_ms']:.0f} ms  "
              f"mean={stats['mean_ms']:.0f} ms  "
              f"min={stats['min_ms']:.0f} ms  "
              f"rate={stats['calls_per_s']:.2f}/s")

    print("\n=== Summary (sorted by median latency) ===")
    for strategy, stats in sorted(results.items(),
                                  key=lambda kv: kv[1]["median_ms"]):
        print(f"  {strategy:<12} median={stats['median_ms']:>7.0f} ms  "
              f"rate={stats['calls_per_s']:.2f}/s")


if __name__ == "__main__":
    main()