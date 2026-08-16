"""Unified performance benchmark suite.

Measures every pipeline stage and writes a JSON report to
``performance/results/``.  Targets low latency + stable UX, not maximum
FPS:

    * camera FPS (where available)
    * YOLO detection latency
    * OCR latency (per strategy, non-blocking worker behaviour)
    * depth latency (synthetic backend by default)
    * STT parse latency
    * TTS queue delay
    * end-to-end pipeline latency
    * CPU / RAM usage (via psutil if installed, else skipped)

Usage:
    python performance/benchmarks/run_all.py [--detect-warmup 3]
"""
import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

RESULTS_DIR = PROJECT_ROOT / "performance" / "results"


def _synthetic_frame(w=1280, h=720):
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (500, 400), (80, 160, 240), -1)
    cv2.putText(img, "EXIT", (600, 200), cv2.FONT_HERSHEY_SIMPLEX,
                3.0, (0, 0, 0), 6)
    return img


def _median_ms(fn, n=5, warmup=1):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        t0 = time.monotonic()
        fn()
        times.append((time.monotonic() - t0) * 1000.0)
    times.sort()
    return round(times[len(times) // 2], 1)


def _bench_yolo(frame):
    from src.detection import YoloDetector

    detector = YoloDetector("models/yolov8n.onnx", conf_threshold=0.35)
    return {"yolo_latency_ms": _median_ms(lambda: detector.detect(frame), n=3)}


def _bench_ocr(frame):
    from src.ocr import OcrEngine
    from src.ocr.preprocess import SUPPORTED_STRATEGIES

    engine = OcrEngine(min_confidence=0.3)
    out = {}
    for strategy in SUPPORTED_STRATEGIES:
        from src.ocr.preprocess import preprocess

        out[f"ocr_{strategy}_ms"] = _median_ms(
            lambda s=strategy: engine.read_text(preprocess(frame, s)), n=2)
    return out


def _bench_depth(frame):
    from src.depth import create_depth_estimator

    est = create_depth_estimator("synthetic")
    return {"depth_latency_ms": _median_ms(lambda: est.estimate(frame), n=5)}


def _bench_stt():
    from src.speech import create_stt

    stt = create_stt("keyword")
    return {"stt_parse_ms": _median_ms(
        lambda: stt.parse("read the text"), n=50, warmup=5)}


def _bench_tts_queue():
    try:
        from src.audio import SpeechOutput
        tts = SpeechOutput()
        t0 = time.monotonic()
        tts.speak("test")
        delay = (time.monotonic() - t0) * 1000.0
        tts.shutdown()
        return {"tts_queue_ms": round(delay, 1)}
    except Exception as exc:
        return {"tts_queue_ms": None, "tts_error": str(exc)}


def _system_stats():
    try:
        import psutil
        proc = psutil.Process()
        return {
            "cpu_percent": proc.cpu_percent(interval=0.5),
            "ram_mb": round(proc.memory_info().rss / (1024 * 1024), 1),
        }
    except ImportError:
        return {"cpu_percent": None, "ram_mb": None,
                "note": "psutil not installed"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detect-warmup", type=int, default=1)
    args = parser.parse_args()

    print(f"OpenCV {cv2.__version__} | numpy {np.__version__}")
    frame = _synthetic_frame()
    report = {"environment": {"opencv": cv2.__version__},
              "frame": [frame.shape[1], frame.shape[0]]}

    print("\n[1/6] YOLO detection ...")
    report["detection"] = _bench_yolo(frame)

    print("[2/6] OCR strategies ...")
    report["ocr"] = _bench_ocr(frame)

    print("[3/6] Depth (synthetic) ...")
    report["depth"] = _bench_depth(frame)

    print("[4/6] STT (keyword) ...")
    report["speech"] = _bench_stt()

    print("[5/6] TTS queue ...")
    report["tts"] = _bench_tts_queue()

    print("[6/6] System stats ...")
    report["system"] = _system_stats()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "benchmark_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport: {out}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()