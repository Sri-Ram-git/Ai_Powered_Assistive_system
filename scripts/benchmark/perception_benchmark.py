"""Perception pipeline benchmark (Phase 1 + Phases 4/5/22).

Measures the *actual* runtime behaviour of the vision pipeline, headless
(no window, no speech), and reports honest numbers for:

    camera    raw camera FPS (no AI), frame jitter, dropped frames
    detect    YOLO inference latency (median/mean/p90) for a model+conf
    blocking  the CURRENT desktop path (single loop: read -> detect ->
              track) with OCR disabled — this is the "BEFORE" number
    pipeline  the async engine (grab/detect threads, latest-frame
              semantics) with OCR disabled — the "AFTER" number

Everything writes a JSON report to ``performance/results/``.

Usage:
    python scripts/benchmark/perception_benchmark.py [--camera 0] [--model models/yolov8s.onnx] [--conf 0.35] [--seconds 8] [--mode all]
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.camera import Camera  # noqa: E402
from src.detection import YoloDetector  # noqa: E402
from src.tracking import IoUTracker  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402

_logger = setup_logger("PerceptionBenchmark")
RESULTS_DIR = PROJECT_ROOT / "performance" / "results"


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(p / 100.0 * (len(ordered) - 1))))
    return round(ordered[idx], 1)


def _stats_ms(values: List[float]) -> Dict:
    if not values:
        return {"n": 0, "mean_ms": None, "median_ms": None, "p90_ms": None}
    return {
        "n": len(values),
        "mean_ms": round(float(np.mean(values)), 1),
        "median_ms": round(float(np.median(values)), 1),
        "p90_ms": _percentile(values, 90),
    }


def _system_stats() -> Dict:
    try:
        import psutil

        proc = psutil.Process()
        return {
            "cpu_percent": proc.cpu_percent(interval=0.5),
            "ram_mb": round(proc.memory_info().rss / (1024 * 1024), 1),
        }
    except ImportError:
        return {"cpu_percent": None, "ram_mb": None, "note": "psutil missing"}


def bench_camera(camera_id: int, seconds: float) -> Dict:
    """Raw camera FPS + frame-arrival jitter with no AI running."""
    with Camera(camera_id=camera_id, resolution=(1280, 720)) as cam:
        warmup = 30
        for _ in range(warmup):
            cam.read()

        intervals: List[float] = []
        start = time.monotonic()
        frames = 0
        last = time.monotonic()
        while time.monotonic() - start < seconds:
            cam.read()
            frames += 1
            now = time.monotonic()
            intervals.append((now - last) * 1000.0)
            last = now

        elapsed = time.monotonic() - start
        fps = frames / elapsed
        return {
            "fps": round(fps, 2),
            "frames": frames,
            "interval_ms": _stats_ms(intervals),
            "frame_interval_p95_ms": _percentile(intervals, 95),
        }


def bench_detect(model_path: str, conf: float, seconds: float) -> Dict:
    """YOLO inference latency over live camera frames (OCR off)."""
    detector = YoloDetector(model_path, input_size=640,
                            conf_threshold=conf, iou_threshold=0.45)
    with Camera(camera_id=0, resolution=(1280, 720)) as cam:
        for _ in range(3):
            cam.read()

        latencies: List[float] = []
        start = time.monotonic()
        while time.monotonic() - start < seconds:
            frame = cam.read()
            t0 = time.monotonic()
            dets = detector.detect(frame)
            latencies.append((time.monotonic() - t0) * 1000.0)

        return {
            "model": Path(model_path).name,
            "conf": conf,
            "objects_last_frame": len(dets),
            "latency_ms": _stats_ms(latencies),
        }


def bench_blocking(model_path: str, conf: float, detect_every: int,
                   seconds: float) -> Dict:
    """Replicate the CURRENT desktop loop headlessly (OCR disabled).

    This is exactly what ``src/assist/assist_app.py`` does today minus
    the window/speech: one thread reading, detecting every N frames and
    tracking — every slow stage blocks the next frame.  OCR is off.
    """
    detector = YoloDetector(model_path, input_size=640,
                            conf_threshold=conf, iou_threshold=0.45)
    tracker = IoUTracker(iou_threshold=0.3, max_missed=8)
    with Camera(camera_id=0, resolution=(1280, 720)) as cam:
        for _ in range(3):
            cam.read()

        start = time.monotonic()
        frames = 0
        detects = 0
        t_detect_total = 0.0
        max_frame_ms = 0.0
        tracks = []
        while time.monotonic() - start < seconds:
            t_frame = time.monotonic()
            frame = cam.read()
            frames += 1
            if frames % detect_every == 0:
                t0 = time.monotonic()
                dets = detector.detect(frame)
                t_detect_total += (time.monotonic() - t0)
                detects += 1
                tracks = tracker.update(dets)
            frame_ms = (time.monotonic() - t_frame) * 1000.0
            max_frame_ms = max(max_frame_ms, frame_ms)

        elapsed = time.monotonic() - start
        loop_fps = frames / elapsed
        return {
            "model": Path(model_path).name,
            "conf": conf,
            "detect_every": detect_every,
            "loop_fps": round(loop_fps, 2),
            "frames": frames,
            "detect_runs": detects,
            "mean_detect_ms": round(t_detect_total / max(1, detects), 1),
            "worst_single_frame_ms": round(max_frame_ms, 1),
            "tracks_last": len(tracks),
        }


def bench_pipeline(seconds: float) -> Dict:
    """Async engine (grab/detect threads) with OCR disabled.

    Uses the real ``AsyncVisionPipeline`` so the numbers reflect the
    production engine, not a hand-rolled imitation.
    """
    from src.core.config import PipelineConfig
    from src.core.pipeline import AsyncVisionPipeline

    cfg = PipelineConfig()
    cfg.ocr_enabled = False
    cfg.navigation_enabled = False
    cfg.model_path = "models/yolov8s.onnx"

    pipe = AsyncVisionPipeline(config=cfg)
    pipe.start(timeout=5.0)
    if not pipe.state_snapshot().get("running"):
        return {"error": "pipeline failed to start"}

    fps_samples: List[float] = []
    start = time.monotonic()
    while time.monotonic() - start < seconds:
        fps_samples.append(pipe.state_snapshot().get("fps", 0.0))
        time.sleep(0.5)

    metrics = pipe.metrics
    yolo_sum = metrics.summary("yolo_latency_ms") if metrics else None
    frames_processed = (
        metrics.counter("frames_processed") if metrics else None)

    report = {
        "display_fps": round(float(np.median(fps_samples)), 2),
        "yolo_mean_ms": round(float(yolo_sum["mean"]), 1)
        if yolo_sum else None,
        "frames_processed": frames_processed,
    }
    pipe.stop()
    return report


def sweep_confidences(model_path: str, seconds: float,
                      thresholds=(0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)) -> Dict:
    """Phase 4: detections + latency at each confidence threshold.

    Runs the *same* camera frames through the detector at each
    threshold and reports average detection count and confidence — the
    evidence for choosing a default threshold.
    """
    detector = YoloDetector(model_path, input_size=640,
                            conf_threshold=0.10, iou_threshold=0.45)
    with Camera(camera_id=0, resolution=(1280, 720)) as cam:
        for _ in range(3):
            cam.read()

        per_conf: Dict[float, List] = {t: [] for t in thresholds}
        start = time.monotonic()
        while time.monotonic() - start < seconds:
            frame = cam.read()
            raw = detector._parse_outputs
            # Re-run parsing at each threshold without re-inferring.
            blob, ratio, pad_x, pad_y = detector._letterbox(frame)
            detector._net.setInput(blob)
            outputs = detector._net.forward()
            for t in thresholds:
                saved = detector._conf
                detector._conf = t
                try:
                    dets = detector._parse_outputs(
                        outputs, ratio, pad_x, pad_y,
                        frame.shape[1], frame.shape[0])
                finally:
                    detector._conf = saved
                per_conf[t].append((len(dets),
                                    [d.confidence for d in dets]))

        report = {}
        for t in thresholds:
            counts = [c for c, _ in per_conf[t]]
            confs = [cf for _, cs in per_conf[t] for cf in cs]
            report[str(t)] = {
                "mean_detections": round(float(np.mean(counts)), 2)
                if counts else 0.0,
                "max_detections": int(max(counts)) if counts else 0,
                "mean_confidence": round(float(np.mean(confs)), 2)
                if confs else None,
            }
        return report


def _write_report(data: Dict, name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{name}.json"
    out.write_text(json.dumps(data, indent=2, default=str),
                   encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", default=str(PROJECT_ROOT / "models" / "yolov8s.onnx"))
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--mode", default="all",
                        choices=["camera", "detect", "blocking",
                                 "pipeline", "sweep", "all"])
    args = parser.parse_args()

    model_path = str(Path(args.model))
    if not Path(model_path).is_absolute():
        model_path = str(PROJECT_ROOT / model_path)

    print(f"OpenCV {cv2.__version__} | camera {args.camera} | "
          f"{Path(model_path).name} @ conf {args.conf} | "
          f"{args.seconds}s")

    report: Dict = {
        "environment": {"opencv": cv2.__version__},
        "mode": args.mode,
        "system": _system_stats(),
    }

    if args.mode in ("camera", "all"):
        print(f"\n[1] Raw camera FPS ({args.seconds}s, no AI) ...")
        report["camera"] = bench_camera(args.camera, args.seconds)
        print(f"    -> {report['camera']['fps']} fps")
        report["system"] = _system_stats()

    if args.mode in ("detect", "all"):
        print(f"\n[2] YOLO latency ({Path(model_path).name}) ...")
        report["detect"] = bench_detect(model_path, args.conf, args.seconds)
        print(f"    -> {report['detect']['latency_ms']}")
        report["system"] = _system_stats()

    if args.mode in ("blocking", "all"):
        print("\n[3] CURRENT desktop loop (blocking, OCR off) ...")
        report["blocking"] = bench_blocking(
            model_path, args.conf, detect_every=2, seconds=args.seconds)
        print(f"    -> {report['blocking']['loop_fps']} fps")
        report["system"] = _system_stats()

    if args.mode in ("sweep",):
        print("\n[S] Confidence threshold sweep ...")
        report["sweep"] = sweep_confidences(model_path, args.seconds)
        print(json.dumps(report["sweep"], indent=2))
        report["system"] = _system_stats()

    if args.mode in ("pipeline", "all"):
        print("\n[4] Async engine (OCR off) ...")
        report["pipeline"] = bench_pipeline(args.seconds)
        print(f"    -> display_fps={report['pipeline'].get('display_fps')}")

    out = _write_report(report, "perception_baseline")
    print(f"\nReport: {out}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
