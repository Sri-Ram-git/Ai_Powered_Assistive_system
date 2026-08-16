"""Audit performance benchmark (temporary, not part of the test suite).

Measures, on synthetic frames only:
  - YoloDetector inference latency (640x640 letterbox)
  - RapidOCR engine inference latency
  - End-to-end pipeline FPS via a stub camera (no hardware)
  - Distance/direction helper latency

Run:  python scripts/audit/benchmark.py
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL = PROJECT_ROOT / "models" / "yolov8n.onnx"


def synth_frame(w=1280, h=720, seed=0):
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    cv2.circle(frame, (w // 2, h // 2), 80, (60, 60, 220), -1)
    cv2.rectangle(frame, (200, 100), (500, 420), (80, 200, 90), -1)
    return frame


def bench(label, fn, n=5, warmup=1):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    avg = sum(times) / len(times)
    print(f"{label:<32} min={times[0]:7.2f}ms  median={times[len(times)//2]:7.2f}ms  "
          f"avg={avg:7.2f}ms")
    return avg


def main():
    if not MODEL.exists():
        print(f"MODEL NOT FOUND: {MODEL}")
        sys.exit(1)

    from src.detection import YoloDetector
    from src.ocr import OcrEngine

    print(f"OpenCV: {cv2.__version__} | numpy: {np.__version__}")
    print(f"Model: {MODEL.name} ({MODEL.stat().st_size / 1e6:.1f} MB)\n")

    # --- YOLO detection latency at 1280x720 (letterboxed to 640x640) ---
    frame = synth_frame(1280, 720)
    detector = YoloDetector(str(MODEL))
    print("=== Object detection (1280x720 synthetic, CPU) ===")
    n = 10
    t0 = time.perf_counter()
    results = detector.detect(frame)
    first = (time.perf_counter() - t0) * 1000.0
    print(f"first-inference (incl. compile) = {first:.1f}ms, detections={len(results)}")
    bench("steady-state detect", lambda: detector.detect(frame), n=n, warmup=2)

    # --- OCR latency ---
    print("\n=== OCR (RapidOCR, synthetic text image) ===")
    text_frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    cv2.putText(text_frame, "EXIT 12", (100, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 0), 5)
    ocr = OcrEngine(min_confidence=0.3)
    n = 3
    t0 = time.perf_counter()
    items = ocr.read_text(text_frame)
    first_ocr = (time.perf_counter() - t0) * 1000.0
    print(f"first OCR call (incl. init) = {first_ocr:.1f}ms, lines={len(items)}")
    bench("steady-state OCR", lambda: ocr.read_text(text_frame), n=n, warmup=1)

    # --- Navigation helpers ---
    from src.navigation.guidance import direction_of, distance_estimate
    print("\n=== Navigation helpers (micro) ===")
    box = (300, 100, 200, 400)
    bench("direction_of", lambda: direction_of(box, 1280), n=50, warmup=10)
    bench("distance_estimate", lambda: distance_estimate(box, 720, 1.7), n=50, warmup=10)

    # --- Decision engine ---
    from src.decision import DecisionEngine, FrameSummary
    from src.tracking import IoUTracker, TrackingMonitor
    print("\n=== Decision + tracking (micro) ===")
    summary = FrameSummary(detections=results, ocr_items=items,
                           frame_w=1280, frame_h=720, read_ocr_text=True)
    engine = DecisionEngine(cooldown_seconds=0.0)
    engine.reset()
    bench("DecisionEngine.decide", lambda: engine.decide(summary), n=50, warmup=5)

    tracker = IoUTracker()
    monitor = TrackingMonitor()
    dets = results[:3] if results else []
    bench("IoUTracker.update", lambda: tracker.update(dets), n=50, warmup=5)
    tracks = tracker.active_tracks
    bench("TrackingMonitor.events",
          lambda: monitor.events(tracks, 1280, 720), n=50, warmup=5)

    # --- End-to-end pipeline FPS via stub camera (no hardware) ---
    print("\n=== End-to-end pipeline (stub camera, no hardware) ===")
    _run_pipeline_bench(frame)

    print("\nDone.")


def _run_pipeline_bench(frame):
    """Simulate the assist_app loop for ~8s: detect/track/OCR on a fixed
    synthetic frame, counting loop iterations that complete."""
    import src.camera.camera as camera_mod

    class StubCam:
        camera_id = 0
        resolution = (1280, 720)
        actual_fps = 30.0

        def __init__(self, frame):
            self._frame = frame

        def read(self):
            return self._frame.copy()

        def stop(self):
            pass

    # Monkey-patch Camera so YoloDetector-style imports still work is not
    # needed; we drive the modules directly at the same cadence as the app.
    from src.detection import YoloDetector
    from src.ocr import OcrEngine
    from src.decision import DecisionEngine, FrameSummary
    from src.tracking import IoUTracker, TrackingMonitor

    detector = YoloDetector(str(MODEL))
    ocr = OcrEngine(min_confidence=0.3)
    tracker = IoUTracker()
    monitor = TrackingMonitor()
    engine = DecisionEngine(cooldown_seconds=0.0, read_ocr_text=True)

    st = StubCam(frame)
    detect_every = 2
    ocr_every = 10

    n_frames = 0
    t0 = time.perf_counter()
    deadline = t0 + 8.0
    last_ocr = []
    tracks = []
    while time.perf_counter() < deadline:
        f = st.read()
        n_frames += 1
        if n_frames % detect_every == 0:
            dets = detector.detect(f)
            tracks = tracker.update(dets)
        if n_frames % ocr_every == 0:
            last_ocr = ocr.read_text(f)
        monitor.events(tracks, f.shape[1], f.shape[0])
        engine.decide(FrameSummary(detections=[t for t in tracks],
                                   ocr_items=last_ocr,
                                   frame_w=f.shape[1], frame_h=f.shape[0]))
    elapsed = time.perf_counter() - t0
    fps = n_frames / elapsed
    print(f"loop frames      = {n_frames}")
    print(f"elapsed          = {elapsed:.2f}s")
    print(f"loop rate        = {fps:.2f} fps (fixed 1280x720 synthetic frame)")

    # Inference-only rate (detection every 2nd frame):
    det_frames = n_frames // detect_every
    print(f"inference calls  = {det_frames} detections in {elapsed:.2f}s "
          f"-> ~{det_frames / elapsed:.1f} detect/s")

    camera_mod.Camera = StubCam  # keep linters quiet; unused


if __name__ == "__main__":
    main()