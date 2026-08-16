# PERFORMANCE_RESULTS.md

**Date:** 16 Aug 2026
**Method:** `scripts/audit/benchmark.py` + targeted one-off measurements
**Hardware note:** CPU-only, no GPU. All inputs synthetic (no hardware camera used
for the latency numbers). Real-camera feed FPS measured separately via the web
dashboard.

## Object detection — YOLOv8n (OpenCV DNN, 640×640 letterbox)

| Input frame | first-call (incl. compile) | steady-state avg | range observed |
|---|---|---|---|
| 1280×720 synthetic | 80.7 ms | 65.4 ms | 58.3–67.2 ms |
| 1280×720 synthetic (repeat) | — | 69.3 ms | 57.9–76.0 ms |
| 640×480 synthetic | — | 61.9–189 ms | 61.9–332.7 ms |

- ≈ **13–17 detection/s** at 720p steady state.
- Latency is **highly variable** (58 ms → 330 ms across identical runs); the model
  loads once (~1 s) and the first inference includes compile cost.
- Detection on the real camera feed added negligible visible stall because the server
  decouples grab from inference.

## OCR — RapidOCR (CPU, ONNX)

| Input | first call (incl. init) | steady-state avg |
|---|---|---|
| 640×480 synthetic text ("EXIT 12") | 4,502 ms | 4,739 ms (min 4,172 ms) |

- **~4.1–4.7 s per OCR frame** — the dominant CPU bottleneck of the whole pipeline.
- OcrEngine constructor itself is fast; the cost is inference on the first real image.

## Navigation / decision / tracking (micro-benchmarks)

| Operation | avg latency |
|---|---|
| `direction_of` | < 0.01 ms |
| `distance_estimate` | 0.14 ms |
| `DecisionEngine.decide` | 0.01 ms |
| `IoUTracker.update` | < 0.01 ms |
| `TrackingMonitor.events` | < 0.01 ms |

These add negligible overhead; the frame budget is entirely inference-bound.

## End-to-end loop (stub camera, no hardware)

| Metric | Value |
|---|---|
| Loop frames processed in 8.7 s | 12 |
| Loop rate | 1.37 fps |
| Detection calls (every 2nd frame) | 6 (~0.7 detect/s) |

- The loop is **bounded by OCR latency**, not detection: every 10th frame triggers a
  ~4.5 s OCR call that stalls the single-threaded loop.
- A detection-only pass would run at ~13–17 fps (per the numbers above), i.e. the
  pipeline is ~10× faster with OCR disabled.

## Web dashboard (real camera, 1280×720)

| Endpoint / metric | Value |
|---|---|
| `GET /api/state` | 200, JSON valid |
| Feed FPS reported | 9.3 |
| Resolution | 1280×720 |
| Latency (AI loop) reported | 0.1 ms (idle scene, no detections) |

- The grab thread decouples the live feed from the (slow) inference thread, so the
  feed stays responsive (~9 fps) even when OCR blocks inference.
- `assist_app.py` runs the same components **synchronously**, so it inherits the
  ~1.4 fps OCR-bound cadence instead.

## Key takeaways

1. CPU inference budget: YOLO ~60–75 ms/frame; RapidOCR ~4.5 s/frame.
2. OCR is the thing to optimise or throttle for real-time feel.
3. The server's thread split is the right architecture for absorbing the OCR cost.
4. All numbers are for this specific machine; relative costs (OCR >> YOLO) are
   expected to hold on comparable CPU-only laptops.