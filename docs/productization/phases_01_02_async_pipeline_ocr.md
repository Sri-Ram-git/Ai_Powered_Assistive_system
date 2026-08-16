# Phase 1 + 2 — Async Core Pipeline & OCR Decoupling

## What changed

The biggest architectural weakness was that OCR (~2.8–4.7 s/frame on CPU)
blocked the synchronous `assist_app.py` loop, and in the server it ran
inside the single inference thread.

### New core engine: `src/core/`

```
Camera
  │  (grab thread, never runs AI)
  ▼
FrameManager ──────────────► annotated JPEG ──► UI / /video_feed
  │
  ├─► detect thread: YOLO → IoUTracker → DecisionEngine / TrackingMonitor
  │       └─► publishes latest tracks/guidance (LatestResults)
  │
  └─► OcrWorker (background thread + latest-frame slot)
          └─► publishes latest OCR result (never blocking)
```

* `src/core/frame_manager.py` — `FrameManager`: thread-safe latest-frame
  store with published/consumed/dropped counters and a rolling FPS
  estimate.  Slow workers naturally see only the newest frame (implicit
  frame dropping — exactly right for real time).
* `src/core/results.py` — `LatestResults`: per-stage latest results
  (detections, tracks, OCR, guidance) published atomically.
* `src/core/pipeline.py` — `AsyncVisionPipeline`: coordinates grab +
  detect + OCR worker threads.  **No Flask dependency.**
* `src/core/config.py` — `PipelineConfig` moved here so the core is
  usable without the server layer.

### OCR worker: `src/ocr/worker.py` + `src/ocr/preprocess.py`

* `OcrWorker` runs OCR on a dedicated thread with a single-slot "latest
  frame" design — if OCR is busy and a newer frame arrives, the newer
  frame replaces the pending one, so a slow OCR never builds a backlog.
  Consumers read the latest *completed* result via `latest_result()`.
* `preprocess.py` — pluggable input strategies (`none`, `gray`,
  `threshold`, `contrast`, `downscale`, `downscale2`) for benchmarking.

### Server delegation

`src/server/pipeline.py` is now a thin wrapper over `AsyncVisionPipeline`
(the camera factory is resolved dynamically so tests can still patch
`src.server.pipeline.Camera`).  `PipelineConfig` is re-exported from
`src/core/config.py`.

## Why it matters

* OCR can never freeze camera capture, detection, tracking, the UI, or
  speech — it runs on its own thread and its result is consumed lazily.
* The core pipeline is now usable headless (no Flask), enabling the
  API/core split (Phase 13) and testing without a webcam.

## Benchmark (CPU-only, RapidOCR)

`python scripts/benchmark_ocr.py` on a 1280×720 synthetic text frame:

| Strategy   | median latency | rate |
|------------|----------------|------|
| none       | 4997 ms        | 0.21/s |
| threshold  | 5114 ms        | 0.20/s |
| contrast   | 5130 ms        | 0.20/s |
| downscale  | 5214 ms        | 0.20/s |
| downscale2 | 5285 ms        | 0.20/s |
| gray       | 5293 ms        | 0.20/s |

**Finding:** on this machine RapidOCR is CPU-inference-bound; input
preprocessing gives no meaningful latency reduction.  The decisive fix is
the *asynchronous worker* (decoupling), not input size.  Therefore
RapidOCR is kept (benchmark does not justify a swap), and the pipeline is
restructured so OCR latency no longer multiplies end-to-end latency.

> Note: this is one machine's measurement; the trade-off table for
> RapidOCR vs PaddleOCR vs EasyOCR is documented in
> `docs/productization/model_selection.md` (Phase 25).