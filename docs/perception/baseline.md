# Perception baseline — BEFORE (Phase 1)

Measured 16 Aug 2026 on the local Windows machine, camera 0 (laptop
webcam), OpenCV 5.0.0, all stages in-process, **OCR disabled** in every
measurement (Phase 21).  Numbers are real, from
`scripts/benchmark/perception_benchmark.py` (see
`performance/results/perception_baseline.json`).

## The problem this hardening fixes

The desktop app (`src/assist/assist_app.py`) ran **one blocking loop**:
`cam.read()` → YOLO (yolov8s) → OCR (every 10 frames) → `cv2.imshow`.
Every slow stage froze the next frame.  The brief's "camera → detection
→ label → tracking → smooth display → intelligent speech" all happened
sequentially in one thread.

## Measured BEFORE numbers

| Metric | Value | Notes |
|---|---|---|
| Raw camera FPS (no AI, 1280x720) | **9.89 fps** | hardware ceiling on this laptop camera |
| YOLOv8s live-frame latency | **median 510 ms, mean 528 ms** | inflated by CPU contention (see below) |
| Blocking desktop loop FPS (OCR off) | **3.23 fps** | the user's real experience pre-hardening |
| Async engine display FPS (OCR off, JPEG on) | **4.11 fps** | grab loop wasted CPU on JPEG encode |
| YOLOv8s latency while engine encodes JPEG | **mean 663 ms** | encode + inference fighting for CPU |
| CPU / RAM during run | ~3 % / ~51 MB (idle-ish) | machine under no other load |

### Root causes found

1. **Blocking loop** — one thread for camera + detection + OCR + display;
   display FPS ≈ YOLO throughput (~3 fps).
2. **JPEG encode on every frame** — the async engine's grab loop encoded
   a 1280x720 JPEG per frame for the web feed, dragging the desktop path
   too.
3. **OCR in the real-time path** — when enabled, OCR blocked for seconds
   every 10 frames; it must live entirely off the vision thread (or be
   off).
4. **Tracker** — IoU-only association: no class consistency (a "chair"
   box could steal a "person" ID), no box smoothing (jitter), no label
   voting (label flips), no confidence smoothing.
5. **Detector heuristics hid failures** — hardcoded per-class confidence
   floors and a "drop tall laptops" filter could silently discard real
   objects; config did not flow from YAML into the engine (conf was
   hardcoded 0.35).

## Baseline artifacts

- `scripts/benchmark/perception_benchmark.py` — camera / detect /
  blocking / pipeline / sweep modes.
- `performance/results/perception_baseline.json` — raw JSON.
- Live visualiser output: `evaluation/object_detection/raw/scene_vis.png`
  (0 objects visible in the empty room at conf 0.40 — honest).