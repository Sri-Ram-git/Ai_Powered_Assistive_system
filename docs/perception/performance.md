# Performance (Phases 6-7, 21-23)

## Architecture: real-time async pipeline

The desktop app now runs the async engine (`src/core/AsyncVisionPipeline`)
instead of a blocking loop:

```
grab thread     camera → latest-frame store (never blocks, never encodes)
detect thread   YOLO → hardened IoU tracker → decision/monitor → planner
                (runs at its own pace; publishes latest results)
speech          engine phrase → SpeechQueue → SpeechOutput (non-blocking)
display loop    latest frame + latest tracks overlay → imshow at camera FPS
```

- **Latest-frame semantics** (`FrameManager`): a slow worker never
  blocks the camera; intermediate frames are overwritten, never queued.
- **Detect every N, track at its own rate**: detection runs every 2
  frames (configurable); the tracking/decision/safety stages run in the
  same worker but are cheap.
- **OCR is off the vision thread** and, by default, not even loaded
  (`ocr.enabled: false`, Phase 21).  No RapidOCR model load at startup,
  no worker thread, no blocking.
- **JPEG encoding is opt-in** (`encode_jpeg`), so the desktop path never
  wastes CPU producing web frames.

## Phase 22 — targets vs measured (16 Aug 2026, this machine)

| Target | Required | Measured | Verdict |
|---|---|---|---|
| Camera/display FPS | ≥ 20 if hardware allows | camera ceiling **9.9**, display **9.7** | hardware-limited; ceiling honest |
| Detection throughput | ≥ 10 FPS | ~84 ms/frame ⇒ **~12 Hz** detections | met |
| Tracking = camera FPS | display smoothness | tracker runs per detect tick; display = camera FPS | met |
| Speech non-blocking | never blocks vision | SpeechQueue + worker; vision loop never calls TTS | met |
| OCR in real-time path | none | disabled by default; not loaded | met |

### Before / after (OCR off)

| Metric | BEFORE | AFTER | Δ |
|---|---|---|---|
| Desktop display FPS | 3.23 | 9.72 | ×3 |
| YOLO latency under load | 510-663 ms | ~84 ms mean | ~6× |
| Blocking stages in loop | camera+dect+ocr+display | none (async) | — |

## Phase 23 — visual debug mode

Press `d` in the desktop app to overlay: camera FPS, YOLO latency, track
count, pending speech, and the **raw** (un-smoothed, cyan) boxes against
the **smoothed** boxes — so the smoothing/tracking effect is visible.

## Tooling

- `scripts/benchmark/perception_benchmark.py` — camera / detect /
  blocking / pipeline / threshold-sweep modes, JSON report.
- `performance/benchmarks/run_all.py` — broader stage benchmark.