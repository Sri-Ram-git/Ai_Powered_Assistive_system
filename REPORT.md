# Technical Audit — AI-Powered Assistive Vision System

**Audit date:** 16 Aug 2026
**Audited by:** opencode (automated, evidence-based)
**Repository:** `C:\Users\SRIRAM\myprojectss\project_repo\ai_vision_system`
**Branch:** `master` (clean working tree, up to date with `origin/master`)

> **How to read this report.** Every claim is backed by evidence gathered during this
> audit: source inspection, `pytest` runs, performance benchmarks, failure-mode
> exercises, live web-dashboard checks, and `git` inspection. No feature is reported
> as working unless it was observed or its code path was read and exercised.
>
> Status vocabulary:
> - **IMPLEMENTED** — code exists and was verified to work (tested or executed).
> - **PARTIALLY IMPLEMENTED** — code exists but is limited, incomplete, or only
>   partially verified.
> - **NOT IMPLEMENTED** — absent or clearly stubbed.
> - **NOT VERIFIED** — cannot be confirmed in this environment (needs hardware or
>   external conditions we could not satisfy).

---

## 1. Executive Summary

**Status: IMPLEMENTED**

This is a working, modular computer-vision assistive system. Weeks 1–3 deliverables
are present and operational:

| Capability | Status | Evidence |
|---|---|---|
| Camera capture, mirror, FPS, screenshots, threaded recording, draggable HUD | IMPLEMENTED | `src/camera/*`, live camera smoke test, tests |
| Image fundamentals & processing utilities | IMPLEMENTED | `src/image_fundamentals`, `src/image_processing`, tests |
| Morphology & shape detection | IMPLEMENTED | `src/morphology`, tests |
| Live Vision Playground app | IMPLEMENTED | `src/playground/playground.py` |
| YOLOv8 object detection (ONNX via OpenCV DNN, CPU-only) | IMPLEMENTED | `src/detection/detector.py`, tests, bench |
| IoU multi-object tracking | IMPLEMENTED | `src/tracking/tracker.py`, tests |
| Tracking guidance monitor (speech on change) | IMPLEMENTED | `src/tracking/monitor.py`, tests |
| RapidOCR text recognition (CPU ONNX) | IMPLEMENTED | `src/ocr/ocr_engine.py`, tests, bench |
| Navigation guidance (direction, distance, cues) | IMPLEMENTED | `src/navigation/guidance.py`, tests |
| Rule-based decision engine (cooldown + dedup) | IMPLEMENTED | `src/decision/engine.py`, tests |
| OS-native TTS speech output (pyttsx3/SAPI5) | IMPLEMENTED | `src/audio/tts.py`, real-engine init verified |
| End-to-end Assist App | IMPLEMENTED | `src/assist/assist_app.py` (code + config read) |
| Flask web dashboard (MJPEG + JSON state) | IMPLEMENTED | live boot + `/api/state` + `/video_feed` verified |
| Tests | IMPLEMENTED | 159/159 passed |

**Headline numbers observed during this audit:**

- Test suite: **159 passed, 0 failed** (51.9 s).
- YOLO steady-state inference on 720p synthetic frames: **~58–76 ms avg**
  (≈ 13–17 inference/s), with high variance (single calls up to ~330 ms).
- RapidOCR steady-state on a synthetic text frame: **~4.1–4.7 s per call**
  (the dominant CPU bottleneck on this machine).
- End-to-end loop (stub camera, no hardware): **~1.37 loop fps** — bounded almost
  entirely by OCR latency, not detection.
- Live web dashboard: served `/`, `/api/state` (200), `/video_feed` (200) with real
  camera feed at 1280×720; reported feed FPS ~9.3.

**Primary risks (see §26):** OCR latency on CPU; distance model is heuristic and
uncalibrated; `matplotlib` and `pytest-cov` are declared in `requirements.txt` but not
installed; `VideoRecorder.start()` does not validate the camera is running.

---

## 2. Repository Structure

**Status: IMPLEMENTED**

```
ai_vision_system/
├── .gitignore / LICENSE / README.md / requirements.txt
├── assets/            # sample_images/, sample_videos/, screenshots/ (all git-ignored except .gitkeep)
├── configs/           # assist_config.yaml, camera_config.yaml, logging_config.yaml
├── docs/              # 14 per-module + architecture docs
├── logs/              # app.log (git-ignored)
├── models/            # yolov8n.onnx (12.8 MB, git-ignored)
├── scripts/audit/     # THIS audit's temporary tools (benchmark.py, failure_modes.py)
├── src/               # 40 Python files, ~4,950 LOC
│   ├── assist/  audio/  camera/  decision/  detection/
│   ├── image_fundamentals/  image_processing/  morphology/  navigation/
│   ├── ocr/  playground/  server/  tracking/  utils/
└── tests/             # 14 Python files, ~1,255 LOC, 159 test cases
```

- `src/` = 40 `.py` files ≈ 4,952 lines.
- `tests/` = 14 `.py` files ≈ 1,255 lines.
- 15 git commits on `master`; latest two are the recent speech/TTS fixes.

---

## 3. Environment & Toolchain

**Status: IMPLEMENTED**

| Component | Value |
|---|---|
| OS | Windows (win32) |
| Python | 3.13.14 |
| OpenCV | 5.0.0.93 |
| NumPy | 2.5.1 |
| Pillow | 12.2.0 |
| PyYAML | 6.0.3 |
| onnxruntime | 1.28.0 |
| rapidocr-onnxruntime | 1.2.3 |
| pyttsx3 | 2.99 |
| Flask | 3.1.3 |
| flask-cors | 6.0.5 |
| pytest | 9.1.1 |
| matplotlib | **NOT INSTALLED** (declared in requirements.txt) |
| pytest-cov | **NOT INSTALLED** (declared in requirements.txt) |

**Findings:**
- `requirements.txt` declares `matplotlib>=3.7.0` and `pytest-cov>=4.1.0` but neither
  is installed in this environment. `--cov` therefore fails ("unrecognized arguments").
- No source file imports `matplotlib`, so runtime is unaffected — it appears to be a
  **declared-but-unused** dependency (declared "for later modules" per README).
- `pytest-cov` is used by nobody in `tests/`; no coverage config exists in the repo.

---

## 4. Version Control State

**Status: IMPLEMENTED**

- Branch `master`, working tree **clean** (`git status` = "nothing to commit").
- `origin` = `https://github.com/Sri-Ram-git/Ai_Powered_Assistive_system.git`.
- Up to date with `origin/master` (no unpushed commits).
- 15 commits total. Recent history:
  - `99c22ff` Stop repeating speech; narrate all detected objects with distance
  - `ef13602` Fix TTS speaking only once per session (pyttsx3 SAPI5)
  - `e65ed01` test -1
  - `d1911e5` feat: complete Week 1 modules 3-8
  - `c124478`, `da7d59c`, `38083b8`, `b9a1048`, `148cc0b`, `423a4af`, `6d01444`,
    `99670e2`, `251505b`, `55ffd07`, `1926523`
- `.gitignore` correctly excludes `models/*.onnx`, `assets/*` (except `.gitkeep`),
  `logs/*.log`, and personal media. Only the synthetic sample scene
  `src/image_fundamentals/sample_images/test_scene.png` is versioned.

**Audit note:** the `scripts/audit/` directory (benchmark + failure-mode tools) is a
new untracked artifact created for this audit. It is not committed.

---

## 5. Configuration System

**Status: IMPLEMENTED**

Three YAML configs exist and are wired into code:

| File | Used by |
|---|---|
| `configs/assist_config.yaml` | `assist_app.py`, `PipelineConfig.from_yaml` |
| `configs/camera_config.yaml` | camera tooling (constants; not read by code paths we exercised) |
| `configs/logging_config.yaml` | documents logging defaults; `logger.py` reads none of it (uses hardcoded defaults) |

Verified keys in `assist_config.yaml`:
- `detection`: model_path `models/yolov8n.onnx`, input_size 640, conf 0.45, iou 0.45, every_n_frames 2
- `tracking`: iou 0.3, max_missed 8, distance_change_metres 1.0, min_announce_interval 3.0
- `navigation`: vertical_fov 55 (reference_heights commented out)
- `ocr`: min_confidence 0.3, max_boxes 50, every_n_frames 10, ask_before_reading false
- `speech`: rate 165, volume 1.0
- `decision`: cooldown_seconds 4.0, min_priority 5, speak_ocr_text true, max_ocr_chars 80
- `camera`: id 0, resolution [1280, 720]
- `app`: display_scale 1.0, jpeg_width 960, jpeg_quality 70

**Findings:**
- `assist_app.py` reads `detection.conf_threshold` default `0.35` and applies
  `ocr.ask_before_reading`; `server/pipeline.py` **hardcodes** `conf_threshold=0.35`
  and `iou_threshold=0.45` (does not read them from config). Minor inconsistency.
- `logging_config.yaml` is **not consumed** by `src/utils/logger.py` — the logger uses
  hardcoded `logs/app.log`, `logging.INFO`. Config file is documentation only.
- `camera_config.yaml` is not read by `Camera`/`CameraManager` (they use constructor
  defaults). Also documentation-only.

---

## 6. Camera Module

**Status: IMPLEMENTED**  — files: `src/camera/{camera.py, camera_manager.py, camera_utils.py, hud.py, camera_test.py}`

Verified behaviours:
- `Camera` (CAP_DSHOW backend, mirror default True): opens device 0 at 640×480
  (smoke-tested live), tracks `actual_fps`, `frame_count`, exposes
  `start/stop/read/set_resolution`, context-manager support.
- `CameraNotFoundError` raised for device 999 (verified live).
- `CameraAccessError` if `read()`/`set_resolution()` before `start()` (verified).
- `InvalidResolutionError` for `0×-5` (verified).
- `CameraManager.list_cameras()` probed device 0 successfully; DSHOW emits a warning
  on this machine but opens fine.
- `camera_utils`: `take_screenshot`, `record_video`, `draw_fps`, `draw_timestamp`,
  `show_feed`, `get_screen_size` (Win32), `open_fullscreen_window`, `scale_to_fit`,
  `auto_select_resolution`, `VideoRecorder`.
- `HUD` + `Canvas` + `FontManager`: PIL-based monochrome overlay with draggable
  top/bottom bars, REC pill, toasts, average FPS. Pure presentation (no pipeline
  coupling) — verified by reading.

**Finding (minor):**
- `VideoRecorder.start()` does **not** validate that the camera is running. When given
  an idle camera, it starts, then its worker loop logs repeated
  "Frame dropped during recording" warnings and produces a video file containing no
  frames. Verified in failure-mode exercise. Recommend a `camera.is_running` guard.

---

## 7. Image Fundamentals

**Status: IMPLEMENTED** — `src/image_fundamentals/image_utils.py`

Stateless utilities: `read_image`, `save_image`, `resize`, `crop`, `flip`, `rotate`,
colour conversions (`to_grayscale/to_rgb/to_bgr/to_hsv/to_bgr_from_hsv`),
`image_info`, `image_stats`, `pixel_value`, `histogram`, `histogram_image`.

Verified: 21 tests pass; failure-mode checks confirm `ImageError` for missing file,
empty image, out-of-bounds crop, bad scale/flip/pixel access, `None` input. A
`sample_images/test_scene.png` (9.8 KB) synthetic scene is versioned and used by
demos/tests.

---

## 8. Image Processing

**Status: IMPLEMENTED** — `src/image_processing/processing.py`

Stateless filters: Gaussian/median/bilateral blur, fixed & adaptive threshold, Canny,
Sobel X/Y/magnitude, Laplacian, sharpen (unsharp mask), brightness/contrast, noise
add/remove.

Verified: 17 tests pass; failure-mode checks confirm `ProcessingError` for even
kernels, out-of-range brightness/contrast, invalid noise type/amount.

---

## 9. Morphology & Shape Detection

**Status: IMPLEMENTED** — `src/morphology/contour_utils.py`, `shape_detector.py`

Erode/dilate/open/close, contour metrics (area, perimeter, bbox, hull, defects,
centroid), `ShapeDetector` (circle/rectangle/triangle/polygon-n classification via
vertex count + circularity).

Verified: 14 tests pass; failure-mode checks confirm `ProcessingError` for even
kernels.

---

## 10. Vision Playground App

**Status: IMPLEMENTED** — `src/playground/playground.py`

Fullscreen live app: 7 filters, grayscale/edge/threshold toggles, screenshot save,
5 s threaded recording, draggable HUD. Code read fully; requires webcam + interactive
display, so live-run **NOT VERIFIED** in this session (no GUI interaction executed).

---

## 11. Object Detection (YOLOv8)

**Status: IMPLEMENTED** — `src/detection/detector.py`

- `YoloDetector` loads `models/yolov8n.onnx` (12.8 MB) via `cv2.dnn.readNetFromONNX`.
- Letterbox → 640×640 → `blobFromImage` → forward → decode `[1,84,8400]` → per-class
  confidence threshold → NMS → scale-back to original coords.
- 80 COCO class names; `NAVIGATION_CLASSES` maps to coarse categories
  (person/vehicle/traffic signal/obstacle/object).
- Per-class higher confidence bars for `laptop/tv/book/remote/mouse/cell phone/toilet`
  plus an aspect-ratio heuristic to drop "false laptops".
- `DetectionResult` exposes `label`, `confidence`, `box`, `category`, `center`, `area`.
- `label_detections()` draws boxes/labels.

Verified: model loads and infers; 15 tests pass; `DetectionError` raised for a missing
model file; `detect()` returns `[]` for empty frames. Live web dashboard ran real
detection on the live feed. **Detection accuracy on real scenes NOT VERIFIED** (no
ground truth / labelled dataset available; real camera frame at audit time yielded 0
detections, consistent with an empty scene).

**Perf (CPU):** ~58–76 ms steady-state average on 720p synthetic frames (≈13–17
inference/s). High variance observed (58 ms→330 ms across runs).

---

## 12. Object Tracking

**Status: IMPLEMENTED** — `src/tracking/tracker.py`

- `IoUTracker`: greedy IoU association, stable `track_id`, `age`, `missed`,
  `first_seen`, `max_missed` drop. Pure NumPy, no ML.
- `TrackedObject` exposes `area`, `alive`, `category`, `center`.

Verified: 16 tracking tests pass. Association logic exercised end-to-end via pipeline
e2e test with the real ONNX model.

---

## 13. Tracking Guidance Monitor

**Status: IMPLEMENTED** — `src/tracking/monitor.py`

`TrackingMonitor` converts tracks into spoken phrases **on change**: announces new
objects, re-announces meaningful distance changes (≥ `distance_change_metres`) or
direction changes, respects per-track `min_announce_interval`. Pure-function core
`events()` + stateful memory wrapper. Uses `guidance.distance_estimate` /
`reference_height` / `direction_of`.

Verified: covered by navigation/tracking tests and pipeline e2e; the "person ahead"
repetition bug fix is exercised by the `cue_identity` regression tests.

---

## 14. OCR

**Status: IMPLEMENTED** — `src/ocr/ocr_engine.py`

- `OcrEngine` wraps RapidOCR (CPU ONNX). Typed `OcrResult` (text, confidence,
  axis-aligned box), `read_text()`, `text_of()`, `draw_text_boxes()`.
- Rejects non-ndarray input with `OcrError`; returns `[]` for `None`/empty (verified).

Verified: 9 tests pass; real OCR run on synthetic "EXIT 12" text produced 1 line.
**Perf (CPU):** first call ~4.5 s, steady-state **~4.1–4.7 s per frame** — the single
biggest CPU cost in the pipeline on this machine.

---

## 15. Navigation & Guidance

**Status: IMPLEMENTED** — `src/navigation/guidance.py`

- `direction_of` (left/ahead/right by centre thirds), `distance_estimate` (pinhole
  model using vertical FOV + per-label reference heights), `reference_height`,
  `distance_phrase`, `nearest_obstacle` (largest box), `scene_cues`.
- `scene_cues` emits: traffic light, stop sign, crosswalk/do-not-walk keywords,
  per-person, per-vehicle, and **all other detections** (cell phone, bottle, laptop,
  chair, …) with direction + distance — the fix for the "phone never spoken" review
  item.

Verified: 17 navigation tests pass; distance/direction helpers benchmarked at
< 0.2 ms.

**Limitation:** distance is a heuristic based on an assumed 55° vertical FOV and
per-class reference heights; **accuracy is NOT VERIFIED** against ground truth. This
is the documented biggest accuracy lever (README §Distance accuracy).

---

## 16. Decision Engine

**Status: IMPLEMENTED** — `src/decision/engine.py`

- Pure `evaluate(FrameSummary)` → prioritised `Decision`s (traffic light 0, stop sign
  1, crosswalk/do-not-walk 2, text/person 3, vehicles 4, obstacle/generic 5).
- `DecisionEngine` adds cooldown (`cooldown_seconds`), `min_priority` gating, OCR-text
  reading toggle, and **identity-based dedup** via `cue_identity()` which strips
  distance jitter ("about 5 metres" vs "about 6 metres" are the same message).
- `decide()` accepts `already_spoken` (phrases emitted by the tracking monitor) so one
  object is never narrated twice in a frame.

Verified: 22 decision tests pass, including the new regression tests for
`cue_identity`, already-spoken skip, cooldown, and identity change.

---

## 17. Speech Output (TTS)

**Status: IMPLEMENTED** — `src/audio/tts.py`

- `SpeechOutput` — non-blocking pyttsx3 (SAPI5 on Windows) worker.
- Engine is created **inside the worker thread** and driven with
  `startLoop(False)` + an `iterate()` pump until `isBusy()` clears — this is the fix
  for the "speaks only once per session" SAPI5 bug (documented in the module docstring
  and commit `ef13602`).
- `speak()` enqueues; `say_now()` waits via a `done` event; `shutdown()` joins worker.

Verified: 7 audio tests pass; `SpeechOutput()` initialised successfully on the real
engine during the audit. Earlier (pre-audit) session confirmed 3 phrases via `say_now`
in 9.4 s with real audible output.

---

## 18. Assist App (End-to-End)

**Status: IMPLEMENTED** — `src/assist/assist_app.py`

Wires the full pipeline: Camera → (throttled) Detect → Track → OCR → Tracking
Monitor + Decision Engine → TTS, with tracked boxes/IDs/distances and OCR text drawn,
and a draggable HUD. Keyboard: m (mute), t (OCR mode), r (read text), s (screenshot),
space (reset), q (quit).

Verified by code read + all component tests + real-camera pipeline exercise. **Full
interactive GUI run NOT VERIFIED** (requires a human at the webcam/display).

---

## 19. Web Dashboard & Server

**Status: IMPLEMENTED** — `src/server/app.py`, `src/server/pipeline.py`

- `PipelineServer`: camera → grab thread (annotate + JPEG for MJPEG, never runs AI) →
  inference thread (detect/track/OCR/decision/monitor, publishes JSON state). This
  split keeps the live feed responsive while inference is slow.
- Endpoints: `GET /` (dashboard HTML), `GET /video_feed` (MJPEG),
  `GET /api/state` (JSON detections, distances, guidance, FPS, latency, OCR text).
- `PipelineConfig.from_yaml` reads `assist_config.yaml`.

**Live verified during audit:** server booted on port 5123 with the real camera
(1280×720), `/` and `/api/state` returned 200, `/api/state` JSON was well-formed
(`{"detections":[],"error":null,"fps":9.29,"guidance":null,...}`), `/video_feed`
returned 200 with the MJPEG content-type. 5 server tests pass.

---

## 20. Utilities & Logging

**Status: IMPLEMENTED**

- `src/utils/exceptions.py`: typed exception hierarchy (Camera*, Image, Processing,
  Detection, Ocr, Speech, Decision errors).
- `src/utils/logger.py`: `setup_logger(name)` — console + rotating single file
  handler at `logs/app.log`. Verified: `logs/app.log` grew during the audit.
- Note: `logging_config.yaml` is not consumed by the logger (see §5).

---

## 21. Testing & Test Coverage

**Status: IMPLEMENTED** (coverage metrics NOT VERIFIED)

- **159 tests passed, 0 failed** (`python -m pytest tests -q`, 51.9 s).
- Distribution:
  - test_audio 7, test_camera 14, test_decision 22, test_detection 15,
    test_image_utils 21, test_morphology 14, test_navigation 17, test_ocr 9,
    test_pipeline_e2e 2, test_processing 17, test_server 5, test_tracking 16.
- Hardware-free by design: real camera/model are stubbed except the e2e test which
  uses the real ONNX model with a synthetic camera.
- **No coverage numbers available**: `pytest-cov` is not installed and no coverage
  config exists. Statement coverage is therefore **NOT VERIFIED**.
- No `conftest.py` fixtures beyond helpers; tests use plain pytest.

---

## 22. Performance Characteristics

**Status: IMPLEMENTED** (measured; see `PERFORMANCE_RESULTS.md` for raw data)

Measured on this machine (CPU-only), synthetic inputs:

| Component | Latency | Notes |
|---|---|---|
| YOLO detect, 1280×720 → 640×640 | ~58–76 ms avg | high variance (up to ~330 ms) |
| YOLO detect, 640×480 → 640×640 | ~62–190 ms | same variance |
| RapidOCR steady-state | ~4.1–4.7 s | dominant bottleneck |
| `direction_of` / `distance_estimate` | < 0.2 ms | negligible |
| `DecisionEngine.decide` | ~0.01 ms | negligible |
| `IoUTracker.update` / `TrackingMonitor.events` | < 0.1 ms | negligible |
| End-to-end loop (stub camera, no HW) | ~1.37 loop fps | OCR-bound |
| Web feed (real camera) | ~9.3 fps feed | grab thread decoupled from inference |

Interpretation: on this hardware the CPU budget is dominated by OCR. The server's
grab/infer thread split keeps the live video feed usable (~9 fps) while OCR blocks
only the inference thread. On the single-threaded `assist_app.py` loop, OCR stalls the
whole frame loop (~1.4 fps).

---

## 23. Failure Modes & Error Handling

**Status: IMPLEMENTED** (one gap found)

Executed `scripts/audit/failure_modes.py` — **34 passed, 1 failed**:

- Verified exceptions: missing/corrupt image, empty save, out-of-bounds/negative
  crop, bad scale/flip/pixel, `None` image info, even kernel sizes, invalid
  brightness/contrast/noise args, camera read/set before start, invalid resolution,
  missing model, non-image OCR input, OCR empty/None returns `[]`.
- `CameraNotFoundError` for device 999; real TTS engine init succeeded.
- **Gap:** `VideoRecorder.start()` with an idle camera does not raise
  `RecordingError`; it logs repeated warnings and writes an empty video. See §6.

---

## 24. Model & Asset Verification

**Status: IMPLEMENTED**

- `models/yolov8n.onnx`: present, 12,805,802 bytes (~12.8 MB), git-ignored.
- Loads and infers via `cv2.dnn.readNetFromONNX` (verified in tests, benchmarks, and
  the live dashboard).
- Synthetic sample scene `src/image_fundamentals/sample_images/test_scene.png`
  (9,854 bytes) is the only versioned media asset.
- `assets/` contains only `.gitkeep` placeholders; no personal media committed.

---

## 25. Documentation

**Status: IMPLEMENTED**

- `README.md`: quickstart, module table, config guide, keys, layout, security,
  coding standards — accurate against the code we read.
- `docs/` has 14 files: architecture, camera, image_fundamentals, image_processing,
  morphology, playground, detection, ocr, decision, audio, navigation, tracking,
  assist, server.

**Note:** `docs/architecture.md` describes the design; we did not verify every doc
line against code line-by-line, so full doc-code consistency is **NOT VERIFIED**.

---

## 26. Known Issues, Limitations & Recommendations

**Status: PARTIALLY IMPLEMENTED** (this is the issues register)

1. **OCR latency is the system bottleneck.** ~4–5 s/call on CPU. Every `ocr_every`
   frames the inference thread blocks for seconds. In `assist_app.py` this stalls the
   whole loop. Recommendations: raise `ocr.every_n_frames`, run OCR in its own thread
   in the app (the server already does), reduce input resolution for OCR, or accept
   the cadence.
2. **Distance accuracy is heuristic & uncalibrated.** Pinhole model with assumed
   FOV/reference heights; no calibration or ground-truth validation exists. Highest
   priority if distance-based speech must be trustworthy.
3. **Declared-but-missing deps.** `matplotlib` and `pytest-cov` in requirements.txt
   are not installed and not used. Either install (for coverage reporting) or remove.
4. **`VideoRecorder.start()` doesn't guard on camera running.** Produces an empty
   file with warnings instead of failing fast. Add an `is_running` check.
5. **Config inconsistencies.** `server/pipeline.py` hardcodes conf/iou thresholds
   instead of reading `assist_config.yaml`; `camera_config.yaml` and
   `logging_config.yaml` are not consumed by code.
6. **No code coverage metrics** (pytest-cov absent).
7. **Real-world accuracy (detection/OCR/distance) NOT VERIFIED** — no labelled
   dataset or ground-truth benchmark in the repo.
8. **No persistence/DB layer.** Everything is in-memory; restarting loses state.
   Fine for the current MVP scope.

---

## 27. Overall Verdict & Next Steps

**Status: IMPLEMENTED (MVP complete, deployment-ready path exists)**

The system delivers everything claimed for Weeks 1–3 and the recent review fixes
(speech dedup, all-object narration, TTS one-shot bug). It is testable, documented,
and privacy-conscious (no personal media in git).

**Recommended next steps (in priority order):**
1. Install `pytest-cov` (or add coverage config) and publish real coverage numbers.
2. Decide OCR strategy for real-world use (throttling, resizing, or async thread).
3. Calibrate `vertical_fov` / `reference_heights` against a known-distance test and
   document measured accuracy.
4. Add a guard in `VideoRecorder.start()` for a non-running camera.
5. Reconcile `server/pipeline.py` config reads with `assist_config.yaml`.
6. For the Flutter/React Native phone app (per earlier discussion): Flutter is
   recommended — it can talk HTTP to the existing `src/server` dashboard endpoints,
   reusing `/api/state` and `/video_feed` unchanged.

---

*Evidence artifacts produced by this audit (in `scripts/audit/`): `benchmark.py`,
`failure_modes.py`. Companion reports: `TEST_RESULTS.md`, `PERFORMANCE_RESULTS.md`,
`architecture-current.md`.*