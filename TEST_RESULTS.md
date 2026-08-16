# TEST_RESULTS.md

**Date:** 16 Aug 2026
**Command:** `python -m pytest tests -q`
**Result:** **159 passed, 0 failed** — 51.92 s
**Environment:** Windows, Python 3.13.14, OpenCV 5.0.0.93, numpy 2.5.1,
onnxruntime 1.28.0, rapidocr-onnxruntime 1.2.3, pytest 9.1.1

## Summary

| Metric | Value |
|---|---|
| Total tests | 159 |
| Passed | 159 |
| Failed | 0 |
| Skipped | 0 |
| Duration | 51.92 s |
| Coverage | NOT VERIFIED (`pytest-cov` not installed) |

## Per-file breakdown

| Test file | Tests | Area |
|---|---|---|
| tests/test_audio.py | 7 | TTS SpeechOutput (fake engine) |
| tests/test_camera.py | 14 | Camera lifecycle, manager, utils, HUD |
| tests/test_decision.py | 22 | DecisionEngine, cue_identity, cooldown, priorities |
| tests/test_detection.py | 15 | YoloDetector parsing, NMS, letterbox |
| tests/test_image_utils.py | 21 | Image fundamentals I/O, transforms, stats |
| tests/test_morphology.py | 14 | Erode/dilate/open/close, contours, shapes |
| tests/test_navigation.py | 17 | direction, distance, scene_cues, obstacles |
| tests/test_ocr.py | 9 | RapidOCR wrapper, OcrResult |
| tests/test_pipeline_e2e.py | 2 | Real ONNX model + stub camera end-to-end |
| tests/test_processing.py | 17 | Filters, thresholds, edges, noise |
| tests/test_server.py | 5 | Flask endpoints, PipelineConfig, helpers |
| tests/test_tracking.py | 16 | IoU association, monitor events |

## Notable observations

- Tests are hardware-free: real camera/model are stubbed; the only test using the
  real ONNX model feeds a synthetic camera.
- OpenCV emitted a DSHOW backend warning ("backend generally available but can't be
  used to capture by index") during enumeration — the camera still opens on device 0.
- `cv2.imread` emits a warning for the intentionally-missing file in the image-utils
  error-path test; the test passes (ImageError raised as designed).

## Coverage status

Coverage metrics are **NOT VERIFIED**: `pytest-cov` is declared in
`requirements.txt` but not installed, so `pytest --cov=...` fails with
"unrecognized arguments: --cov". No coverage config file exists in the repo.