# Mobile Path

## Goal

Eventually run the assistive vision experience on a **phone** (Android /
iOS).  This document is the *design path* — what already translates and
what must change.  No mobile SDK is committed yet.

## What already transfers

* **The JSON API is transport-agnostic.**  `/api/health`, `/api/state`,
  `/api/mode`, `/api/command`, `/api/metrics` are plain HTTP+JSON.  A
  phone client can drive the same pipeline.
* **The core engine is hardware-independent.**  `AsyncVisionPipeline`
  consumes a `Camera` object; a phone camera adapter would implement the
  same interface.
* **All AI is local and lightweight** (YOLOv8n ~13 MB, RapidOCR CPU).
  Phone-class NPUs can run both via ORT Mobile / Core ML / NNAPI.
* **Deterministic safety + response planner** have zero cloud coupling.

## What changes on mobile

| Area               | Desktop (now)                    | Mobile path                           |
|--------------------|----------------------------------|---------------------------------------|
| Camera            | OpenCV `VideoCapture`            | OS camera API (AVFoundation/Camera2)  |
| UI                | Flask dashboard                  | Native app calling `/api/*`           |
| Inference runtime | OpenCV DNN + onnxruntime CPU     | ONNX Runtime Mobile / CoreML / NNAPI  |
| Model size        | ~13 MB YOLO + ~60 MB OCR         | Same models, quantized INT8 preferred |
| Speech            | pyttsx3 + keyword STT            | OS TTS + on-device STT                |
| Hosting           | device hosts its own Flask app   | app can run headless behind an SDK    |

## Recommended architecture

    phone UI (native) ── HTTP/JSON ──> device host (Flask API + core)
        or, fully on-device later:
    phone UI ──> embedded core (PyTorch Mobile / ORT Mobile) ──> sensors

**Phase A (near-term): remote control.**  Keep the compute on the host
device (or a Pi); ship a thin mobile app that renders `/video_feed` and
reads `/api/state`, with mode + command buttons.  This needs zero new
AI work and validates the API contract.

**Phase B (mid-term): on-device inference.**  Port `YoloDetector` +
`OcrEngine` to ORT Mobile.  The engine's `Camera`/pipeline abstraction
means the perception code moves with minimal change.

**Phase C (long-term): full standalone.**  Local STT/TTS, INT8 models,
privacy-complete local experience.

## Non-goals for now

* No separate mobile codebase exists yet — do not add one speculatively.
* The dashboard remains the reference UI.
* Mobile must not regress the offline-first guarantee (P22).