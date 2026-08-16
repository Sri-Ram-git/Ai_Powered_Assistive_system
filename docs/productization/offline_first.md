# Offline-First Design

## Principle

The assistive device **must work fully offline**.  A user navigating a
room cannot wait for (or depend on) a network.  Offline capability is a
hard requirement, not a fallback.

## What already runs offline

| Stage                 | Runtime                  | Network needed? |
|-----------------------|--------------------------|-----------------|
| Camera capture        | OpenCV                   | no              |
| Object detection      | YOLOv8n, OpenCV DNN      | no              |
| Tracking / guidance   | IoU tracker + monitor    | no              |
| Distance estimation   | pinhole model + calibration | no           |
| OCR                   | RapidOCR (onnxruntime)   | no              |
| Speech output         | pyttsx3 (OS engine)      | no              |
| Speech input          | keyword STT (default)    | no              |
| Safety engine         | deterministic code       | no              |
| Response planner      | priority/dedup/cooldown  | no              |
| Scene description     | DeterministicVLM         | no              |
| Depth (synthetic)     | model-free               | no              |
| Web dashboard/API     | Flask, localhost         | no              |

The **entire safety-critical path** — camera → perception → safety →
response — is deterministic and offline.  No LLM is ever on that path.

## What is optional / cloud-only

* **RemoteVLM** (scene descriptions via a cloud model) — opt-in, with a
  guaranteed offline fallback (`DeterministicVLM`) on any error/timeout.
* **Faster-Whisper STT** — a local model, downloaded once, then offline.
* **Prometheus scraping** of `/api/metrics` — offline-capable.

These are *enhancements*; the device is fully functional without them.

## Architecture consequences

1. **No hard external deps at runtime.**  All models are local files
   (`models/manifest.yaml` tracks provenance).  The container image
   (P19) bakes the CPU runtime; nothing phones home.
2. **Graceful degradation.**  Every optional backend falls back to a
   deterministic local equivalent (VLM, STT, depth) — the pipeline
   never crashes because a cloud call failed.
3. **Network = diagnostics only.**  `/api/metrics` and the dashboard are
   for the operator; they are not required for the device to assist.

## Keeping it offline-first (checklist for new code)

- [ ] New stage has a model-free / deterministic fallback path.
- [ ] No synchronous network call on any loop (grab/detect/planner).
- [ ] Cloud features are explicitly opt-in via `.env`/config, never
      implied defaults.
- [ ] If a new dependency needs a download, it is optional and its
      absence degrades gracefully (see depth, whisper, VLM).
- [ ] The safety path remains pure code and offline.