# Object-Aware OCR Architecture

## 1. Goal

Coordinate object detection and OCR into ONE perception system:

```
If YOLO detects an object that is likely to contain text,
immediately inspect that object's region for text.
```

The camera must stay live at all times. OCR is a *secondary, triggered,
asynchronous* stage bound to object tracks — never a full-screen scanner
and never a blocking step.

## 2. Why this design

The previous system OCR'd the whole frame every N frames. That had three
problems:

1. **Waste** — a plain background, a wall, or a person consumed the same
   OCR budget as a bottle label.
2. **Blind to objects** — the result was not associated with *what* was
   detected, so the user could not ask "what does that bottle say?".
3. **Repetition** — the same text was re-read and re-spoken every frame.

Object-aware OCR fixes all three by gating OCR behind a policy
(object → eligible?), a text-presence check (object → text likely?),
and a trigger policy (already read it? → skip).

## 3. Existing system (as inspected)

| Stage | Module | Notes |
|---|---|---|
| YOLO | `src/detection/detector.py` | `YoloDetector.detect(frame)` → `List[DetectionResult]`; COCO-80 classes; missing model must not kill the loop. |
| Tracking | `src/tracking/tracker.py` | `IoUTracker.update(detections)` → `List[TrackedObject]` with stable `track_id`, smoothed box, label voting. |
| Decision | `src/decision/engine.py` | `DecisionEngine.decide(summary)`; crosswalk cues use OCR text; `speak_ocr_text` narrates OCR text. |
| OCR | `src/ocr/ocr_engine.py`, `worker.py`, `preprocess.py` | `OcrEngine.read_text(image)` → `List[OcrResult]`; `OcrWorker` runs full-frame OCR on a thread with latest-result semantics; `preprocess()` has 6 strategies. |
| TTS | `src/audio/tts.py`, `speech_queue.py` | `SpeechOutput.speak(text)` non-blocking; `SpeechQueue` prioritises/dedupes/rate-limits. |
| UI | `src/assist/assist_app.py`, `src/camera/hud.py` | Fullscreen cv2 window; HUD bar; `r` key reads current OCR text aloud. |
| Threads | `src/core/pipeline.py` | grab + detect + OCR-worker threads; `FrameManager` latest-frame hub; `LatestResults` per-stage. |
| Config | `src/core/config.py`, `configs/assist_config.yaml` | `PipelineConfig.from_yaml`; `ocr.*` section. |

### Where each event happens today

```
YOLO results   src/core/pipeline.py:_detect_loop  (detector.detect every N frames)
Tracking       src/core/pipeline.py:_detect_loop  (tracker.update)
OCR trigger    src/core/pipeline.py:_detect_loop  (full-frame submit every ocr_every)
OCR runs       src/ocr/worker.py:OcrWorker._run   (worker thread)
TTS trigger    src/core/pipeline.py:_detect_loop  (decision engine -> speech_callback)
UI gets OCR    src/assist/assist_app.py           (state["ocr_text"], 'r' key)
```

## 4. New architecture

```
                     CAMERA
                       │
                       ▼
                 FRAME MANAGER
                       │
                       ▼
                     YOLO
                       │
                       ▼
                   TRACKER
                       │
         ┌─────────────┴─────────────┐
         │                           │
   normal object              text-bearing object   ← ocr_policy.yaml
         │                           │
         │                           ▼
         │                 OCR TRIGGER POLICY       ← new track / moved /
         │                           │                stale / user / cooldown
         │                           ▼
         │                      OBJECT ROI           ← padded, clamped, validated
         │                           │
         │                      TEXT PRESENCE        ← cheap heuristic gate
         │                           │ (text likely?)
         │                           ▼
         │                   OBJECT OCR WORKER        ← 1 worker thread,
         │                           │                 newest-request-wins slot
         │                  preprocessing variants ─┐
         │                           │               │ best result by
         │                           ▼               │ confidence+length
         │                   TRACK OCR STORE          ← per track_id:
         │                           │                 text, confidence,
         │                           │                 timestamp, votes
         │                           ▼
         └─────────────┬─────────────┘
                       ▼
                 SCENE / STATE
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
          CAMERA UI          TTS QUEUE
              │                 │
              ▼                 ▼
        SIDE TEXT PANEL       SPEAKER
              │
              ▼
         READ ALOUD            (uses saved text — no re-OCR)
```

### 4.1 New modules

| Module | Responsibility |
|---|---|
| `src/ocr/policy.py` | Load `configs/ocr_policy.yaml`; object → priority tier map, validated against the detector's real COCO classes. |
| `src/ocr/roi.py` | Padded, clamped, validated object ROI; smart upscaling for small text. |
| `src/ocr/text_presence.py` | Cheap heuristic (edge density + connected components) deciding whether an ROI is worth OCR. |
| `src/ocr/object_ocr.py` | `ObjectOcrResult`/`TrackOcrEntry`, text validation, preprocessing-variant selection, `TrackOcrStore` (voting + expiry + dedup), trigger policy. |
| `src/ocr/object_worker.py` | `ObjectOcrWorker`: one worker thread, newest-request-wins, timeout bookkeeping, result callbacks. |

### 4.2 Reused as-is (no duplication)

- `src/ocr/ocr_engine.py` — the RapidOCR wrapper.
- `src/ocr/preprocess.py` — extended with two strategies (adaptive threshold,
  sharpen) but the existing ones are reused.
- `src/tracking/tracker.py`, `src/detection/detector.py` — untouched.
- `src/audio/tts.py`, `src/audio/speech_queue.py` — READ ALOUD and auto-read
  enqueue through the existing queue.
- `src/core/pipeline.py` — the detect loop hosts the new trigger.

## 5. Object → OCR eligibility policy

`configs/ocr_policy.yaml`:

```yaml
ocr_policy:
  high_priority:  [book, bottle, laptop, "cell phone", tv, "stop sign"]
  medium_priority: [cup, backpack, handbag, suitcase, keyboard, remote]
  low_priority:   []
  disabled:       [person, chair, "dining table", potted plant]
```

Rules:

- Only labels that exist in the detector's `COCO_NAMES` are accepted;
  anything else is ignored with a warning (never invent classes).
- `disabled` objects are never OCR'd (person, chair, …).
- A label not listed falls back to `medium` (configurable `default_tier`).

## 6. ROI extraction

- Pad the track box by `ocr_roi_padding` (default 10% of box width/height)
  so text near the object edge is not clipped.
- Clamp to image bounds.
- Reject when `x1 >= x2`, `y1 >= y2`, or width/height below
  `ocr_roi_min_w`/`ocr_roi_min_h` — no OCR on degenerate ROIs.

### 6.1 Smart upscaling (small text)

Small objects hold tiny text; OCR degrades below ~10-12 px character height.

`smart_upscale(roi)`:

- `min side < 32 px` → ×3
- `min side < 64 px` → ×2
- otherwise ×1

Upscaling is capped at `ocr_max_upscale` (default 3). The policy is
benchmarked at 1×/2×/3×/4× on the evaluation dataset (see the integration
report) before tuning.

## 7. Text-presence gate

OCR runs only when the ROI plausibly contains text. The gate is cheap
(ms-level) and runs before the slow OCR call:

```
gray → CLAHE → Canny edges
edge_density            = mean(edges)
connected components    = small, elongated blobs (2..50% of ROI dims)
score = 0.5·min(1, density/0.06) + 0.5·min(1, components/25)
has_text = score >= threshold (0.35)
```

A blank wall or a plain object yields ~0 components and is skipped. The
threshold is configurable and the gate can be disabled.

## 8. Preprocessing variants

Each accepted ROI is OCR'd over up to `ocr_variants` candidate
preprocessings, and the best result is kept (highest confidence, then
longest text):

```
ROI
├── original            (BGR)
├── contrast            (CLAHE)
├── threshold           (adaptive threshold)
└── sharpen             (unsharp mask)
```

Short-circuit: if the first variant already yields high-confidence text,
the remaining variants are skipped. Cost is bounded: at most N OCR calls
per request, and requests are rate-limited by the trigger policy.

## 9. Track-aware OCR

Results are keyed by `track_id`:

```python
TrackOcrEntry(
    track_id, label, text, confidence, raw_text,
    roi, variant, timestamp, latency_ms, votes=deque, stable=False,
)
```

- A track is OCR'd once; re-OCR happens only on the trigger conditions
  (below), so the same bottle is not re-read every frame.
- **Temporal voting**: a noisy one-frame read ("COC4 C0LA") does not
  replace the stable result ("COCA COLA") — only 2 consecutive identical
  reads adopt a new stable text.
- **Expiry**: entries for tracks that have disappeared are dropped
  (`ocr_history_max` retained for the UI history).
- Results feed `scene/state` → the decision engine (crosswalk cues) and the
  side panel.

## 10. Trigger policy

OCR is submitted only when:

1. a new text-bearing track appears;
2. an existing text-bearing track moved significantly (`ocr_move_px`);
3. the last result is stale (`ocr_stale_after_s`);
4. the user explicitly asks (READ TEXT / READ ALOUD path);
5. the track is inside a useful size range (largest/nearest first).

Per-track cooldown `ocr_cooldown_s` prevents thrash. Multiple eligible
objects are ordered by: priority tier → area → confidence; only the best
target is submitted per tick.

## 11. Async worker (never blocks vision)

`ObjectOcrWorker` keeps a **single pending-request slot**. If OCR is busy
and a newer request arrives, the newer request replaces the pending one —
a backlog can never form.

- Runs on its own thread; the camera/grab/detect/UI threads never wait.
- If a single OCR call exceeds `ocr_timeout_ms`, the request is marked
  `timeout` and logged (`OCR_TIMEOUT track_id label elapsed`).
- OCR failure yields `status="error"`, never an exception in the caller.
- TTS is already asynchronous (`SpeechOutput` + `SpeechQueue`); READ ALOUD
  enqueues saved text, so TTS never blocks anything either.

## 12. Side text panel

The desktop app keeps the camera at ~70-75% of the window and adds a
right-hand panel (PIL-rendered, antialiased):

```
┌──────────────────────────────┬───────────────────┐
│         CAMERA               │ DETECTED TEXT     │
│                              │ [Bottle #7]       │
│                              │ COCA COLA         │
│                              │ Conf 92% 10:42:10 │
│                              │───────────────────│
│                              │ [READ ALOUD]      │
│                              │ [COPY] [CLEAR]    │
│                              │───────────────────│
│                              │ HISTORY           │
│                              │ • EXIT (sign)     │
│                              │ • INTRODUCTION…   │
└──────────────────────────────┴───────────────────┘
```

- Auto-updates from the track OCR store.
- READ ALOUD speaks the **already saved** text — no re-OCR.
- COPY copies the selected text; CLEAR clears the panel/history.
- Debug mode adds OCR latency, variant, and source ROI.
- Mouse clicks drive the buttons; keyboard shortcuts remain.

## 13. READ ALOUD + auto-read

- **READ ALOUD** (button / `r` key): enqueue the stored text to the TTS
  queue immediately.
- **Auto Read OCR** (`ocr.auto_read`), OFF by default: when a *new stable*
  track result arrives, enqueue `"Text says, …"` through the existing
  speech pipeline (dedup via `SpeechQueue`, no repeats).

## 14. Performance contract

| Metric | Requirement |
|---|---|
| Camera rendering | never blocked by OCR/TTS |
| OCR latency | async; result appears when ready |
| OCR queue | newest-wins (≤1 pending) |
| Repeats | stable text OCR'd once; spoken once |
| Non-text objects | zero OCR cost |

## 15. Tests

- Eligibility policy, ROI padding/validation/upscale, text presence.
- Trigger/cooldown, dedup, track-aware store, voting, validation, timeout,
  queue replacement, worker, read-aloud action, multiple-object priority,
  OCR/TTS failure, camera responsiveness while OCR runs.
- Integration: object → trigger → ROI → OCR → store → panel → READ ALOUD → TTS.