# OCR Integration Report — Object-Aware Text Recognition

Date: 2026-08-17
Scope: object-aware OCR on top of the existing YOLO + IoU-tracker pipeline,
with a side text panel in the desktop app (READ / NOW / COPY / CLEAR) and
read-aloud via the existing SpeechQueue/TTS.

## 1. Objective (30-point spec, condensed)

OCR text **on text-bearing detected objects** (books, bottle labels, laptop
screens, signs, cups...) instead of scanning the whole scene; results are
track-aware (stable, de-duplicated, per-object) and shown in a side panel;
never block camera / YOLO / tracking / UI / TTS; never re-read the same text.

## 2. What changed

| Area | File(s) | Role |
|---|---|---|
| Policy | `configs/ocr_policy.yaml`, `src/ocr/policy.py` | Which labels are worth OCR (COCO-only tiers high/medium/low/disabled); edit without code |
| ROI | `src/ocr/roi.py` | Padded, clamped, validated object ROI + smart upscale (min side <32 -> x3, <64 -> x2, else x1) |
| Gate | `src/ocr/text_presence.py` | Cheap (sub-ms) gradient + connected-component test; veto when no character blobs |
| Core | `src/ocr/object_ocr.py` | `ObjectOcrResult`, text validation, variant selection, `TrackOcrStore` (2-vote temporal consensus, expiry, history), `OcrTrigger` (new/moved/stale + per-track cooldown), `rank_targets` |
| Worker | `src/ocr/object_worker.py` | Async, single pending slot (newest replaces old), presence gate first, variant search, timeout marking |
| Engine | `src/ocr/ocr_engine.py` | RapidOCR built with `det_limit_type='max'` (see §5) |
| Preprocess | `src/ocr/preprocess.py` | + `adaptive`, `sharpen` strategies |
| Pipeline | `src/core/pipeline.py` | Schedule 1 eligible object/tick, store results, `ocr_text` state, READ ALOUD (`read_latest_text`), manual read (`request_manual_ocr`), auto-read dedupe, metrics, reset/clear |
| Config | `src/core/config.py`, `configs/assist_config.yaml` | `object_ocr_enabled`, `ocr_policy_path`, ROI/presence/variant/trigger/auto-read/timeout knobs |
| UI | `src/assist/text_panel.py`, `src/assist/assist_app.py` | Right-side TEXT panel (READ / NOW / COPY / CLEAR buttons, history, status, debug counters) + keys `r c n x` + mouse hover |

## 3. Behaviour contract (verified)

* **Non-blocking**: OCR runs on its own daemon thread; the detect loop only
  submits ROIs (never waits).  Existing `test_realtime_pipeline` proves the
  grab/detect loop keeps publishing frames.
* **Newest-request-wins**: worker holds a single pending slot — a new
  submission replaces an unstarted one (`replaced` counter; unit test
  `test_newest_request_replaces_pending`).
* **One object per tick**: `_schedule_object_ocr` picks the highest-priority
  eligible track that is currently due (new/moved/stale) and submits at most
  one ROI per detect tick, bounding OCR CPU cost.
* **Temporal stability**: `TrackOcrStore` adopts a new text only after 2
  consecutive identical reads; garbage reads never adopt.
* **READ ALOUD speaks saved text** (no re-OCR; `read_latest_text` returns
  False when nothing read).  **Auto-read** off by default, deduped 8 s.
* **Manual read** (`request_manual_ocr`, NOW button / `n`): best eligible
  track first, else the whole frame.
* **Timeout**: a call exceeding `ocr_timeout_ms` (2000) is marked `timeout`,
  logged `OCR_TIMEOUT`, counted in metrics.
* **Cooldown/stale/move** per track prevent repeat OCR of the same text.

## 4. Tests

```
393 passed  (340 pre-existing + 53 new)
ruff check src tests scripts  -> All checks passed
```

New test files:
* `tests/test_object_ocr.py` — policy, ROI/upscale, presence, validation,
  variant selection, store voting/expiry, trigger, ranking, worker semantics.
* `tests/test_object_ocr_pipeline.py` — scheduling, store/state update,
  read-aloud (no re-OCR), auto-read, manual read, reset, disabled path.

## 5. Benchmark — previous version vs this version

Key finding: RapidOCR's default detection resize policy
(`det_limit_type='min'`, limit 736) **upscales any image whose smaller side
is < 736 px** so the min side becomes 736.  A 96x24 bottle label became a
~3270x736 image — a single OCR call took ~10 s, *slower* than a full frame.
This version configures `det_limit_type='max'` (only shrink images whose
largest side exceeds the limit).

Measured on this machine (`scripts/benchmark/ocr_compare.py --runs 3`,
median of 3):

| Input | previous (min limit) | this version (max limit) | speed-up |
|---|---|---|---|
| full 1280x720 frame | 800 - 1160 ms | 356 - 411 ms | **2.2-2.8x** |
| mid 320x72 ROI (label) | ~1250 - 1300 ms | 72 - 152 ms | **8.4-17x** |
| small 96x24 ROI | 19-58 ms* | 53 - 129 ms | consistent (recognises text) |

\* the old min-limit path was unstable on tiny images: sometimes near-empty
detections (fast), sometimes the pathological ~10 s upscale.

End-to-end object-aware path (ROI + presence + up to 3 variants):

| Sample | median latency | status |
|---|---|---|
| label ROI 320x72 | 297 ms | ok/none |
| small ROI 96x24 | 145 ms | ok/contrast |
| blank ROI | 1 ms | no_text (gate) |

## 6. Evaluation dataset (`assets/ocr_eval/`, `scripts/benchmark/object_ocr_eval.py`)

12 synthetic object-like regions (bottle label, book cover, laptop screen,
EXIT / STOP / DO NOT ENTER signs, cup, remote, low-contrast label/screen,
tiny laptop/sign), rendered deterministically, wrapped in a 640x480 scene,
with ground truth.  Images + `ground_truth.json` + `results.json` saved.

Results through the full object-aware path (mean over samples):

| Metric | previous engine (min limit) | this version |
|---|---|---|
| mean char accuracy | 0.837 | **0.926** |
| exact hits (acc >= 0.9) | 8 / 12 | **9 / 12** |
| partial | 4 | 3 |
| missed | 0 | 0 |
| presence-gate false negatives | 2 | **0** |
| mean latency per sample | 31 376 ms | **306 ms** |

Accuracy by contrast: high 0.924, low 0.933.  By kind: sign 1.000,
label 0.931, screen 0.842.  Winning preprocessing variant: `none` 7,
`contrast` 4, `adaptive` 1 — the multi-variant approach pays off
(4/12 samples were rescued by `contrast`; earlier runs also saw `adaptive`
win on low-light material).

## 7. Known limitations

* Synthetic dataset only — real camera footage would validate end-to-end
  recognition more strongly.  Generated images are cheap to regenerate and
  inspect.
* RapidOCR is not interruptible mid-call: the timeout marks a call that
  overran; it does not abort it.
* The presence gate is a heuristic; pathological textured regions (e.g. a
  busy fabric pattern) can still slip through (bounded by the worker slot +
  cooldown).
* Accuracy on multi-line / low-resolution real text will be lower than the
  synthetic 0.926; treat numbers as comparative, not absolute.

## 8. Repro

```
python scripts/benchmark/ocr_compare.py --runs 3
python scripts/benchmark/object_ocr_eval.py
python -m pytest tests/test_object_ocr.py tests/test_object_ocr_pipeline.py
```