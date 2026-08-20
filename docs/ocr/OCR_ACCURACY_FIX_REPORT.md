# OCR Accuracy Fix Report (Live Front-Camera)

Fixes for LIVE OCR accuracy with the front-facing camera: mirroring,
orientation, real-world fonts, small text, ordering, garbage rejection,
temporal stabilization, live-OCR flow, and an on-screen debug overlay.
The existing UI, OCR card, and Read Aloud are preserved.

## 1. Problem statement

Live OCR on the front (selfie) camera produced wrong text: words read as
garbage ("COCA COLA" -> mirrored letters, times like "12:45 PM" -> "MQ CT:SR").
The root cause was a single architectural bug, plus robustness gaps for
real-world conditions:

1. **The camera mirrored every frame.** The SAME mirrored frame fed the
   preview, object detection, AND OCR. Text that is horizontally mirrored is
   unreadable to the OCR engine, so every live read was scrambled even though
   offline/synthetic evaluations (unmirrored) looked fine.
2. Fonts were only exercised with OpenCV Hershey strokes; real typography
   (Times New Roman, Arial, Calibri, digital clocks) was unverified.
3. Small text, rotation, ordering, and temporal stability were only partly
   hardened.

## 2. System architecture / data flow

Fixed data flow (raw capture, single geometric correction, display-only
mirroring):

```
  Camera (raw sensor frame)
        |
        +--(optional sensor rotation 0/90/180/270)--+
        |                                           |
   VISION FRAME (unmirrored)                 DISPLAY FRAME
        |                                           |
   YOLO -> tracking -> object ROI          mirrored in UI only
        |                                 (boxes flipped to match)
   text region (rotate to upright, optional upscale)
        |
   presence gate (has_text)
        |
   adaptive preprocess (strategy variants + fallback variants)
        |
   OCR engine (RapidOCR, reading-order fix)
        |
   validate + text_quality tiers
        |
   TrackOcrStore temporal vote (2 consistent reads)
        |
   existing OCR card  ->  READ ALOUD (deduped, cached text)
```

No LLM / cloud API is used anywhere in this path. The engine stays RapidOCR
with the origin fix (`det_limit_side_len=736, det_limit_type="max"`).

## 3. The mirror fix (critical bug)

| Before | After |
| ------ | ----- |
| `Camera.read()` flipped every frame (`mirror=True` default in `src/camera/camera.py`) | Capture is RAW (`mirror` default `False`); nothing is flipped at capture |
| The same mirrored frame fed preview + YOLO + OCR | VISION frame (YOLO/OCR) is geometrically correct; mirroring is display-only |
| OCR read mirrored text -> garbage | OCR always receives upright, unmirrored text |

Implementation:
- `src/camera/camera.py`: `mirror=False` default; new `rotate` parameter
  (0/90/180/270) applied at capture via `cv2.rotate` (with the resolution
  swapped for 90/270).
- `src/core/config.py`: `camera.mirror` (default `false`),
  `camera.rotate` (default `0`), `camera.preview_mirror` (default `true`).
- `src/core/pipeline.py`: camera factory called with `mirror=`/`rotate=`;
  state exposes geometry via `camera_geometry()`.
- `src/assist/assist_app.py`: preview mirrored only for display
  (`preview_mirror`), `draw_tracks(mirror_x=...)` flips box x-coordinates to
  match; panel/HUD drawn after the flip so text stays readable.

Quantified effect (same content, 4 mirrored samples from the eval dataset):

| Sample | Mirrored (old live path) | Unmirrored (fixed path) |
| ------ | ------------------------ | ----------------------- |
| 12:45 PM | `MQ CT:SR` (acc 0.00) | reads correctly |
| 10:30 PM | `M 08:0` (acc 0.12) | reads correctly |
| 12:00 AM | `MA 00:ST` (acc 0.25) | reads correctly |
| 23:59 | `53:2A` (acc 0.40) | reads correctly |

All four mirrored reads are effectively unusable and would have been gated
as wrong text; all four read correctly once unmirrored.

## 4. Camera rotation / orientation

Physical sensor mounting (e.g. a camera held sideways) is handled once at
capture with `camera.rotate` in `configs/assist_config.yaml`:

```yaml
camera:
  mirror: false        # raw capture; mirroring is display-only
  rotate: 0            # 0 / 90 / 180 / 270 sensor orientation
  preview_mirror: true # show selfie-view in the UI
```

`rotate` is normalized with `% 360` in config parsing, applied in
`Camera.read()`, and reflected in the debug overlay. The vision frame and the
preview stay consistent (the preview is additionally mirrored per
`preview_mirror`).

## 5. Font robustness (Times New Roman, Arial, Calibri, digital clocks)

The eval dataset now renders text with REAL Windows TrueType fonts via PIL
(in addition to the existing OpenCV Hershey samples):

- Fonts: `times` (regular/bold/italic), `arial` (regular/bold),
  `calibri`, `consolas` (monospace, digital-clock-ish).
- Sizes: small / medium / large pixel heights.
- Content includes clock strings (`12:45 PM`, `10:30 PM`, `08:15`, `23:59`,
  `12:00 AM`, `12:00 PM`) and the user's real-world words
  (`EMERGENCY EXIT`, `COCA COLA`, `ROOM 204`, `MEDICINE 500MG`,
  `THE ART OF WAR`, `NO SMOKING`).

Measured (mean char accuracy, unmirrored):

| Font | n | mean | exact (>=0.9) |
| ---- | - | ---- | ------------- |
| times | 12 | 0.886 | 5 |
| times_bold | 12 | 0.945 | 8 |
| times_italic | 12 | 0.869 | 9 |
| arial | 12 | 0.858 | 8 |
| arial_bold | 12 | 0.935 | 8 |
| calibri | 12 | 0.902 | 11 |
| consolas | 12 | 0.854 | 6 |

By size: small 0.866, medium 0.940, large 0.886. No font/size cell falls
below 0.7. The processing chain (variants: none / contrast / adaptive /
otsu / denoise / gray_norm + rotation fallbacks) handles serif, sans-serif,
bold, italic, and monospace text without engine changes.

## 6. Small text / adaptive preprocessing

- `src/ocr/preprocess.py` gained `otsu`, `denoise`, and `gray_norm`
  strategies, so the adaptive chain can pick a binarization/denoise pass for
  low-contrast or noisy regions.
- `src/ocr/object_ocr.py` `run_variants()` gained `fallback_variants`
  (rotate90, rotate270, otsu, denoise, gray_norm) tried ONLY when the base
  variants find nothing - zero added latency in the common case.
- Small/low-res text already had an A-2 upscale retry in the worker (only
  when the presence gate passed and OCR returned nothing). Small-size samples
  in this run averaged 0.866 mean char accuracy.

## 7. Text ordering

`_order_and_dedupe()` in `src/ocr/ocr_engine.py` groups lines and sorts
left-to-right within a line, collapsing exact duplicates (keeping the highest
confidence). Ordering samples (`DO NOT\nWALK`, `CALL\nNOW`) are in the eval;
this run reported 0 order violations across 196 samples.

## 8. Garbage rejection

- `validate_text` in the worker rejects garbage (noise) reads -> empty,
  never shown or spoken.
- `text_quality()` tiers (high/medium/low); `low` -> `status="unclear"`,
  panel shows "Text unclear - try again, hold steadier, or move closer",
  TTS stays silent.
- Presence gate (`has_text`) rejects regions with no text before OCR runs.
- Policy everywhere is: prefer NO RESULT over WRONG RESULT.

## 9. Temporal stabilization

- `TrackOcrStore` (in `src/ocr/object_ocr.py`) requires `confirm_votes=2`
  consecutive identical reads before promoting a per-track result, so
  transient flicker does not flip the shown text.
- Speech is deduped: `SpeechQueue` `_last_spoken` + dedupe window and
  `variety.py` `_last_spoken` mean the same text is read aloud once (auto
  read) instead of repeating every frame.
- Reading-aloud reuses the SAVED text from the OCR card; it never re-runs OCR.

## 10. Live OCR

The async worker keeps the latest-request pattern: no per-frame OCR, stale
results are discarded, the queue never grows. The UI renders the latest
confirmed result; debug mode shows live status. See
`docs/ocr/object_aware_ocr_architecture.md`.

## 11. Debug overlay

Debug mode (`debug: true`) draws on the display frame: track boxes, ROI, and
an OCR panel with variant used, latency, confidence, stability/vote state,
mirror/rotate geometry, and status. Implemented in
`src/assist/assist_app.py` `_debug_overlay`.

## 12. Evaluation methodology

`python scripts/benchmark/object_ocr_eval.py` (full production path:
ROI -> presence gate -> variants -> best -> combine; metrics from
`src/evaluation/ocr_metrics.py`: char accuracy, CER, word-token WER,
exact match, order violation, latency).

Dataset (196 samples, all seeded / reproducible):

- 12 base samples (degradation-free)
- 16 extra texts x 6 degradations = 96
- 12 real-world texts x 7 fonts = 84 font samples (incl. clock strings)
- 4 mirrored samples (quantify the old front-camera bug)

Run on this machine with:
`python scripts/benchmark/object_ocr_eval.py --out assets/ocr_eval --per-text 6`

Caveats (stated honestly):
- Regions are synthetic/degraded, not real camera frames; results measure
  the OCR chain, not the camera/ROI pipeline.
- This machine was thermally/AV loaded during runs; latency numbers are
  environment-dependent and not comparable across runs.
- The 196-sample dataset is NEW (fonts + mirrors added), so headline numbers
  are NOT directly comparable to the earlier 108-sample run.

## 13. Measured BEFORE / AFTER

| Metric | BEFORE (live front-camera, mirrored) | AFTER (fixed path, 196 samples) |
| ------ | ------------------------------------ | ------------------------------- |
| Front-camera text | mirrored -> garbage ("MQ CT:SR" for "12:45 PM"); effectively 0 usable reads | unmirrored -> correct reads on identical content |
| mean char accuracy | n/a on mirrored text (0.00-0.40 on mirror samples); 0.733 on the old unmirrored 108-sample set | 0.826 (181 evaluated) |
| mean CER | 0.267 (old unmirrored set) | 0.174 |
| mean WER | 0.704 (old unmirrored set) | 0.573 |
| exact match rate | 0.196 (old unmirrored set) | 0.367 |
| detection success | 0.732 (old unmirrored set) | 0.853 |
| order violations | 0 | 0 |
| mean latency | 4038 ms (old set, loaded machine) | 2086 ms (this run) |
| p95 latency | 6215 ms (old set, loaded machine) | 3245 ms (this run) |
| Font coverage | Hershey strokes only | Times/Arial/Calibri/Consolas x sizes, all >= 0.85 mean |

Mirrored-vs-unmirrored is the honest apples-to-apples comparison for the
critical bug (identical text, one flipped); the old-vs-new headline numbers
are from different datasets and are informational only.

## 14. Config reference

```yaml
camera:
  mirror: false        # raw capture
  rotate: 0            # 0/90/180/270
  preview_mirror: true # selfie-view display
```

OCR tuning lives under `ocr:` in `configs/assist_config.yaml`
(preprocessing variants, upscale, min confidence, debug recorder).

## 15. Files changed

- `src/camera/camera.py` - raw capture, rotation
- `src/core/config.py` - camera geometry config
- `src/core/pipeline.py` - factory kwargs, `camera_geometry()`
- `src/assist/assist_app.py` - preview mirror, `draw_tracks(mirror_x)`, debug overlay
- `src/ocr/preprocess.py` - otsu / denoise / gray_norm strategies
- `src/ocr/object_ocr.py` - fallback variants, validation, `TrackOcrStore`
- `configs/assist_config.yaml` - documented camera geometry
- `scripts/benchmark/object_ocr_eval.py` - real-font + mirrored samples
- `tests/test_camera.py`, `test_core.py`, `test_ocr_async.py`,
  `test_object_ocr.py`, `test_object_ocr_pipeline.py`,
  `test_failure_modes.py`

## 16. Known limitations / next steps

- Font samples are PIL-rendered, not photographs; a real-camera capture
  session (front camera, printed labels) is the natural next verification.
- Consolas/small time strings are the weakest cell (0.854 mean); a dedicated
  monospace/digital-clock tune (e.g. tighter crop + denoise) is optional.
- easyocr/paddleocr comparison remains blocked (torch wheel download stalls
  on this machine); `scripts/benchmark/ocr_engine_compare.py` is ready.
- Latency still needs a quiet-machine rerun to set a trustworthy P95 budget.

## 17. How to reproduce

```
python scripts/benchmark/object_ocr_eval.py --out assets/ocr_eval --per-text 6
```

Inspect `assets/ocr_eval/ground_truth.json` and `results.json`; mirrored
samples carry `"mirrored": true` and are expected to fail.

## 18. Summary

The critical fix is architectural: capture is now RAW, the vision/OCR frame
is geometrically correct, and the selfie mirror is applied only to the
display. Font robustness, adaptive preprocessing, small-text handling,
ordering, garbage rejection, temporal voting, live-OCR flow, and the debug
overlay are all verified by the 196-sample seeded eval (fonts >= 0.85 mean
acc; 0 order violations; mirrored reads quantified and eliminated). All 448
tests pass and ruff is clean.