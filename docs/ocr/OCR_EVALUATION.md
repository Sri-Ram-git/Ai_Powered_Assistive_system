# OCR Evaluation

Methodology, dataset, and metrics for the object-aware OCR subsystem.  The
goal is *honest, reproducible* measurement — never manufactured numbers.

## Dataset

`scripts/benchmark/object_ocr_eval.py` generates a fully synthetic but
*object-like* dataset of text-bearing regions the user actually encounters:
bottle labels, medicine packs, book covers, laptop screens, door/safety
signs, room-number plates, product codes, and multi-line reading-order
samples.

Every sample is a tuple rendered deterministically from a seeded PRNG
(`zlib.crc32`-based — Python's `hash()` is randomised per process, so the
old version was not reproducible):

    (name, text, kind, box_w, box_h, contrast, font_idx, aug_seed)

- **12 base samples** (degradation-free, kept from the original dataset).
- **16 extra content texts** × `--per-text` (default 6) seeded degradations
  → **100+ total samples** covering:
  - categories: `label`, `screen`, `sign`, `lowlight`;
  - fonts (SIMPLEX / DUPLEX / COMPLEX / TRIPLEX);
  - degradations applied per-sample: rotation −12°..12°, mild perspective
    skew, Gaussian blur (k∈{0,1,2}), lighting gain 0.7..1.3, and small
    sizes (48×24 up to 420×96) that exercise the smart-upscale retry;
  - multi-line texts (`"DO NOT\nWALK"`) that exercise reading-order
    (line grouping + left-to-right) rather than raw detection.

The pipeline under test is the real production path:

    extract_roi -> text-presence gate -> preprocessing variants -> best
    result -> combine

Degradations are applied *inside* the object box; the ground truth text is
never mutated, and the ROI is recomputed from the warped content box, so a
misread is attributable to recognition (not box drift).

## Metrics

Per sample, computed by `src.evaluation.ocr_metrics`:

| Metric               | Definition                                                     |
| -------------------- | -------------------------------------------------------------- |
| `cer`                | Levenshtein over characters ÷ reference character count        |
| `wer`                | Levenshtein over **word tokens** ÷ reference word count        |
| `exact_match`        | 1 if case/whitespace-normalised strings are identical          |
| `accuracy`           | 1 − CER (kept from the original benchmark)                     |
| `order_violation`    | same word *set* but different order (scrambled reading order)  |

Summary aggregates (also in `results.json`):

- mean CER / WER / exact-match / detection-success (via
  `aggregate_ocr_metrics`);
- mean + P95 latency per sample;
- accuracy by contrast and by kind;
- winning preprocessing variant (tells us whether the contrast / adaptive
  variants earn their extra latency).

## Running

```text
python scripts/benchmark/object_ocr_eval.py            # full 100+
python scripts/benchmark/object_ocr_eval.py --limit 20 # quick smoke
```

Outputs to `assets/ocr_eval/`:

- `<name>.png` — scene images (frame + object box) for visual inspection;
- `ground_truth.json` — texts + boxes;
- `results.json` — per-row CER/WER/exact/accuracy/latency/variant/status
  plus the summary block.

## Interpreting

- `status == "ok"` → accuracy ≥ 0.90 (essentially exact).
- `status == "partial"` → some words/characters recovered.
- `status == "miss"` → nothing usable (the "Text unclear" path in the app).
- `presence_gate` / `roi_rejected` → the cheap gate or ROI stage refused
  the region — counted separately, never as a recognition "miss".
- `order_violation` → recognition succeeded but reading order was wrong;
  regression against `_order_and_dedupe` in `src/ocr/ocr_engine.py`.

### Engines compared

When alternative engines are installed, `scripts/benchmark/ocr_engine_compare.py`
runs the same dataset through each engine's adapter and emits the measured
table (see that script).  Absent an installed engine, the table records the
attempt and the blocker honestly.
## Real-typography and mirror samples (accuracy-fix round)

Since the accuracy-fix round the dataset also contains:

- **Font samples** (`font_*`): the user's real-world words and clock
  strings (`12:45 PM`, `10:30 PM`, `08:15`, `23:59`, `12:00 AM`,
  `12:00 PM`, `EMERGENCY EXIT`, `COCA COLA`, `ROOM 204`,
  `MEDICINE 500MG`, `THE ART OF WAR`, `NO SMOKING`) rendered with real
  Windows TrueType fonts via PIL: times (regular/bold/italic), arial
  (regular/bold), calibri, consolas, at small/medium/large sizes.  Rows
  carry `font` and the per-font/per-size numbers are in
  `docs/ocr/OCR_ACCURACY_FIX_REPORT.md` section 5.
- **Mirrored samples** (`mirrored_*`): the same text horizontally flipped,
  simulating the OLD front-camera path.  These are EXPECTED to fail and are
  excluded from the headline metrics (`mirrored: true`); they quantify why
  OCR must receive unmirrored frames.
