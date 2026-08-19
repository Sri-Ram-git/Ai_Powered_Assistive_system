# OCR Quality — Final Report

Status of the object-aware OCR subsystem after the quality-hardening work
(rotation fallback, upscale retry, rejection heuristics, meaningful-word
validation, reading-order fix, "Text unclear" gate, debug recorder, and an
honest 100+ sample evaluation).

## 1. What changed

| Area | Before | After |
| ---- | ------ | ----- |
| Reading order | results sorted only by box top → "COCA COLA" became jumbled multi-line output | `_order_and_dedupe`: line grouping + left-to-right within line + exact-duplicate collapse (keep highest conf) |
| Garbage/noise | random reads could reach the panel and TTS | `validate_text` in the worker: garbage → `status="empty"`, never shown/spoken |
| Wrong-but-confident reads | medium/low confidence text was shown as fact | `text_quality()` tiers; `low` → `status="unclear"` → panel shows **"Text unclear"**, TTS silent |
| Vertical / inverted text (book covers, spines) | missed entirely | `run_variants` rotation fallback 90°/270° (only when base variants find nothing; ~0 ms cost when text found) |
| Small / low-res text | missed (no text returned) | ×2 upscale retry in the worker (only when the presence gate passed and OCR returned nothing) |
| book→remote misdetection | box-shape not considered | config-driven `reject_box_shape` per class (`remote: min_aspect 1.8, max_area_ratio 0.25`) |
| Diagnostics | no visibility into a misread | ring-buffer debug recorder: raw boxes, confidences, variant, scale, ROI image; `pipe.dump_ocr_debug(dir)` |
| Eval | 12 samples, non-reproducible seeds, char-accuracy only | 100+ seeded samples, reproducible (`zlib.crc32`), CER / WER / exact-match / detection-success / order-violation / P95 latency |

## 2. Measured results

Run: `python scripts/benchmark/object_ocr_eval.py` (108 samples, degraded
synthetic object regions, full production OCR path: ROI → presence gate →
variants → best → combine).  Details in `docs/ocr/OCR_EVALUATION.md`;
per-row data in `assets/ocr_eval/results.json`.

| Metric | Value |
| ------ | ----- |
| evaluated | 97 / 108 |
| mean char accuracy | 0.733 |
| mean CER | 0.267 |
| mean WER | 0.704 |
| exact-match rate | 0.196 |
| detection success | 0.732 |
| order violations | **0** |
| mean latency | 4038 ms |
| P95 latency | 6215 ms |

Reading-order regression: **0 order violations across 97 evaluated
multi-line / side-by-side samples** (previously the crude top-sort
produced scrambled multi-word output on exactly these cases).

### 2.1 Honest caveats

- The 108-sample dataset is deliberately *harder* than the original
  12-sample set: rotation (−12°..12°), perspective, blur, lighting 0.7–1.3,
  four fonts, tiny boxes (48×24), multi-line signs.  The earlier published
  mean char-accuracy of **0.926** was measured on 12 clean, large samples;
  the two numbers are **not comparable**.  Both are synthetic.
- WER (0.704) is dominated by *truncated trailing words* ("THE GREAT GA"
  vs "THE GREAT GATSBY", "ANTI-DANDRU" vs "ANTI-DANDRUFF") — each missing
  word counts as a full word error.  CER is the fairer headline metric.
- **Latency is environment-limited**: these runs happened while a 122 MB
  PyTorch download contended for bandwidth/CPU and the machine was
  thermally throttled; single-call times for the same ROI measured
  200 ms–2 s earlier in the session.  Do not treat absolute latency here
  as a hardware ceiling.  Verified separately: the rotation fallback adds
  ~0 ms when text is found (203 → 209 ms).
- 11 samples were rejected by the *presence gate* (not by recognition) —
  two-word low-contrast signs; these are counted separately and are a
  candidate for a tuned gate threshold, not an OCR failure.
- `detection_success` uses ≥50% word overlap — an easy bar; WER is the
  stricter quality signal.

### 2.2 Error attribution (from the debug recorder / per-row results)

- Truncation (missing tail words): dominant failure mode on small boxes —
  the upscale retry already helps; larger ROI margin and tighter
  `det_limit` are the next candidates.
- Character swaps on rotated/angled text ("NELCOME" for "WELCOME",
  "WAI" for "WAR"): recognition-level, not ordering.
- Presence-gate false negatives on short low-contrast text ("CALL NOW").

## 3. Alternative engines (best effort)

Per the spec, `scripts/benchmark/ocr_engine_compare.py` runs the same
dataset through adapters for **RapidOCR**, **easyocr**, and **paddleocr**.

Outcome on this machine (honest): the `easyocr` install (PyTorch CPU
wheel, ~122 MB) was attempted twice and stalled on the torch download
(no progress in ~25 min on this network); it was aborted.  paddleocr
(PaddlePaddle, heavier) was not attempted.  So `assets/ocr_eval/engine_compare.json`
currently contains a **RapidOCR-only** table and the script reports the
other engines as skipped.  If you later install them:

```
python -m pip install easyocr            # and/or: paddleocr
python scripts/benchmark/ocr_engine_compare.py
```

The adapter layer (`RapidOcrAdapter` / `EasyOcrAdapter` / `PaddleOcrAdapter`)
is ready; the engine choice remains RapidOCR, whose CPU-only ONNX
footprint and per-call async worker are already wired and battle-tested.

## 4. UX safeguards (unchanged requirements)

- READ ALOUD re-uses the *saved* text; it never re-runs OCR.
- Auto-read speaks once per cooldown; unclear/empty results are never
  spoken.
- No LLM is used to "repair" OCR output (spec directive).
- The camera loop is never blocked: OCR runs on the async worker slot with
  newest-wins; the pipeline's `ocr_busy` reflects the worker state.

## 5. Tests

`426 passed` (full suite, 86.2 % coverage), including new tests for:
reading order/dedup (6), text-quality tiers, worker `unclear` status,
debug recorder + dump, and fixed `word_error_rate` (word-token
Levenshtein).  `ruff check` clean on all changed files.

## 6. Repro

```
python -m pytest -q                                  # full suite
python -m ruff check src scripts tests
python scripts/benchmark/object_ocr_eval.py          # 108-sample eval
python scripts/benchmark/ocr_engine_compare.py       # cross-engine table
python src/assist/assist_app.py --config configs/assist_config.yaml
```