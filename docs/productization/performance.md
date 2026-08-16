# Performance

## Where performance matters

This is an **assistive device** (camera + speech).  The UX contract is
not "maximum FPS" — it is:

1. **Low, stable latency** between an event and the spoken response.
2. **Non-blocking** perception — a slow OCR pass must never stall the
   object-detection loop.
3. **Graceful degradation** on weak hardware (Raspberry Pi-class CPU).

Everything below is measured on this repo's baseline environment:
Windows / Python 3.13 / CPU only.

## How to measure

Run the unified benchmark suite — it exercises every stage and writes a
JSON report:

    python performance/benchmarks/run_all.py

Per-stage focused scripts:

    python scripts/benchmark_ocr.py       # OCR latency per strategy
    python scripts/benchmark_stt.py       # speech-command parse latency
    python scripts/optimize_model.py      # detection before/after size + INT8

## Baseline numbers (this machine, CPU)

| Stage          | Median latency | Notes                                    |
|----------------|----------------|------------------------------------------|
| YOLO detect    | ~48-70 ms      | 640 input, OpenCV DNN, CPU (see below)   |
| OCR (RapidOCR) | ~3.4-3.8 s     | inference-bound; independent of strategy |
| Depth (synthetic) | ~6 ms       | placeholder backend, not a real sensor   |
| STT parse (keyword) | < 1 ms     | deterministic parser, no model           |
| TTS enqueue    | < 1 ms         | async queue; playback happens off-loop   |

Full report is committed at `performance/results/benchmark_report.json`.

## What these numbers mean

**YOLO (~50-70 ms @ 640)** is comfortably inside a 5 fps budget.  The
perception loop runs detection every *N* frames (config
`detection.every_n_frames`), so 640 is the right default.

**OCR (~3.5 s) is the long pole.**  It is *inference-bound*, not
preprocessing-bound — every input strategy (`gray`, `threshold`,
`contrast`, `downscale`, `downscale2`) measures the same ~3.4-3.8 s.
This matches the earlier finding that preprocessing does not reduce
latency on this CPU.  **Therefore OCR must never run in the detection
loop.**  The fix is architectural, already in place:

- OCR runs in its own worker thread (`src/ocr/worker.py`), publishing a
  *latest* result.
- The perception loop reads `latest_ocr` if a new one is ready; it never
  waits for OCR.
- Only text that actually changes gets spoken (dedup in the response
  planner).

**TTS is async by design.**  `SpeechOutput` enqueues; the actual
playback latency is dominated by pyttsx3's engine, not our queue.  The
response planner applies a cooldown so speech never stacks.

## Optimization analysis (P12)

### What does NOT apply here — and why

1. **INT8 dynamic quantization** (`onnxruntime.quantization`) is
   **not usable in this pipeline today**:
   - The detector runs through **OpenCV's DNN module**
     (`cv2.dnn.readNetFromONNX`), not onnxruntime.
   - OpenCV DNN cannot execute INT8-quantized graphs.
   - The tooling additionally needs the `onnx` package, which fails to
     install on this Windows setup (path-too-long).  The quantizer in
     `scripts/optimize_model.py` is gated on `onnx` being importable and
     reports a clear SKIP otherwise.
   - When (if) the detector moves to onnxruntime, run
     `python scripts/optimize_model.py` again — the INT8 path is ready.

2. **Smaller input sizes** (416/320) do not work with *this* ONNX
   export: `yolov8n.onnx` is a **fixed-shape (640) export**, and OpenCV
   fails its reshape layer at other sizes.  A dynamic-shape re-export is
   required before input-size tuning is possible.

### What DOES apply today (all config-driven)

| Lever                    | Config key                   | Effect            |
|--------------------------|------------------------------|-------------------|
| Frame skip (detection)   | `detection.every_n_frames`   | 2 => half the YOLO cost |
| Frame skip (OCR)         | `ocr.every_n_frames`         | OCR runs at a fraction of FPS |
| Async workers            | (pipeline design)            | slow stages never block the loop |
| Response cooldown/dedup  | `planner.cooldown_seconds`   | speech never stacks |

## Rules for future optimization work

- **Never optimize blindly.**  Every change must come with a before/after
  measurement and a documented trade-off (accuracy vs latency vs memory).
- Run `python performance/benchmarks/run_all.py` before and after, and
  commit the updated JSON report with the change.
- Keep the safety path deterministic and low-latency.  No LLM/VLM sits on
  the safety-critical decision path.
- If accuracy is not re-measured, the latency claim is not a claim —
  say so in the report.