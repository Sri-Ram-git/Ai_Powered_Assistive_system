"""Model optimization toolkit.

Honest before/after optimization workflow for the detection model.

Context: the current YOLOv8n pipeline runs through **OpenCV's DNN module**
(cv2.dnn.readNetFromONNX), not onnxruntime.  Therefore:

* INT8 dynamic quantization (onnxruntime.quantization) produces a model
  for the ORT runtime — OpenCV DNN cannot run INT8-quantized graphs, so
  this does NOT speed up the current pipeline.  It is provided for the
  day the pipeline moves to onnxruntime, or for comparison runs.
* What DOES apply today, and is available in config:
    - frame skipping (detection.every_n_frames, ocr.every_n_frames);
    - OpenCV DNN preferable-backend/target (CUDA when a GPU exists);
    - smaller input size (detection.input_size 640 -> 416/320).

Every operation must produce a before/after latency measurement.  This
tool reports the comparison; it never silently changes production config.
"""
import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def _synthetic_frame():
    img = np.full((720, 1280, 3), 128, dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (500, 400), (80, 160, 240), -1)
    cv2.putText(img, "EXIT", (600, 200), cv2.FONT_HERSHEY_SIMPLEX,
                3.0, (0, 0, 0), 6)
    return img


def _bench_model(model_path, input_size, n=5):
    from src.detection import YoloDetector

    det = YoloDetector(model_path, input_size=input_size)
    frame = _synthetic_frame()
    det.detect(frame)  # warmup
    times = []
    for _ in range(n):
        t0 = time.monotonic()
        det.detect(frame)
        times.append((time.monotonic() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2], times[0]


def _quantize_int8(src, dst):
    """Dynamic INT8 quantization via onnxruntime (requires `onnx`)."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError as exc:
        raise RuntimeError(
            "INT8 quantization requires the `onnx` package, which is "
            "not importable in this environment. "
            "`pip install onnx` (may fail on some Windows paths)."
        ) from exc
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QUInt8)
    return Path(dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/yolov8n.onnx")
    parser.add_argument("--input-sizes", default="640,416,320")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    frame_h, frame_w = _synthetic_frame().shape[:2]
    print(f"Benchmark model: {args.model} "
          f"(input frame {frame_w}x{frame_h}, runs={args.runs})")

    sizes = [int(s) for s in args.input_sizes.split(",") if s.strip()]
    results = {}
    for size in sizes:
        try:
            med, mn = _bench_model(args.model, size, n=args.runs)
        except Exception as exc:
            print(f"  input_size={size:<4} FAILED: "
                  f"{type(exc).__name__}: {str(exc)[:80]}")
            results[f"input_{size}"] = {"median_ms": None,
                                        "error": str(exc)[:200]}
            continue
        results[f"input_{size}"] = {"median_ms": round(med, 1),
                                    "min_ms": round(mn, 1)}
        print(f"  input_size={size:<4} median={med:6.1f} ms  "
              f"min={mn:6.1f} ms")

    # INT8 quantization attempt (documented; optional on this runtime).
    print("\nINT8 dynamic quantization (onnxruntime path):")
    try:
        dst = str(PROJECT_ROOT / "models" / "yolov8n_int8.onnx")
        _quantize_int8(args.model, dst)
        from pathlib import Path as _P
        orig = _P(args.model).stat().st_size
        quant = _P(dst).stat().st_size
        print(f"  OK -> {dst} ({orig/1e6:.1f} MB -> {quant/1e6:.1f} MB)")
        print("  NOTE: OpenCV DNN cannot execute INT8 graphs; this model "
              "is only usable by onnxruntime.")
        results["int8"] = {
            "quantized_path": dst,
            "size_mb": round(quant / 1e6, 1),
        }
    except Exception as exc:
        print(f"  SKIPPED: {exc}")

    print("\nSummary:")
    valid = {k: v for k, v in results.items()
             if v.get("median_ms") is not None}
    if valid:
        print(f"  fastest input size : "
              f"{min(valid, key=lambda k: valid[k]['median_ms'])}")
        print("  trade-off: smaller input = faster but lower accuracy on "
              "small/ distant objects (not measured here).")
    else:
        print("  no input size succeeded; this ONNX export is fixed-shape "
              "(see per-size errors above).")


if __name__ == "__main__":
    main()