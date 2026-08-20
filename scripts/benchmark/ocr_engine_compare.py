"""Cross-engine OCR benchmark: RapidOCR vs installed alternatives.

Runs the *same* deterministic dataset from ``object_ocr_eval`` through each
available engine's adapter with a single plain pass (no preprocessing
variants, no upscale retry — that machinery is engine-agnostic and would
mask raw recognition differences) and emits a measured comparison table.

Engines are loaded lazily; an engine that cannot be imported (e.g. the
PyTorch-based ``easyocr`` download never completed on a slow machine) is
reported honestly rather than silently skipped.

Usage:
    python scripts/benchmark/ocr_engine_compare.py [--per-text 3]
        [--limit 60]
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from object_ocr_eval import (  # noqa: E402
    _augment,
    _canonical,
    _generate_dataset,
    _place_in_frame,
    _render_text,
    _stable_seed,
    char_accuracy,
)
from src.evaluation.ocr_metrics import (  # noqa: E402
    aggregate_ocr_metrics,
    character_error_rate,
    exact_match,
    word_error_rate,
)
from src.ocr.ocr_engine import OcrResult, _axis_aligned_box  # noqa: E402
from src.ocr.roi import extract_roi  # noqa: E402
from src.ocr.text_presence import has_text  # noqa: E402


def _available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


class RapidOcrAdapter:
    name = "rapidocr"

    def __init__(self) -> None:
        from src.ocr import OcrEngine
        self.engine = OcrEngine(min_confidence=0.3)

    def read_text(self, image: np.ndarray):
        return self.engine.read_text(image)


class EasyOcrAdapter:
    name = "easyocr"

    def __init__(self) -> None:
        import easyocr
        self.reader = easyocr.Reader(["en"], gpu=False)

    def read_text(self, image: np.ndarray):
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        raw = self.reader.readtext(image, detail=1, paragraph=False)
        return [OcrResult(
            text=str(r[1]).strip(),
            confidence=float(r[2]),
            box=_axis_aligned_box(r[0]),
        ) for r in raw if str(r[1]).strip()]


class PaddleOcrAdapter:
    name = "paddleocr"

    def __init__(self) -> None:
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)

    def read_text(self, image: np.ndarray):
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".png")
        try:
            cv2.imwrite(path, image)
            raw = self.ocr.ocr(path, cls=False)
        finally:
            import os
            os.close(fd)
            os.remove(path)
        results = []
        for page in (raw or []):
            for box, (text, conf) in (page or []):
                if text and text.strip():
                    results.append(OcrResult(
                        text=str(text).strip(),
                        confidence=float(conf),
                        box=_axis_aligned_box(box),
                    ))
        return results


def _engines():
    engines = [RapidOcrAdapter()]
    if _available("easyocr"):
        engines.append(EasyOcrAdapter())
    else:
        print("[skip] easyocr not installed (PyTorch download may not have "
              "completed); excluded from comparison.")
    if _available("paddleocr"):
        engines.append(PaddleOcrAdapter())
    else:
        print("[skip] paddleocr not installed; excluded from comparison.")
    return engines


def evaluate(engine, sample) -> dict:
    name, gt, kind, box_w, box_h, contrast, font_idx, aug_seed = sample
    truth = _canonical(gt)
    region = _render_text(gt, kind, box_w, box_h, contrast,
                          seed=_stable_seed(name), font_idx=font_idx)
    region, cbox = _augment(region, seed=_stable_seed(name, 1) + aug_seed)
    frame, box = _place_in_frame(region, cbox, seed=_stable_seed(name, 2))

    entry = {"name": name, "truth": truth, "kind": kind,
             "result": "", "accuracy": 0.0, "cer": 1.0, "wer": 1.0,
             "exact": 0, "latency_ms": 0.0, "status": "error",
             "order_violation": False}

    roi = extract_roi(frame, box, padding=0.1)
    if roi is None:
        entry["status"] = "roi_rejected"
        return entry
    if not has_text(roi.image):
        entry["status"] = "presence_gate"
        return entry

    started = time.monotonic()
    try:
        items = engine.read_text(roi.image)
        entry["latency_ms"] = (time.monotonic() - started) * 1000.0
        text = " ".join(r.text for r in items).strip()
        entry["result"] = _canonical(text)
        entry["accuracy"] = char_accuracy(truth, entry["result"])
        entry["cer"] = character_error_rate(truth, entry["result"])
        entry["wer"] = word_error_rate(truth, entry["result"])
        entry["exact"] = exact_match(truth, entry["result"])
        if truth and entry["result"]:
            if (sorted(truth.split()) == sorted(entry["result"].split())
                    and truth != entry["result"]):
                entry["order_violation"] = True
        entry["status"] = "ok" if entry["accuracy"] >= 0.9 else \
            ("partial" if entry["accuracy"] > 0.0 else "miss")
    except Exception as exc:  # pragma: no cover - env dependent
        entry["latency_ms"] = (time.monotonic() - started) * 1000.0
        entry["status"] = f"error:{exc}"
    return entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-text", type=int, default=3)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--out", default="assets/ocr_eval/engine_compare.json")
    args = parser.parse_args()

    samples = _generate_dataset(args.per_text)
    if args.limit > 0:
        samples = samples[:args.limit]

    engines = _engines()
    print(f"OpenCV: {cv2.__version__} | samples: {len(samples)}")
    print(f"engines: {[e.name for e in engines]}\n")

    report = {}
    for engine in engines:
        print(f"=== {engine.name} ===")
        rows = [evaluate(engine, s) for s in samples]
        ev = [r for r in rows if r["status"] not in ("roi_rejected",
                                                     "presence_gate",
                                                     "error")]
        agg = aggregate_ocr_metrics([r["truth"] for r in ev],
                                    [r["result"] for r in ev])
        acc = float(np.mean([r["accuracy"] for r in ev])) if ev else 0.0
        lat = [r["latency_ms"] for r in ev]
        summary = {
            "engine": engine.name,
            "evaluated": len(ev),
            "exact": sum(1 for r in rows if r["status"] == "ok"),
            "partial": sum(1 for r in rows if r["status"] == "partial"),
            "missed": sum(1 for r in rows if r["status"] == "miss"),
            "order_violations": sum(1 for r in rows if r["order_violation"]),
            "mean_char_acc": round(acc, 3),
            "cer": round(agg["cer"], 3),
            "wer": round(agg["wer"], 3),
            "exact_match": round(agg["exact_match"], 3),
            "detection_success": round(agg["detection_success"], 3),
            "mean_latency_ms": round(float(np.mean(lat)), 1) if lat else 0.0,
            "p95_latency_ms": round(float(np.percentile(lat, 95)), 1)
            if lat else 0.0,
        }
        report[engine.name] = summary
        print(json.dumps(summary, indent=2))
        for r in rows:
            if r["status"] == "miss":
                print(f"    miss {r['name']:<20} truth={r['truth']!r:26} "
                      f"got={r['result']!r}")

    print("\n=== COMPARISON TABLE ===")
    hdr = (f"{'engine':<12} {'acc':>5} {'cer':>5} {'wer':>5} "
           f"{'exact':>5} {'det':>5} {'mean_ms':>8} {'p95_ms':>7} "
           f"{'missed':>6} {'order!':>6}")
    print(hdr)
    print("-" * len(hdr))
    for name, s in report.items():
        print(f"{name:<12} {s['mean_char_acc']:>5.2f} {s['cer']:>5.2f} "
              f"{s['wer']:>5.2f} {s['exact_match']:>5.2f} "
              f"{s['detection_success']:>5.2f} {s['mean_latency_ms']:>8.0f} "
              f"{s['p95_latency_ms']:>7.0f} {s['missed']:>6} "
              f"{s['order_violations']:>6}")

    out = Path(PROJECT_ROOT) / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()